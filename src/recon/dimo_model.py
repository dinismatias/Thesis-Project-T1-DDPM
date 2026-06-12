# src/recon/dimo_model.py
"""DiMo/DDPM model with stable API across the project.

This version preserves the architecture/state-dict names of your current model
while adding the compatibility needed by training, sampling, and denoising scripts:

- Constructor accepts both T=... and timesteps=...
- Exposes model.T, model.timesteps, model.alpha_bars, model.betas
- Exposes model.q_sample(...), backed by model.schedule.q_sample(...)
- Conditional forward requires cond when cond_ch > 0, preventing silent misuse
- load_model accepts checkpoint schemas: model_state, state_dict, model, or raw state dict
- load_model can infer T/data_ch/cond_ch from checkpoint when possible
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Diffusion schedule
# -----------------------------------------------------------------------------


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(float(beta_start), float(beta_end), int(T), dtype=torch.float32)


class DiffusionSchedule(nn.Module):
    def __init__(self, T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        betas = make_beta_schedule(T, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    @property
    def T(self) -> int:
        return int(self.betas.numel())

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        t = t.long().to(x0.device)
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1).to(device=x0.device, dtype=x0.dtype)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0)) * noise


# -----------------------------------------------------------------------------
# UNet backbone
# -----------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, time_dim: int = 128):
        super().__init__()
        self.time_dim = int(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        if t.ndim == 0:
            t = t[None]
        half_dim = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / float(max(half_dim, 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.time_dim:
            emb = F.pad(emb, (0, self.time_dim - emb.shape[-1]))
        return self.mlp(emb)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: Optional[int] = None):
        super().__init__()
        num_groups = 8 if int(out_ch) % 8 == 0 else 1
        self.conv1 = nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, int(out_ch))
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, int(out_ch))
        self.act2 = nn.SiLU()
        self.time_mlp = nn.Linear(int(time_dim), int(out_ch)) if time_dim is not None else None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.gn1(self.conv1(x))
        if temb is not None and self.time_mlp is not None:
            h = h + self.time_mlp(temb)[:, :, None, None]
        h = self.act1(h)
        h = self.act2(self.gn2(self.conv2(h)))
        return h


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base_ch: int = 64,
        time_dim: int = 128,
        use_time: bool = True,
    ):
        super().__init__()
        self.use_time = bool(use_time)
        self.time_dim = int(time_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_dim=self.time_dim) if self.use_time else None
        tdim = self.time_dim if self.use_time else None

        self.down1 = ConvBlock(in_ch, base_ch, tdim)
        self.down2 = ConvBlock(base_ch, base_ch * 2, tdim)
        self.down3 = ConvBlock(base_ch * 2, base_ch * 4, tdim)
        self.pool = nn.MaxPool2d(2)
        self.up2 = ConvBlock(base_ch * 4 + base_ch * 2, base_ch * 2, tdim)
        self.up1 = ConvBlock(base_ch * 2 + base_ch, base_ch, tdim)
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_time:
            if t is None:
                raise ValueError("t must be provided when use_time=True")
            temb = self.time_embed(t)
        else:
            temb = None

        x1 = self.down1(x, temb)
        x2 = self.down2(self.pool(x1), temb)
        x3 = self.down3(self.pool(x2), temb)

        # Use explicit skip sizes instead of scale_factor so odd shapes are safer.
        u2 = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, x2], dim=1), temb)
        u1 = F.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, x1], dim=1), temb)
        return self.out_conv(u1)


# -----------------------------------------------------------------------------
# DDPM wrapper
# -----------------------------------------------------------------------------


class DiMoDDPM(nn.Module):
    def __init__(
        self,
        data_ch: int = 2,
        cond_ch: int = 0,
        T: Optional[int] = None,
        *,
        timesteps: Optional[int] = None,
        base_ch: int = 64,
        time_dim: int = 128,
        use_time: bool = True,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        if T is None:
            T = timesteps if timesteps is not None else 1000
        self.data_ch = int(data_ch)
        self.cond_ch = int(cond_ch)
        self.timesteps = int(T)
        self.base_ch = int(base_ch)
        self.time_dim = int(time_dim)
        self.use_time = bool(use_time)

        self.unet = SimpleUNet(
            in_ch=self.data_ch + self.cond_ch,
            out_ch=self.data_ch,
            base_ch=self.base_ch,
            time_dim=self.time_dim,
            use_time=self.use_time,
        )
        self.schedule = DiffusionSchedule(T=self.timesteps, beta_start=beta_start, beta_end=beta_end)

    @property
    def T(self) -> int:
        return self.timesteps

    @property
    def betas(self) -> torch.Tensor:
        return self.schedule.betas

    @property
    def alphas(self) -> torch.Tensor:
        return self.schedule.alphas

    @property
    def alpha_bars(self) -> torch.Tensor:
        return self.schedule.alpha_bars

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.schedule.q_sample(x0=x0, t=t, noise=noise)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_t.ndim != 4:
            raise ValueError(f"x_t must be [B,C,H,W], got shape {tuple(x_t.shape)}")
        if x_t.shape[1] != self.data_ch:
            raise ValueError(f"x_t has {x_t.shape[1]} channels, expected data_ch={self.data_ch}")

        if self.cond_ch == 0:
            if cond is not None and cond.numel() > 0:
                # Ignore empty/None only; real conditioning with unconditional model is likely a user mistake.
                raise ValueError("cond was passed but model was created with cond_ch=0")
            x_in = x_t
        else:
            if cond is None:
                raise ValueError(f"Model expects cond_ch={self.cond_ch}, but cond=None was passed")
            if cond.ndim != 4:
                raise ValueError(f"cond must be [B,C,H,W], got shape {tuple(cond.shape)}")
            if cond.shape[0] != x_t.shape[0] or cond.shape[2:] != x_t.shape[2:]:
                raise ValueError(f"cond shape {tuple(cond.shape)} incompatible with x_t shape {tuple(x_t.shape)}")
            if cond.shape[1] != self.cond_ch:
                raise ValueError(f"cond has {cond.shape[1]} channels, expected cond_ch={self.cond_ch}")
            x_in = torch.cat([x_t, cond.to(dtype=x_t.dtype, device=x_t.device)], dim=1)

        return self.unet(x_in, t)


# -----------------------------------------------------------------------------
# Checkpoint utilities
# -----------------------------------------------------------------------------


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def extract_state_and_config(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Extract a state dict and config from the checkpoint formats used in this project."""
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif all(isinstance(k, str) for k in ckpt.keys()) and any(k.startswith("unet.") or k.startswith("schedule.") for k in ckpt.keys()):
            state = ckpt
        else:
            raise ValueError(
                "Unrecognized checkpoint dict. Expected one of keys: model_state, state_dict, model, "
                "or a raw state_dict with unet./schedule. keys."
            )
        cfg = ckpt.get("config", ckpt.get("args", {}))
        if not isinstance(cfg, dict):
            try:
                cfg = vars(cfg)
            except Exception:
                cfg = {}
        return _strip_module_prefix(state), cfg
    raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")


def infer_model_dims_from_state(state: Dict[str, torch.Tensor]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (T, data_ch, cond_ch) when inferable from a state dict."""
    T = None
    for k in ("schedule.alpha_bars", "schedule.betas", "alpha_bars", "betas"):
        if k in state:
            T = int(state[k].numel())
            break

    data_ch = None
    cond_ch = None
    out_w = state.get("unet.out_conv.weight")
    in_w = state.get("unet.down1.conv1.weight")
    if out_w is not None:
        data_ch = int(out_w.shape[0])
    if in_w is not None and data_ch is not None:
        cond_ch = int(in_w.shape[1]) - int(data_ch)
    return T, data_ch, cond_ch


def load_model(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    data_ch: Optional[int] = None,
    cond_ch: Optional[int] = None,
    T: Optional[int] = None,
    *,
    timesteps: Optional[int] = None,
    base_ch: int = 64,
    strict: bool = True,
    verbose: bool = True,
) -> DiMoDDPM:
    """Load a DiMoDDPM checkpoint safely.

    Parameters left as None are inferred from checkpoint config/state when possible.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state, cfg = extract_state_and_config(ckpt)
    inferred_T, inferred_data_ch, inferred_cond_ch = infer_model_dims_from_state(state)

    if T is None:
        T = timesteps
    if T is None:
        T = cfg.get("T", cfg.get("timesteps", inferred_T))
    if data_ch is None:
        data_ch = cfg.get("data_ch", inferred_data_ch if inferred_data_ch is not None else 2)
    if cond_ch is None:
        cond_ch = cfg.get("cond_ch", inferred_cond_ch if inferred_cond_ch is not None else 0)

    if T is None:
        raise ValueError("Could not infer diffusion timesteps T from checkpoint; pass T=... explicitly")

    model = DiMoDDPM(data_ch=int(data_ch), cond_ch=int(cond_ch), T=int(T), base_ch=int(base_ch))
    result = model.load_state_dict(state, strict=bool(strict))
    if verbose:
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")
        print(f"[INFO] timesteps={model.T} data_ch={model.data_ch} cond_ch={model.cond_ch}")
        if not strict:
            print(f"[INFO] missing_keys={result.missing_keys}")
            print(f"[INFO] unexpected_keys={result.unexpected_keys}")

    model.to(torch.device(device))
    model.eval()
    return model
