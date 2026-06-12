# src/test/run_denoise.py
"""
Stage-1 DiMo/DDPM denoising script, checkpoint-compatible and conditional-aware.

What this script does
---------------------
It performs image-to-image DDIM-style denoising WITHOUT data consistency.
This is useful as:
  1) a diagnostic of the learned diffusion prior;
  2) an optional initialization generator for Stage-2 reconstruction;
  3) a controlled comparison against the zero-filled image.

Important scientific caveat
---------------------------
A zero-filled MRI image contains structured aliasing, not Gaussian noise. DDPM
Stage-1 denoising may reduce speckle or smooth artifacts, but it should not be
expected to reliably recover missing k-space by itself. The reconstruction step
that enforces measured k-space is Stage-2 sampling with DC.

Typical use from Stage-2 bundle/output
--------------------------------------
python -m src.test.run_denoise ^
  --ckpt "checkpoints/dimo_cond_r4/epoch_0020.pt" ^
  --input_pt "outputs/stage2_run/stage2_recon_output.pt" ^
  --input_key zf_2ch ^
  --strength 0.3 ^
  --num_steps 25 ^
  --out_dir "outputs/stage1_run"

Typical use directly from dataset
---------------------------------
python -m src.test.run_denoise ^
  --ckpt "checkpoints/dimo_cond_r4/epoch_0020.pt" ^
  --acc_root "...\\AccFactor04" ^
  --acc_factor 04 ^
  --index 0 ^
  --input_mode zf ^
  --strength 0.3 ^
  --num_steps 25 ^
  --out_dir "outputs/stage1_run"

Then feed Stage-1 to Stage-2
----------------------------
python -m src.test.run_sampling ^
  --acc_root "...\\AccFactor04" ^
  --index 0 ^
  --ckpt "checkpoints/dimo_cond_r4/epoch_0020.pt" ^
  --cond_mode zf_mask ^
  --init_mode pt ^
  --init_pt "outputs/stage1_run/denoise_output.pt" ^
  --init_key x_out_2ch ^
  --stage2_strength 0.05 ^
  --num_steps 50 ^
  --dc_mode replace ^
  --out_dir "outputs/stage2_from_stage1"
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import torch

try:
    from src.recon.dimo_model import DiMoDDPM
    from src.recon.dimo_dataset import DimoKspaceDataset
    from src.recon.encoding import ifft2c
    from src.utils.complex_ops import twoch_to_complex, complex_to_twoch
except Exception:  # pragma: no cover
    from dimo_model import DiMoDDPM  # type: ignore
    from dimo_dataset import DimoKspaceDataset  # type: ignore
    from encoding import ifft2c  # type: ignore
    from complex_ops import twoch_to_complex, complex_to_twoch  # type: ignore


# -----------------------------------------------------------------------------
# Checkpoint/model compatibility
# -----------------------------------------------------------------------------


def _namespace_to_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "__dict__"):
        return dict(vars(x))
    return {}


def _strip_module_prefix(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = dict(state)
    if any(k.startswith("module.") for k in out.keys()):
        out = {k.replace("module.", "", 1): v for k, v in out.items()}
    return out


def _load_ckpt_state_and_meta(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise ValueError("Checkpoint model state is not dict-like.")

    cfg = _namespace_to_dict(ckpt.get("config"))
    legacy_args = _namespace_to_dict(ckpt.get("args"))
    return _strip_module_prefix(state), cfg, legacy_args


def _infer_timesteps(state: Dict[str, torch.Tensor], cfg: Dict[str, Any], legacy_args: Dict[str, Any], fallback: int) -> int:
    for key in ("schedule.alpha_bars", "schedule.betas", "alpha_bars", "betas"):
        if key in state and hasattr(state[key], "shape") and len(state[key].shape) >= 1:
            return int(state[key].shape[0])
    for src in (cfg, legacy_args):
        for key in ("timesteps", "T"):
            if key in src and src[key] is not None:
                return int(src[key])
    return int(fallback)


def _infer_cond_mode(user_cond_mode: str, user_cond_ch: Optional[int], cfg: Dict[str, Any], legacy_args: Dict[str, Any]) -> str:
    if user_cond_mode != "auto":
        return user_cond_mode
    for src in (cfg, legacy_args):
        value = src.get("cond_mode")
        if isinstance(value, str) and value in {"none", "zf", "zf_mask"}:
            return value
    probe = user_cond_ch
    if probe is None:
        probe = cfg.get("cond_ch", legacy_args.get("cond_ch", None))
    if probe in (0, "0", None):
        return "none"
    if probe in (2, "2"):
        return "zf"
    if probe in (3, "3"):
        return "zf_mask"
    raise ValueError(f"Cannot infer cond_mode from cond_ch={probe!r}. Pass --cond_mode explicitly.")


def _infer_cond_ch(cond_mode: str, user_cond_ch: Optional[int], cfg: Dict[str, Any], legacy_args: Dict[str, Any]) -> int:
    if user_cond_ch is not None:
        return int(user_cond_ch)
    for src in (cfg, legacy_args):
        value = src.get("cond_ch")
        if value is not None:
            return int(value)
    if cond_mode == "none":
        return 0
    if cond_mode == "zf":
        return 2
    if cond_mode == "zf_mask":
        return 3
    raise ValueError(f"Unknown cond_mode: {cond_mode}")


def _make_model(data_ch: int, cond_ch: int, timesteps: int) -> torch.nn.Module:
    try:
        return DiMoDDPM(data_ch=data_ch, cond_ch=cond_ch, T=timesteps)
    except TypeError:
        return DiMoDDPM(data_ch=data_ch, cond_ch=cond_ch, timesteps=timesteps)


def _get_alpha_bars(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    if hasattr(model, "alpha_bars"):
        return getattr(model, "alpha_bars").to(device)
    if hasattr(model, "schedule") and hasattr(model.schedule, "alpha_bars"):
        return model.schedule.alpha_bars.to(device)
    raise AttributeError("Model does not expose alpha_bars or schedule.alpha_bars")


def _get_T(model: torch.nn.Module) -> int:
    if hasattr(model, "T"):
        return int(getattr(model, "T"))
    if hasattr(model, "timesteps"):
        return int(getattr(model, "timesteps"))
    if hasattr(model, "schedule") and hasattr(model.schedule, "T"):
        return int(model.schedule.T)
    return 1000


# -----------------------------------------------------------------------------
# Tensor helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_bchw_2ch(x: torch.Tensor) -> torch.Tensor:
    """Return x as float [B,2,H,W]."""
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
    elif x.ndim == 3:
        if x.shape[0] == 2:
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(1)
            x = torch.cat([x, torch.zeros_like(x)], dim=1)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")
    if x.shape[1] != 2:
        raise ValueError(f"Expected channel dim 2, got shape {tuple(x.shape)}")
    return x.float()


def _mag(x_2ch: torch.Tensor) -> torch.Tensor:
    x_c = twoch_to_complex(_ensure_bchw_2ch(x_2ch))
    mag = torch.abs(x_c)
    if mag.ndim == 4 and mag.shape[1] == 1:
        mag = mag[:, 0]
    return mag


def _scale_from_magnitude(x_2ch: torch.Tensor) -> torch.Tensor:
    mag = _mag(x_2ch)
    return mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8).unsqueeze(1)


def zero_filled_from_kspace_2ch(kspace_2ch: torch.Tensor) -> torch.Tensor:
    kspace_2ch = _ensure_bchw_2ch(kspace_2ch)
    return complex_to_twoch(ifft2c(twoch_to_complex(kspace_2ch)))


def _prepare_mask(mask: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
    mask = mask.to(device).float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1)
        return mask
    if mask.ndim == 3:
        if mask.shape[0] != batch_size:
            if mask.shape[0] == 1:
                mask = mask.expand(batch_size, -1, -1)
            else:
                raise ValueError(f"mask batch {mask.shape[0]} != batch_size {batch_size}")
        return mask.unsqueeze(1)
    raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")


def _build_cond(
    cond_mode: str,
    *,
    x_in_norm: torch.Tensor,
    zf_norm: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    device: torch.device,
    allow_missing_mask: bool = False,
) -> Optional[torch.Tensor]:
    cond_mode = str(cond_mode).lower()
    if cond_mode == "none":
        return None

    # Use the true ZF if available. If not, use x_in only as a last-resort fallback.
    zf_ref = zf_norm if zf_norm is not None else x_in_norm

    if cond_mode == "zf":
        return zf_ref.to(device)

    if cond_mode == "zf_mask":
        if mask is None:
            if not allow_missing_mask:
                raise KeyError(
                    "cond_mode='zf_mask' requires 'mask' in the input .pt bundle or dataset sample. "
                    "Either provide a bundle saved by run_sampling, run from --acc_root, or pass --allow_missing_mask."
                )
            print("[WARN] Missing mask for zf_mask; using zero mask. This is not scientifically equivalent to training.")
            B, _, H, W = zf_ref.shape
            mask_1ch = torch.zeros((B, 1, H, W), device=device, dtype=zf_ref.dtype)
        else:
            mask_1ch = _prepare_mask(mask, batch_size=zf_ref.shape[0], device=device)
        return torch.cat([zf_ref.to(device), mask_1ch], dim=1)

    raise ValueError("cond_mode must be one of: none | zf | zf_mask")


def _load_input_from_pt(path: str, key: Optional[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    meta: Dict[str, Any] = {"input_source": "pt", "input_pt": path}

    if not isinstance(obj, dict):
        return _ensure_bchw_2ch(obj), None, None, None, meta

    preferred_keys = [key] if key else ["x_in_2ch", "zf_2ch", "x_rec_2ch", "x_out_2ch", "img_target_2ch"]
    selected_key = None
    for candidate in preferred_keys:
        if candidate and candidate in obj:
            selected_key = candidate
            break
    if selected_key is None:
        raise KeyError(f"Could not find input key. Requested={key!r}. Available keys: {list(obj.keys())}")

    x_in = _ensure_bchw_2ch(obj[selected_key])
    x_target = obj.get("x_target_2ch", obj.get("img_target_2ch", None))
    if x_target is not None:
        x_target = _ensure_bchw_2ch(x_target)

    zf_ref = obj.get("zf_2ch", None)
    if zf_ref is None and selected_key in {"x_in_2ch", "zf_2ch"}:
        zf_ref = x_in
    if zf_ref is not None:
        zf_ref = _ensure_bchw_2ch(zf_ref)

    mask = obj.get("mask", None)
    if torch.is_tensor(mask):
        mask = mask.float()
    else:
        mask = None

    meta.update({"input_key": selected_key, "available_keys": list(obj.keys())})
    if isinstance(obj.get("meta"), dict):
        meta["input_meta"] = obj["meta"]
    return x_in, x_target, zf_ref, mask, meta


def _load_input_from_dataset(args: argparse.Namespace, target_mode: str, cond_mode: str) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    if args.acc_root is None:
        raise ValueError("When --input_pt is not provided, you must provide --acc_root")

    # The newer dataset accepts acc_root; older variants may ignore cond_mode.
    try:
        ds = DimoKspaceDataset(
            acc_root=args.acc_root,
            acc_factor=args.acc_factor,
            target_mode=target_mode,
            cond_mode=cond_mode,
        )
    except TypeError:
        ds = DimoKspaceDataset(acc_root=args.acc_root, acc_factor=args.acc_factor, target_mode=target_mode)

    sample = ds[int(args.index)]
    x_target = sample.get("img_target_2ch", sample.get("x_target_2ch", None))
    if x_target is not None:
        x_target = _ensure_bchw_2ch(x_target)

    if "zf_2ch" in sample:
        zf = _ensure_bchw_2ch(sample["zf_2ch"])
    elif "kspace_und_2ch" in sample:
        zf = zero_filled_from_kspace_2ch(sample["kspace_und_2ch"])
    else:
        raise KeyError(f"Dataset sample has no zf_2ch or kspace_und_2ch. Keys: {list(sample.keys())}")

    if args.input_mode == "target":
        if x_target is None:
            raise KeyError("--input_mode target requested, but dataset sample has no target image")
        x_in = x_target
    else:
        x_in = zf

    mask = sample.get("mask", None)
    if torch.is_tensor(mask):
        mask = mask.float()
    else:
        mask = None

    meta = {
        "input_source": "dataset",
        "acc_root": args.acc_root,
        "acc_factor": args.acc_factor,
        "index": int(args.index),
        "input_mode": args.input_mode,
    }
    return x_in, x_target, zf, mask, meta


# -----------------------------------------------------------------------------
# DDIM image-to-image denoising
# -----------------------------------------------------------------------------


@torch.no_grad()
def ddim_denoise_from_image(
    model: torch.nn.Module,
    x_in_2ch: torch.Tensor,
    *,
    cond: Optional[torch.Tensor],
    strength: Optional[float],
    t_start: Optional[int],
    num_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval().to(device)
    x_in_2ch = _ensure_bchw_2ch(x_in_2ch).to(device)
    if cond is not None:
        cond = cond.to(device)

    B = x_in_2ch.shape[0]
    T = _get_T(model)
    alpha_bars = _get_alpha_bars(model, device)

    if strength is not None:
        if not (0.0 <= float(strength) <= 1.0):
            raise ValueError("--strength must be in [0, 1]")
        t_start = int(round(float(strength) * (T - 1)))
    if t_start is None:
        t_start = min(30, T - 1)
    t_start = int(max(0, min(t_start, T - 1)))

    t_seq = torch.linspace(t_start, 0, steps=max(2, int(num_steps)), device=device).round().long()
    t_seq = torch.unique_consecutive(t_seq)
    if t_seq[-1].item() != 0:
        t_seq = torch.cat([t_seq, torch.zeros(1, dtype=torch.long, device=device)])

    eps0 = torch.randn_like(x_in_2ch)
    a_start = alpha_bars[t_start].view(1, 1, 1, 1)
    x_t = torch.sqrt(a_start) * x_in_2ch + torch.sqrt(1.0 - a_start) * eps0
    x_t_start = x_t.clone()

    for i in range(len(t_seq) - 1):
        t = int(t_seq[i].item())
        t_prev = int(t_seq[i + 1].item())
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        try:
            eps_pred = model(x_t, t_batch, cond=cond)
        except TypeError:
            if cond is not None:
                raise TypeError("This model forward() does not accept cond=..., but cond_mode is not none.")
            eps_pred = model(x_t, t_batch)

        a_t = alpha_bars[t].view(1, 1, 1, 1)
        a_prev = alpha_bars[t_prev].view(1, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1.0 - a_t) * eps_pred) / torch.sqrt(a_t).clamp_min(1e-12)
        x_t = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1.0 - a_prev) * eps_pred

    return x_t, x_t_start, t_seq.detach().cpu()


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def save_preview(out_png: Path, x_in: torch.Tensor, x_out: torch.Tensor, x_target: Optional[torch.Tensor] = None) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    panels = [("input", _mag(x_in)[0].detach().cpu().numpy()), ("denoised", _mag(x_out)[0].detach().cpu().numpy())]
    if x_target is not None:
        panels.append(("target", _mag(x_target)[0].detach().cpu().numpy()))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Stage-1 denoising (no DC)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    def clean(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, torch.Tensor):
            return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        try:
            json.dumps(x)
            return x
        except TypeError:
            return str(x)
    path.write_text(json.dumps(clean(obj), indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-1 DiMo DDPM denoising without DC; conditional-aware.")

    # Model/checkpoint.
    p.add_argument("--ckpt", required=True)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--data_ch", type=int, default=2)
    p.add_argument("--cond_mode", default="auto", choices=["auto", "none", "zf", "zf_mask"])
    p.add_argument("--cond_ch", type=int, default=None)
    p.add_argument("--strict_load", action="store_true")

    # Input from .pt or dataset.
    p.add_argument("--input_pt", default=None)
    p.add_argument("--input_key", default=None)
    p.add_argument("--acc_root", default=None)
    p.add_argument("--acc_factor", default="04")
    p.add_argument("--input_mode", default="zf", choices=["zf", "target"])
    p.add_argument("--index", type=int, default=0)

    # DDIM img2img.
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--t_start", type=int, default=None)
    p.add_argument("--strength", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_missing_mask", action="store_true")

    # Runtime/output.
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out_dir", default="outputs/stage1_denoise")
    p.add_argument("--out_name", default="denoise_output", help="Base filename without extension")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(int(args.seed))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    state, cfg, legacy_args = _load_ckpt_state_and_meta(args.ckpt)
    timesteps = _infer_timesteps(state, cfg, legacy_args, fallback=int(args.timesteps))
    cond_mode = _infer_cond_mode(args.cond_mode, args.cond_ch, cfg, legacy_args)
    cond_ch = _infer_cond_ch(cond_mode, args.cond_ch, cfg, legacy_args)
    target_mode = cfg.get("target_mode") or legacy_args.get("target_mode") or "complex"

    model = _make_model(data_ch=int(args.data_ch), cond_ch=int(cond_ch), timesteps=int(timesteps)).to(device)
    load_res = model.load_state_dict(state, strict=bool(args.strict_load))
    print(f"[INFO] Loaded checkpoint: {args.ckpt}")
    print(f"[INFO] timesteps={timesteps} cond_mode={cond_mode} cond_ch={cond_ch} target_mode={target_mode}")
    missing = getattr(load_res, "missing_keys", [])
    unexpected = getattr(load_res, "unexpected_keys", [])
    if missing:
        print(f"[WARN] missing_keys ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if unexpected:
        print(f"[WARN] unexpected_keys ({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")

    if args.input_pt is not None:
        x_in, x_target, zf_ref, mask, meta = _load_input_from_pt(args.input_pt, args.input_key)
    else:
        x_in, x_target, zf_ref, mask, meta = _load_input_from_dataset(args, target_mode=target_mode, cond_mode=cond_mode)

    x_in = _ensure_bchw_2ch(x_in)
    if x_target is not None:
        x_target = _ensure_bchw_2ch(x_target)
    if zf_ref is not None:
        zf_ref = _ensure_bchw_2ch(zf_ref)

    # Normalize in the same units used by training. With only one image available,
    # the safest operational scale is the input magnitude max.
    scale_2ch = _scale_from_magnitude(x_in)
    x_in_norm = x_in / scale_2ch
    zf_norm = (zf_ref / scale_2ch) if zf_ref is not None else None
    cond = _build_cond(
        cond_mode,
        x_in_norm=x_in_norm.to(device),
        zf_norm=(zf_norm.to(device) if zf_norm is not None else None),
        mask=mask,
        device=device,
        allow_missing_mask=bool(args.allow_missing_mask),
    )

    x_out_norm, x_t_start_norm, t_seq = ddim_denoise_from_image(
        model,
        x_in_norm,
        cond=cond,
        strength=args.strength,
        t_start=args.t_start,
        num_steps=int(args.num_steps),
        device=device,
    )

    x_out = x_out_norm.detach().cpu() * scale_2ch
    x_t_start = x_t_start_norm.detach().cpu() * scale_2ch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / f"{args.out_name}.pt"
    out_png = out_dir / f"{args.out_name}.png"
    out_json = out_dir / f"{args.out_name}_meta.json"

    params = {
        "ckpt": args.ckpt,
        "timesteps": int(timesteps),
        "cond_mode": cond_mode,
        "cond_ch": int(cond_ch),
        "target_mode": target_mode,
        "input_key": args.input_key,
        "strength": args.strength,
        "t_start": args.t_start,
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "t_seq": [int(x) for x in t_seq.tolist()],
    }
    meta.update(params)

    torch.save(
        {
            "x_in_2ch": x_in.cpu(),
            "x_t_start_2ch": x_t_start.cpu(),
            "x_out_2ch": x_out.cpu(),
            "x_target_2ch": (x_target.cpu() if x_target is not None else None),
            "zf_2ch": (zf_ref.cpu() if zf_ref is not None else None),
            "mask": (mask.cpu() if torch.is_tensor(mask) else None),
            "scale_2ch": scale_2ch.cpu(),
            "meta": meta,
            "params": params,
        },
        out_pt,
    )
    save_preview(out_png, x_in.cpu(), x_out.cpu(), x_target=x_target.cpu() if x_target is not None else None)
    _save_json(out_json, {"meta": meta, "params": params})

    print(f"Saved: {out_pt}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
