# src/test/run_denoise_v1.py

"""Stage-1 denoising script (NO data consistency).

This is the Stage-1 companion to `run_sampling.py`.

It runs a DDIM-like reverse diffusion *without* any k-space projection/gradient step.
That makes it suitable for:
  - sanity-checking the trained diffusion prior
  - producing a cleaner initialization from an existing image (e.g., zero-filled)

Scientific caveat
-----------------
With an unconditional DDPM trained on clean images (Gaussian forward process),
"denoising" a *zero-filled aliasing artifact* image is not theoretically
well-posed in the same way as Gaussian denoising. Treat this as an empirical
initialization heuristic, not a final reconstruction method.

Usage examples
--------------
From dataset (denoise zero-filled):
  python -m src.test.run_denoise \
    --acc_root /path/to/CMRxRecon/AccFactor04 \
    --acc_factor 04 \
    --ckpt outputs/checkpoints/dimo_best.pt \
    --input_mode zf \
    --t_start 200 --num_steps 50

From a saved tensor (dict containing 'x_2ch'):
  python -m src.test.run_denoise \
    --ckpt outputs/checkpoints/dimo_best.pt \
    --input_pt outputs/sampling_debug_tensors.pt --input_key zf_2ch
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Optional plotting. Matplotlib is already used elsewhere in your repo.
import matplotlib.pyplot as plt

# Match your repo import style, but provide a fallback for standalone use.
try:
    from src.recon.dimo_model import DiMoDDPM
    from src.recon.dimo_dataset import DimoKspaceDataset
    from src.recon.encoding import ifft2c
    from src.utils.complex_ops import twoch_to_complex, complex_to_twoch
except Exception:  # pragma: no cover
    from dimo_model import DiMoDDPM
    from dimo_dataset import DimoKspaceDataset
    from encoding import ifft2c
    from complex_ops import twoch_to_complex, complex_to_twoch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Your `train_dimo.py` saves {'model_state': ...}.
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            return ckpt["model_state"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    # Otherwise assume it's already a state_dict.
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")


def _ensure_bchw_2ch(x: torch.Tensor) -> torch.Tensor:
    """Ensure x is float tensor shaped [B,2,H,W]."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if x.ndim == 2:
        # [H,W] magnitude -> [1,2,H,W] with imag=0
        x = x.unsqueeze(0).unsqueeze(0)
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
    elif x.ndim == 3:
        # Either [2,H,W] or [B,H,W]
        if x.shape[0] == 2:
            x = x.unsqueeze(0)
        else:
            # [B,H,W] magnitude
            x = x.unsqueeze(1)
            x = torch.cat([x, torch.zeros_like(x)], dim=1)
    elif x.ndim == 4:
        # [B,C,H,W]
        pass
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

    if x.shape[1] != 2:
        raise ValueError(
            f"Expected 2 channels (real/imag) in dim=1, got shape {tuple(x.shape)}"
        )

    return x.float()


def zero_filled_from_kspace_2ch(kspace_2ch: torch.Tensor) -> torch.Tensor:
    """Compute zero-filled image (2ch) from undersampled k-space (2ch)."""
    kspace_2ch = _ensure_bchw_2ch(kspace_2ch)
    kspace_c = twoch_to_complex(kspace_2ch)
    img_c = ifft2c(kspace_c)
    return complex_to_twoch(img_c)


@torch.no_grad()
def ddim_denoise_from_image(
    model: DiMoDDPM,
    x_in_2ch: torch.Tensor,
    *,
    t_start: Optional[int] = None,
    strength: Optional[float] = None,
    num_steps: int = 50,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DDIM-like "img2img" denoising (no DC).

    We first *encode* the input image into x_{t_start} by applying the forward
    noising process q(x_t | x_0). Then we run deterministic reverse DDIM steps
    back to t=0.

    Parameters
    ----------
    model:
        Trained DiMoDDPM.
    x_in_2ch:
        Input image as [B,2,H,W] real/imag.
    t_start:
        Starting diffusion timestep. Larger = heavier corruption (more freedom).
    strength:
        Alternative to t_start: value in [0,1]. If set, overrides t_start as
        t_start = round(strength*(T-1)).
    num_steps:
        Number of reverse steps. Uses a strided schedule from t_start->0.

    Returns
    -------
    x_out_2ch:
        Denoised output at t=0.
    x_t_start:
        The noised starting point used (useful for debugging).
    """

    model.eval()
    model = model.to(device)

    x_in_2ch = _ensure_bchw_2ch(x_in_2ch).to(device)
    B = x_in_2ch.shape[0]

    T = int(getattr(model, "timesteps", getattr(model, "T", model.schedule.T)))
    alpha_bars = (model.alpha_bars if hasattr(model, "alpha_bars") else model.schedule.alpha_bars).to(device)

    # --- Decide t_start robustly (support strength and clamp to [0, T-1]) ---
    if strength is not None:
        if not (0.0 <= float(strength) <= 1.0):
            raise ValueError("strength must be in [0,1]")
        t_start = int(round(float(strength) * (T - 1)))

    if t_start is None:
        t_start = min(200, T - 1)  # safe default

    t_start = int(max(0, min(t_start, T - 1)))

    # Build a decreasing list of integer timesteps.
    t_seq = torch.linspace(t_start, 0, steps=num_steps, device=device)
    t_seq = t_seq.round().long()
    t_seq = torch.unique_consecutive(t_seq)
    if t_seq[-1].item() != 0:
        t_seq = torch.cat([t_seq, torch.zeros(1, dtype=torch.long, device=device)])

    # Forward noising: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps
    eps0 = torch.randn_like(x_in_2ch)
    a0 = alpha_bars[t_start].view(1, 1, 1, 1)
    x_t = torch.sqrt(a0) * x_in_2ch + torch.sqrt(1.0 - a0) * eps0
    x_t_start = x_t.clone()

    # Reverse DDIM (eta=0) from t_start -> 0
    for i in range(len(t_seq) - 1):
        t = int(t_seq[i].item())
        t_prev = int(t_seq[i + 1].item())

        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = model(x_t, t_batch)

        a_t = alpha_bars[t].view(1, 1, 1, 1)
        a_prev = alpha_bars[t_prev].view(1, 1, 1, 1)

        # Predict x0
        x0_pred = (x_t - torch.sqrt(1.0 - a_t) * eps_pred) / torch.sqrt(a_t)

        # Deterministic DDIM update
        x_t = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1.0 - a_prev) * eps_pred

    return x_t, x_t_start


def _mag(x_2ch: torch.Tensor) -> torch.Tensor:
    x_c = twoch_to_complex(_ensure_bchw_2ch(x_2ch))
    mag = torch.abs(x_c)
    if mag.ndim == 4 and mag.shape[1] == 1:
        mag = mag[:, 0]  # [B,H,W]
    return mag

def save_preview(
    out_png: Path,
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    x_target: Optional[torch.Tensor] = None,
    title: str = "Stage-1 denoising (no DC)",
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    xin = _mag(x_in)[0].detach().cpu().numpy()   # [H,W]
    xout = _mag(x_out)[0].detach().cpu().numpy() # [H,W]
    panels = [("input", xin), ("denoised", xout)]

    if x_target is not None:
        xt = _mag(x_target)[0].detach().cpu().squeeze(0).numpy()
        panels.append(("target", xt))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage-1 denoising (no DC)")

    # Model + checkpoint
    p.add_argument("--ckpt", required=True, help="Path to trained model checkpoint")
    p.add_argument("--timesteps", type=int, default=1000, help="Must match training")
    p.add_argument("--data_ch", type=int, default=2, help="Must match training")
    p.add_argument("--cond_ch", type=int, default=0, help="Must match training")

    # Input options
    p.add_argument("--input_pt", default=None, help="Optional .pt file containing an input tensor")
    p.add_argument("--input_key", default=None, help="Key in dict .pt (if applicable)")

    # Dataset options (used when input_pt is not provided)
    p.add_argument("--acc_root", default=None, help="Path to AccFactorXX directory")
    p.add_argument("--acc_factor", default="04", help="Acceleration factor string: 04/08/10")
    p.add_argument("--coil_type", default="single", choices=["single", "multi"], help="Dataset coil type")
    p.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split")
    p.add_argument(
        "--input_mode",
        default="zf",
        choices=["zf", "target"],
        help="If dataset is used: denoise zero-filled (zf) or the fully-sampled target",
    )
    p.add_argument("--index", type=int, default=0, help="Dataset index to denoise")

    # DDIM img2img parameters
    p.add_argument("--num_steps", type=int, default=50, help="Number of reverse steps")
    p.add_argument("--t_start", type=int, default=30, help="Starting timestep (overridden by --strength)")
    p.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Alternative to t_start: strength in [0,1], sets t_start=round(strength*(T-1))",
    )

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out_dir", default="outputs/stage1_denoise")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # Create model + load weights
    # Load checkpoint state dict first
    state = _load_state_dict(args.ckpt)

    # Infer diffusion length T from checkpoint schedule tensors
    if "schedule.alpha_bars" in state:
        inferred_T = int(state["schedule.alpha_bars"].shape[0])
    elif "schedule.betas" in state:
        inferred_T = int(state["schedule.betas"].shape[0])
    elif "alpha_bars" in state:
        inferred_T = int(state["alpha_bars"].shape[0])
    else:
        inferred_T = int(args.timesteps)

    if args.timesteps != inferred_T:
        print(f"[INFO] Overriding --timesteps {args.timesteps} -> {inferred_T} (from checkpoint)")
        args.timesteps = inferred_T

    # IMPORTANT: your DiMoDDPM uses T=..., not timesteps=...
    model = DiMoDDPM(data_ch=args.data_ch, cond_ch=args.cond_ch, T=args.timesteps)

    # Handle potential DataParallel prefixes.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    res = model.load_state_dict(state, strict=False)
    missing, unexpected = res.missing_keys, res.unexpected_keys
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}")

    # Load input
    x_target = None
    meta: Dict[str, Any] = {}

    if args.input_pt is not None:
        obj = torch.load(args.input_pt, map_location="cpu")

        if isinstance(obj, dict):
            # You MUST extract x_in from the dict
            key = args.input_key
            if key is None:
                # optional fallback: try common keys
                if "x_in_2ch" in obj:
                    key = "x_in_2ch"
                elif "zf_2ch" in obj:
                    key = "zf_2ch"
                else:
                    raise ValueError(
                        "Input .pt is a dict. Provide --input_key (e.g. x_in_2ch). "
                        f"Available keys: {list(obj.keys())}"
                    )

            if key not in obj:
                raise KeyError(f"--input_key '{key}' not found. Keys: {list(obj.keys())}")

            x_in = obj[key]

            # optional target for preview
            if "x_target_2ch" in obj and obj["x_target_2ch"] is not None:
                x_target = obj["x_target_2ch"]
            else:
                x_target = None
        else:
            x_in = obj
            x_target = None

        x_in = _ensure_bchw_2ch(x_in)
        if x_target is not None:
            x_target = _ensure_bchw_2ch(x_target)

        meta["input_source"] = "pt"
        meta["input_pt"] = args.input_pt
        meta["input_key"] = args.input_key

    else:
        if args.acc_root is None:
            raise ValueError("When --input_pt is not set, you must provide --acc_root")

        ds = DimoKspaceDataset(
            root=args.acc_root,
            acc_factor=args.acc_factor,
            coil_type=args.coil_type,
            split=args.split,
        )
        if not (0 <= args.index < len(ds)):
            raise IndexError(f"index {args.index} out of range (len={len(ds)})")

        sample = ds[args.index]
        kspace_und = sample["kspace_und_2ch"].unsqueeze(0)  # [1,2,H,W] (single-coil)
        x_target = sample.get("img_target_2ch", None)
        if x_target is not None:
            x_target = x_target.unsqueeze(0)

        if args.input_mode == "zf":
            x_in = zero_filled_from_kspace_2ch(kspace_und)
        else:
            if x_target is None:
                raise ValueError("Dataset sample did not include img_target_2ch")
            x_in = x_target

        meta.update(
            {
                "input_source": "dataset",
                "acc_root": args.acc_root,
                "acc_factor": args.acc_factor,
                "coil_type": args.coil_type,
                "split": args.split,
                "index": args.index,
                "input_mode": args.input_mode,
            }
        )


    # --------------------------------------------------------------
    # Match training scale (train_step normalizes each target to O(1)).
    # Normalize x_in (and x_target if available) in the same units before DDIM,
    # then rescale the denoised output back to original units.
    # --------------------------------------------------------------
    x_mag = torch.sqrt(x_in[:, 0] ** 2 + x_in[:, 1] ** 2)                       # [B,H,W]
    scale = x_mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)              # [B,1,1]
    scale_2ch = scale.unsqueeze(1)                                              # [B,1,1,1]

    x_in_norm = x_in / scale_2ch
    x_target_norm = (x_target / scale_2ch) if x_target is not None else None

    # Run denoise
    x_out, x_t_start = ddim_denoise_from_image(
        model,
        x_in_norm,
        t_start=args.t_start,
        strength=args.strength,
        num_steps=args.num_steps,
        device=device,
    )

    # Rescale back to original units
    x_out = x_out * scale_2ch
    x_t_start = x_t_start * scale_2ch

    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"denoise_{meta.get('input_source','input')}_idx{meta.get('index','pt')}"
    out_pt = out_dir / f"{stem}.pt"
    out_png = out_dir / f"{stem}.png"

    torch.save(
        {
            "x_in_2ch": x_in.cpu(),
            "x_t_start_2ch": x_t_start.cpu(),
            "x_out_2ch": x_out.cpu(),
            "x_target_2ch": (x_target.cpu() if x_target is not None else None),
            "scale_2ch": scale_2ch.cpu(),
            "meta": meta,
            "params": {
                "timesteps": args.timesteps,
                "num_steps": args.num_steps,
                "t_start": args.t_start,
                "strength": args.strength,
                "seed": args.seed,
            },
        },
        out_pt,
    )

    save_preview(out_png, x_in, x_out, x_target=x_target)

    print(f"Saved: {out_pt}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
