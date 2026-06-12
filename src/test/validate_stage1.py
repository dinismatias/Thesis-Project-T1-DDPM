"""Validate Stage-1 diffusion prior on *its intended task* (Gaussian denoising).

This addresses Priority-3 from your list: stop evaluating Stage-1 by denoising
ZF aliasing (aliasing != Gaussian noise).

What this does
--------------
1) Load a clean target x0 (img_target_2ch)
2) Sample a timestep t and create x_t = q(x_t | x0)
3) Run deterministic DDIM reverse steps back to t=0
4) Report PSNR/NMSE on magnitude
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    from src.recon.dimo_model import DiMoDDPM
    from src.recon.dimo_dataset import DimoKspaceDataset
    from src.utils.complex_ops import twoch_to_complex
    from src.baselines.cg_sense import compute_basic_metrics_mag
except Exception:  # pragma: no cover
    from dimo_model import DiMoDDPM  # type: ignore
    from dimo_dataset_conditional import DimoKspaceDataset  # type: ignore
    from complex_ops import twoch_to_complex  # type: ignore
    from cg_sense import compute_basic_metrics_mag  # type: ignore


def _load_state_and_config(ckpt_path: str) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("config", {})
    if isinstance(ckpt, dict):
        return ckpt, {}
    raise ValueError("Unrecognized checkpoint format")


@torch.no_grad()
def ddim_denoise(
    model: DiMoDDPM,
    x0: torch.Tensor,
    *,
    t_start: int,
    num_steps: int,
    cond: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = x0.device
    alpha_bars = model.schedule.alpha_bars.to(device)
    B = x0.shape[0]

    eps0 = torch.randn_like(x0)
    a0 = alpha_bars[t_start].view(1, 1, 1, 1)
    x_t = torch.sqrt(a0) * x0 + torch.sqrt(1.0 - a0) * eps0

    t_seq = torch.linspace(t_start, 0, steps=num_steps, device=device).round().long()
    t_seq = torch.unique_consecutive(t_seq)
    if t_seq[-1].item() != 0:
        t_seq = torch.cat([t_seq, torch.zeros(1, device=device, dtype=torch.long)])

    for i in range(len(t_seq) - 1):
        t = int(t_seq[i].item())
        t_prev = int(t_seq[i + 1].item())
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        eps_pred = model(x_t, t_batch, cond=cond)
        a_t = alpha_bars[t].view(1, 1, 1, 1)
        a_prev = alpha_bars[t_prev].view(1, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1.0 - a_t) * eps_pred) / torch.sqrt(a_t)
        x_t = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1.0 - a_prev) * eps_pred

    return x_t


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--acc_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--cond_mode", default="none", choices=["none", "zf", "zf_mask"])
    p.add_argument("--t_start", type=int, default=50)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out_dir", default="outputs/validate_stage1")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    state, cfg = _load_state_and_config(args.ckpt)
    T = int(cfg.get("timesteps", 100))
    if "schedule.alpha_bars" in state:
        T = int(state["schedule.alpha_bars"].shape[0])

    cond_ch = int(cfg.get("cond_ch", 0))
    model = DiMoDDPM(timesteps=T, data_ch=2, cond_ch=cond_ch).to(device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = DimoKspaceDataset(acc_root=args.acc_root, target_mode=cfg.get("target_mode", "rss"), cond_mode="none")
    sample = ds[args.index]
    x0 = sample["img_target_2ch"].unsqueeze(0).to(device)

    # normalize like training
    scale = x0.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    x0n = x0 / scale

    cond = None
    if args.cond_mode != "none":
        zf = sample["zf_2ch"].unsqueeze(0).to(device) / scale
        if args.cond_mode == "zf":
            cond = zf
        else:
            mask = sample["mask"].unsqueeze(0).to(device).float()
            cond = torch.cat([zf, mask.unsqueeze(1)], dim=1)

    t_start = min(max(args.t_start, 0), T - 1)
    x_hat_n = ddim_denoise(model, x0n, t_start=t_start, num_steps=args.num_steps, cond=cond)
    x_hat = x_hat_n * scale

    metrics = compute_basic_metrics_mag(x_hat, x0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "index": args.index,
        "t_start": t_start,
        "num_steps": args.num_steps,
        "metrics_mag": metrics,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
