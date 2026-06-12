"""Quick evaluation sweep: ZF vs CG-SENSE vs Stage-2.

This script is meant for *systematic evaluation* (Priority-4).

It runs over N cases (or a list of indices) and writes a CSV/JSON with metrics.

Usage example (PowerShell):
  python -m src.test.eval_sweep \
    --acc_root ".../AccFactor04" \
    --ckpt "checkpoints/dimo/epoch_0020.pt" \
    --indices 0,1,2,3,4 \
    --out_dir outputs/eval_af4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from src.recon.encoding import SenseOp
    from src.utils.complex_ops import twoch_to_complex as ch2_to_complex
    from src.utils.complex_ops import complex_to_twoch as complex_to_2ch
    from src.recon.dimo_model import DiMoDDPM
    from src.recon.dimo_sample import ddim_with_dc_from_model
    from src.recon.dimo_dataset import DimoKspaceDataset
    from src.baselines.cg_sense import cg_sense_recon, compute_basic_metrics_mag
except Exception:  # pragma: no cover
    from encoding import SenseOp  # type: ignore
    from complex_ops import twoch_to_complex as ch2_to_complex  # type: ignore
    from complex_ops import complex_to_twoch as complex_to_2ch  # type: ignore
    from dimo_model import DiMoDDPM  # type: ignore
    from dimo_sample_conditional_cg import ddim_with_dc_from_model  # type: ignore
    from dimo_dataset_conditional import DimoKspaceDataset  # type: ignore
    from cg_sense import cg_sense_recon, compute_basic_metrics_mag  # type: ignore


def parse_indices(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def _load_state_and_config(ckpt_path: str) -> tuple[Dict[str, torch.Tensor], Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("config", {})
    if isinstance(ckpt, dict):
        return ckpt, {}
    raise ValueError("Unrecognized checkpoint format")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate recon sweeps")
    p.add_argument("--acc_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--indices", default="0,1,2,3,4")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # Stage-2
    p.add_argument("--cond_mode", default="zf_mask", choices=["none", "zf", "zf_mask"])
    p.add_argument("--init_mode", default="zf", choices=["zf", "noise"])
    p.add_argument("--stage2_strength", type=float, default=0.1)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--dc_mode", default="replace", choices=["replace", "grad", "cg"])
    p.add_argument("--dc_lam", type=float, default=0.1)
    p.add_argument("--dc_cg_iter", type=int, default=10)

    p.add_argument("--out_dir", default="outputs/eval")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    indices = parse_indices(args.indices)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    state, cfg = _load_state_and_config(args.ckpt)
    T = int(cfg.get("timesteps", 100))
    if "schedule.alpha_bars" in state:
        T = int(state["schedule.alpha_bars"].shape[0])

    cond_mode = args.cond_mode
    cond_ch = int(cfg.get("cond_ch", 0 if cond_mode == "none" else (2 if cond_mode == "zf" else 3)))

    model = DiMoDDPM(timesteps=T, data_ch=2, cond_ch=cond_ch).to(device)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = DimoKspaceDataset(acc_root=args.acc_root, target_mode=cfg.get("target_mode", "rss"), cond_mode="none")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in indices:
        sample = ds[idx]
        y_2ch = sample["kspace_und_2ch"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        ref = sample.get("img_target_2ch", None)
        if ref is not None:
            ref = ref.unsqueeze(0).to(device)

        sense_op = SenseOp(mask=mask)
        zf = complex_to_2ch(sense_op.adjoint(ch2_to_complex(y_2ch)))

        # CG-SENSE baseline
        x_cg, _ = cg_sense_recon(y_2ch=y_2ch, mask=mask, max_iter=30, tol=1e-6, lam=0.0, device=device)

        # Stage-2
        cond = None
        if cond_mode == "zf":
            cond = zf
        elif cond_mode == "zf_mask":
            cond = torch.cat([zf, mask.unsqueeze(1).float()], dim=1)
        x_init = zf if args.init_mode == "zf" else None

        x_rec = ddim_with_dc_from_model(
            model=model,
            sense_op=sense_op,
            y_k_2ch=y_2ch,
            x_init_2ch=x_init,
            cond=cond,
            strength=args.stage2_strength,
            num_steps=args.num_steps,
            dc_mode=args.dc_mode,
            dc_lam=args.dc_lam,
            dc_cg_iter=args.dc_cg_iter,
            log_residuals=False,
            device=device,
        )

        row = {"index": idx}
        if ref is not None:
            # metrics on magnitude images
            row.update({f"zf_{k}": v for k, v in compute_basic_metrics_mag(zf, ref).items()})
            row.update({f"cg_{k}": v for k, v in compute_basic_metrics_mag(x_cg, ref).items()})
            row.update({f"stage2_{k}": v for k, v in compute_basic_metrics_mag(x_rec, ref).items()})
        rows.append(row)

    # Save
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # CSV (flatten keys)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
