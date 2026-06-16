# src/tools/run_baselines.py
"""Compute ZF and CG-SENSE baselines per case and write a joinable CSV.

The thesis comparison needs, for every (acc_factor, case_index), the classical
references next to the diffusion reconstruction:

    zero-filled (ZF)  <  CG-SENSE  <  DDPM (alternating)

This tool reconstructs ZF and CG-SENSE for the requested cases and records their
magnitude metrics (NMSE / NRMSE / PSNR / SSIM / HFEN) into ``baselines_summary.csv``.
``aggregate_results.py --baselines_csv ...`` then merges these columns onto each
DDPM run row by (acc_factor, case_index).

The dataset, ZF, and sampling operator are built with the *same* run_sampling
helpers used by run_alternating_recon.py, so ZF here matches the ZF the diffusion
runs started from. All five metrics are scale-invariant, so baselines are scored
directly in dataset units.

Usage
-----
    python -m src.tools.run_baselines \
        --acc_root "ChallengeData/SingleCoil/Mapping/TrainingSet" \
        --acc_factors 04 08 10 --indices 0 1 2 \
        --out_csv outputs/baselines/baselines_summary.csv

A checkpoint is optional and only used to inherit target_mode/acc_factor from its
config; CG-SENSE itself needs no trained model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    from src.test.run_sampling import (
        _load_state_and_config,
        _infer_acc_factor,
        _make_dataset,
        _ensure_bchw_2ch,
        _get_first,
        _mask_to_bhw,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import helpers from src.test.run_sampling") from exc

try:
    from src.baselines.cg_sense import cg_sense_recon
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import cg_sense_recon from src.baselines.cg_sense") from exc

try:
    from src.utils.metrics import compute_image_metrics
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import compute_image_metrics from src.utils.metrics") from exc


_METRIC_KEYS = ["nmse_mag", "nrmse_mag", "psnr_mag", "ssim_mag", "hfen_mag"]


def _rs_args(acc_root: str, acc_factor: str, target_mode: str, cond_mode: str) -> argparse.Namespace:
    """Build the minimal Namespace expected by run_sampling._make_dataset."""
    ns = argparse.Namespace()
    ns.acc_root = acc_root
    ns.root = None
    ns.acc = None
    ns.acc_factor = acc_factor
    ns.target_mode = target_mode
    ns.cond_mode = cond_mode
    ns.data_ch = 2
    for name, default in [
        ("simulate_mask", False),
        ("sim_accel", None),
        ("sim_mask_type", "random_1D"),
        ("sim_seed", 0),
        ("sim_center", True),
        ("sim_vary_per_slice", False),
    ]:
        setattr(ns, name, default)
    return ns


def _prefixed(metrics: Dict[str, Optional[float]], prefix: str) -> Dict[str, Optional[float]]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def run_case(
    *,
    acc_root: str,
    acc_factor: str,
    index: int,
    target_mode: str,
    cond_mode: str,
    cfg: Dict[str, Any],
    legacy_args: Dict[str, Any],
    device: torch.device,
    cg_lam: float,
    cg_max_iter: int,
    cg_tol: float,
    save_pt_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    ds = _make_dataset(_rs_args(acc_root, acc_factor, target_mode, cond_mode),
                       cfg, legacy_args, acc_factor, target_mode, cond_mode)
    if index >= len(ds):
        print(f"[SKIP] AF{acc_factor} index {index} out of range (len={len(ds)})")
        return None
    sample = ds[int(index)]

    y_2ch = _ensure_bchw_2ch(_get_first(sample, ["kspace_und_2ch", "y_2ch", "kspace_sub_2ch", "k_und_2ch"]),
                             name="y_2ch").to(device)
    zf_2ch = _ensure_bchw_2ch(_get_first(sample, ["zf_2ch", "x_in_2ch", "img_zf_2ch"]),
                              name="zf_2ch").to(device)
    target = _get_first(sample, ["img_target_2ch", "x_target_2ch", "target_2ch"], required=False)
    if target is not None:
        target = _ensure_bchw_2ch(target, name="x_target").to(device)
    mask_raw = _get_first(sample, ["mask", "mask_hw", "sampling_mask"])
    mask_bhw = _mask_to_bhw(mask_raw, batch_size=zf_2ch.shape[0], device=device)

    if target is None:
        print(f"[WARN] AF{acc_factor} index {index}: no target available; skipping (metrics undefined).")
        return None

    # ZF metrics.
    zf_metrics = compute_image_metrics(zf_2ch, target, suffix="_mag")

    # CG-SENSE reconstruction + metrics.
    x_cg_2ch, cg_info = cg_sense_recon(
        y_2ch=y_2ch, mask=mask_bhw, lam=cg_lam, max_iter=cg_max_iter, tol=cg_tol,
        device=device, x_ref_2ch=target,
    )
    cg_metrics = compute_image_metrics(x_cg_2ch, target, suffix="_mag")

    row: Dict[str, Any] = {
        "acc_factor": str(acc_factor).zfill(2),
        "case_index": int(index),
        "cg_lam": float(cg_lam),
        "cg_max_iter": int(cg_max_iter),
        "cg_iters": int(cg_info.get("iters", -1)),
    }
    row.update(_prefixed(zf_metrics, "zf"))
    row.update(_prefixed(cg_metrics, "cgsense"))

    if save_pt_dir is not None:
        save_pt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "x_rec_2ch": x_cg_2ch.detach().cpu(),
                "zf_2ch": zf_2ch.detach().cpu(),
                "x_target_2ch": target.detach().cpu(),
                "y_2ch": y_2ch.detach().cpu(),
                "mask": mask_bhw.detach().cpu(),
                "metrics": {"baseline": "cg_sense", **_prefixed(cg_metrics, "cgsense")},
                "meta": {"acc_factor": str(acc_factor).zfill(2), "index": int(index),
                         "cg_lam": float(cg_lam), "cg_max_iter": int(cg_max_iter)},
            },
            save_pt_dir / f"cgsense_AF{str(acc_factor).zfill(2)}_case{index}.pt",
        )

    zf_psnr = zf_metrics.get("psnr_mag")
    cg_psnr = cg_metrics.get("psnr_mag")
    print(f"[OK] AF{str(acc_factor).zfill(2)} case{index} | "
          f"ZF PSNR={zf_psnr:.2f} | CG-SENSE PSNR={cg_psnr:.2f} (iters={row['cg_iters']})")
    return row


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = ["acc_factor", "case_index", "cg_lam", "cg_max_iter", "cg_iters"]
    columns += [f"zf_{k}" for k in _METRIC_KEYS]
    columns += [f"cgsense_{k}" for k in _METRIC_KEYS]
    for r in rows:
        for k in r.keys():
            if k not in columns:
                columns.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute ZF and CG-SENSE baselines per case.")
    p.add_argument("--acc_root", required=True, help="AccFactor folder or broader TrainingSet folder.")
    p.add_argument("--acc_factors", nargs="+", default=["04"], help="Acceleration ids, e.g. 04 08 10.")
    p.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2], help="Case indices to evaluate.")
    p.add_argument("--ckpt", default=None, help="Optional; only used to inherit target_mode from its config.")
    p.add_argument("--target_mode", default="complex", choices=["complex", "rss"])
    p.add_argument("--cond_mode", default="none", choices=["none", "zf", "zf_mask"],
                   help="Only affects dataset construction; irrelevant to the baselines themselves.")
    p.add_argument("--cg_lam", type=float, default=0.0)
    p.add_argument("--cg_max_iter", type=int, default=30)
    p.add_argument("--cg_tol", type=float, default=1e-6)
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument("--save_pt_dir", default=None, help="Optional dir to save CG-SENSE .pt payloads for panels.")
    p.add_argument("--out_csv", default="outputs/baselines/baselines_summary.csv")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    cfg: Dict[str, Any] = {}
    legacy_args: Dict[str, Any] = {}
    target_mode = args.target_mode
    if args.ckpt:
        try:
            _, cfg, legacy_args, _ = _load_state_and_config(args.ckpt)
            target_mode = args.target_mode or cfg.get("target_mode") or legacy_args.get("target_mode") or "complex"
        except Exception as e:  # pragma: no cover
            print(f"[WARN] Could not read config from --ckpt ({e}); using defaults.")

    save_pt_dir = Path(args.save_pt_dir) if args.save_pt_dir else None

    rows: List[Dict[str, Any]] = []
    for af in args.acc_factors:
        af = str(af).zfill(2)
        acc_factor = _infer_acc_factor(args.acc_root, None, cfg, legacy_args, af)
        for idx in args.indices:
            try:
                row = run_case(
                    acc_root=args.acc_root, acc_factor=acc_factor, index=int(idx),
                    target_mode=target_mode, cond_mode=args.cond_mode,
                    cfg=cfg, legacy_args=legacy_args, device=device,
                    cg_lam=args.cg_lam, cg_max_iter=args.cg_max_iter, cg_tol=args.cg_tol,
                    save_pt_dir=save_pt_dir,
                )
            except Exception as e:  # pragma: no cover
                print(f"[FAIL] AF{acc_factor} case{idx}: {e}")
                row = {"acc_factor": acc_factor, "case_index": int(idx), "error": str(e)}
            if row is not None:
                rows.append(row)

    write_csv(rows, Path(args.out_csv))
    print(f"\n[DONE] Wrote {len(rows)} baseline rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
