# src/tools/make_panels.py
"""Render reconstruction comparison panels with error maps.

For each run it produces a single figure with:

    ZF  |  (prior)  |  recon  |  target  |  |recon - target|  error map

The grayscale panels share one display window (based on the target) so visual
contrast is comparable; the error map uses its own colormap and scale with a
colorbar. Metric values (NMSE / PSNR / SSIM / HFEN) are written into the title,
taken from the saved payload / metrics_summary.json when available and otherwise
recomputed from the tensors.

Inputs it understands (saved by run_alternating_recon.py):
  - best_cycle.pt / cycle_NN.pt : have x_rec_2ch, x_prior_2ch, zf_2ch, x_target_2ch
  - alternating_recon_output.pt : has x_rec_2ch, zf_2ch, x_target_2ch (no prior)

Usage
-----
Single run (auto-pick best_cycle.pt then final output):
    python -m src.tools.make_panels --run_dir outputs/sweeps_cpu_alt/AF04_case0_aggressive_equal

Specific file:
    python -m src.tools.make_panels --pt outputs/.../best_cycle.pt --out panel.png

Whole sweep tree (one panel per run, written as panel.png inside each run):
    python -m src.tools.make_panels --root outputs/sweeps_gpu_b2_alt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from src.utils.metrics import compute_image_metrics
except Exception:  # pragma: no cover
    from metrics import compute_image_metrics  # type: ignore


def _to_mag(x: Optional[torch.Tensor]):
    """Return a [H,W] magnitude numpy array from a 2-channel/complex tensor."""
    if x is None or not torch.is_tensor(x):
        return None
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] == 2:
        mag = torch.sqrt(x[0] ** 2 + x[1] ** 2)
    elif x.ndim == 3 and x.shape[0] == 1:
        mag = x[0]
    elif x.ndim == 2:
        mag = x
    else:
        raise ValueError(f"Cannot reduce tensor of shape {tuple(x.shape)} to magnitude")
    return mag.numpy()


def _pick_pt(run_dir: Path, prefer: str) -> Optional[Path]:
    candidates: List[Path] = []
    if prefer == "best":
        candidates = [run_dir / "best_cycle.pt", run_dir / "alternating_recon_output.pt"]
    elif prefer == "final":
        candidates = [run_dir / "alternating_recon_output.pt", run_dir / "best_cycle.pt"]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the highest-numbered cycle_NN.pt.
    cycles = sorted(run_dir.glob("cycle_*.pt"))
    cycles = [c for c in cycles if re.search(r"cycle_\d+\.pt$", c.name)]
    if cycles:
        return cycles[-1]
    for c in (run_dir / "best_cycle.pt", run_dir / "alternating_recon_output.pt"):
        if c.exists():
            return c
    return None


def _metrics_for_title(payload: Dict[str, Any], pt_path: Path,
                       rec: Optional[torch.Tensor], target: Optional[torch.Tensor]) -> Dict[str, Optional[float]]:
    """Best-effort metric dict with keys nmse_mag/psnr_mag/ssim_mag/hfen_mag."""
    keys = ["nmse_mag", "nrmse_mag", "psnr_mag", "ssim_mag", "hfen_mag"]

    # 1) payload['metrics'] may already be a per-cycle dict.
    m = payload.get("metrics")
    if isinstance(m, dict) and m.get("nmse_mag") is not None:
        return {k: m.get(k) for k in keys}

    # 2) sibling metrics_summary.json best cycle.
    summ = pt_path.parent / "metrics_summary.json"
    if summ.exists():
        try:
            obj = json.loads(summ.read_text(encoding="utf-8"))
            bm = (obj.get("best_cycle") or {}).get("best_cycle_metrics") or {}
            if bm.get("nmse_mag") is not None:
                return {k: bm.get(k) for k in keys}
        except Exception:
            pass

    # 3) recompute from tensors.
    if target is not None and rec is not None:
        return compute_image_metrics(rec, target, suffix="_mag")
    return {k: None for k in keys}


def _fmt(v: Optional[float], spec: str) -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "NA"


def render_panel(pt_path: Path, out_png: Path, *, cmap: str = "gray",
                 error_cmap: str = "magma", error_quantile: float = 0.99) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] matplotlib unavailable, cannot render {pt_path}: {exc}")
        return False

    payload = torch.load(pt_path, map_location="cpu")
    if not isinstance(payload, dict):
        print(f"[WARN] Unexpected payload type in {pt_path}: {type(payload)}")
        return False

    rec = payload.get("x_rec_2ch")
    prior = payload.get("x_prior_2ch")
    zf = payload.get("zf_2ch", payload.get("x_in_2ch"))
    target = payload.get("x_target_2ch")

    zf_m, prior_m, rec_m, tgt_m = _to_mag(zf), _to_mag(prior), _to_mag(rec), _to_mag(target)

    panels: List[Tuple[str, Any]] = []
    if zf_m is not None:
        panels.append(("ZF / input", zf_m))
    if prior_m is not None:
        panels.append(("prior", prior_m))
    if rec_m is not None:
        panels.append(("reconstruction", rec_m))
    if tgt_m is not None:
        panels.append(("target", tgt_m))

    if not panels:
        print(f"[WARN] No displayable tensors in {pt_path}")
        return False

    # Shared grayscale window from the target (fallback: recon, then ZF).
    ref_img = tgt_m if tgt_m is not None else (rec_m if rec_m is not None else zf_m)
    import numpy as np
    vmax = float(np.quantile(ref_img, 0.995)) if ref_img is not None else None
    if not vmax or vmax <= 0:
        vmax = float(ref_img.max()) if ref_img is not None else None

    err = None
    if rec_m is not None and tgt_m is not None:
        err = np.abs(rec_m - tgt_m)

    n = len(panels) + (1 if err is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.3), squeeze=False)
    axes = axes[0]

    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")

    if err is not None:
        emax = float(np.quantile(err, error_quantile))
        if not emax or emax <= 0:
            emax = float(err.max()) if err.max() > 0 else 1.0
        ax = axes[len(panels)]
        im = ax.imshow(err, cmap=error_cmap, vmin=0.0, vmax=emax)
        ax.set_title("|recon - target|")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    m = _metrics_for_title(payload, pt_path, rec, target)
    title = (
        f"{pt_path.parent.name}  ({pt_path.name})\n"
        f"NMSE={_fmt(m.get('nmse_mag'), '.4e')}  "
        f"PSNR={_fmt(m.get('psnr_mag'), '.2f')} dB  "
        f"SSIM={_fmt(m.get('ssim_mag'), '.4f')}  "
        f"HFEN={_fmt(m.get('hfen_mag'), '.4f')}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"[OK] {out_png}")
    return True


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render recon comparison panels with error maps.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir", default=None, help="A single run folder.")
    src.add_argument("--pt", default=None, help="A specific .pt payload to render.")
    src.add_argument("--root", default=None, help="Walk a tree and render one panel per run.")

    p.add_argument("--out", default=None, help="Output PNG (single mode). Default: <run>/panel.png")
    p.add_argument("--out_dir", default=None,
                   help="In --root mode, write all panels here as <run_name>.png "
                        "instead of panel.png inside each run.")
    p.add_argument("--prefer", default="best", choices=["best", "final"],
                   help="Which payload to render when several exist.")
    p.add_argument("--cmap", default="gray")
    p.add_argument("--error_cmap", default="magma")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.pt is not None:
        pt = Path(args.pt)
        out = Path(args.out) if args.out else pt.with_name("panel.png")
        render_panel(pt, out, cmap=args.cmap, error_cmap=args.error_cmap)
        return

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        pt = _pick_pt(run_dir, args.prefer)
        if pt is None:
            print(f"[WARN] No renderable .pt found in {run_dir}")
            return
        out = Path(args.out) if args.out else run_dir / "panel.png"
        render_panel(pt, out, cmap=args.cmap, error_cmap=args.error_cmap)
        return

    # --root mode.
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"--root does not exist: {root}")
    out_dir = Path(args.out_dir) if args.out_dir else None

    run_dirs = sorted({p.parent for marker in ("best_cycle.pt", "alternating_recon_output.pt")
                       for p in root.rglob(marker)} |
                      {p.parent for p in root.rglob("cycle_*.pt")},
                      key=lambda d: str(d))
    if not run_dirs:
        print(f"[WARN] No runs with renderable .pt found under {root}")
        return

    n_ok = 0
    for d in run_dirs:
        pt = _pick_pt(d, args.prefer)
        if pt is None:
            continue
        out = (out_dir / f"{d.name}.png") if out_dir else (d / "panel.png")
        if render_panel(pt, out, cmap=args.cmap, error_cmap=args.error_cmap):
            n_ok += 1
    print(f"\n[DONE] Rendered {n_ok}/{len(run_dirs)} panels.")


if __name__ == "__main__":
    main()
