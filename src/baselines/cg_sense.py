"""CG-SENSE baseline reconstruction (single- or multi-coil).

This implements a classical *data-consistent* baseline by solving the
least-squares problem

    x* = argmin_x || A x - y ||_2^2 + lam ||x||_2^2

where A is the MRI encoding operator (masked FFT for single-coil, or
masked SENSE for multi-coil).

Why this file exists
--------------------
Your diffusion Stage-2 already enforces data consistency, but you still need a
strong classical reference beyond zero-filled.

This module:
  - reuses **your** `encoding.py::SenseOp` so FFT scaling/centering matches
    everything else in your pipeline
  - provides a clean Conjugate Gradient solver on the normal equations
  - optionally reports metrics vs a provided reference

Inputs/outputs
--------------
Main function: `cg_sense_recon(...)`
  input: y_2ch, mask, optional coil sensitivities (mps)
  output: x_cg_2ch and an `info` dict (iters, residual norms, metrics)

CLI
---
You can also run it as a script for quick smoke tests, either:
  (A) from your dataset root + acc + index, or
  (B) from a `.pt` bundle like `run_sampling.py --dump_stage1_bundle ...`

Example (from bundle):
  python -m src.baselines.cg_sense \
    --input_pt outputs/stage1_input_bundle.pt \
    --out_dir outputs/cg_sense_smoke \
    --max_iter 30 --tol 1e-6 --lam 0.0

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# -----------------------------------------------------------------------------
# Robust imports (src.* first, then local fallbacks)
# -----------------------------------------------------------------------------

try:
    from src.recon.encoding import SenseOp
except Exception:  # pragma: no cover
    from encoding import SenseOp  # type: ignore

try:
    from src.utils.complex_ops import ch2_to_complex, complex_to_2ch
except Exception:  # pragma: no cover
    from complex_ops import ch2_to_complex, complex_to_2ch  # type: ignore

try:
    from src.recon.dimo_dataset import DimoKspaceDataset as _Dataset
except Exception:  # pragma: no cover
    try:
        from dimo_dataset import DimoKspaceDataset as _Dataset  # type: ignore
    except Exception:
        _Dataset = None  # type: ignore

Tensor = torch.Tensor


def _as_mask_bhw(mask: Tensor, B: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return mask as [B,H,W] float tensor."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"mask must be [H,W] or [B,H,W], got {tuple(mask.shape)}")
    if mask.shape[0] == 1 and B > 1:
        mask = mask.expand(B, -1, -1)
    if mask.shape[0] != B:
        raise ValueError(f"mask batch {mask.shape[0]} != B={B}")
    return mask.to(device=device, dtype=dtype)


def _inner_prod(a: Tensor, b: Tensor) -> Tensor:
    """Complex inner product <a,b> = sum conj(a)*b (scalar)."""
    return torch.sum(torch.conj(a) * b)


@torch.no_grad()
def cg_sense_recon(
    *,
    y_2ch: Tensor,
    mask: Tensor,
    mps: Optional[Tensor] = None,
    lam: float = 0.0,
    max_iter: int = 30,
    tol: float = 1e-6,
    x0_2ch: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    x_ref_2ch: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Conjugate-gradient SENSE reconstruction.

    Args:
        y_2ch: undersampled k-space in 2ch real/imag.
              Single-coil: [B,2,H,W]. Multi-coil: [B,2*C,H,W].
        mask: sampling mask [H,W] or [B,H,W], values in {0,1}.
        mps: optional coil sensitivity maps [B,C,H,W] complex.
        lam: optional L2/Tikhonov regularization weight.
        max_iter: CG iterations.
        tol: relative residual tolerance on ||r||/||b||.
        x0_2ch: optional initial image guess in 2ch image domain [B,2,H,W].
        device: torch device.
        verbose: print residuals.
        x_ref_2ch: optional reference image in 2ch for metrics.

    Returns:
        x_cg_2ch: reconstructed image in 2ch [B,2,H,W].
        info: dict with iteration logs + optional metrics.
    """
    if device is None:
        device = y_2ch.device

    y_2ch = y_2ch.to(device)
    if y_2ch.ndim == 3:
        y_2ch = y_2ch.unsqueeze(0)
    if y_2ch.ndim != 4:
        raise ValueError(f"y_2ch must be [B,2*C,H,W] or [2*C,H,W], got {tuple(y_2ch.shape)}")

    B, ch, H, W = y_2ch.shape
    if ch % 2 != 0:
        raise ValueError(f"y_2ch channel dim must be even (real/imag pairs), got {ch}")

    dtype = y_2ch.dtype
    mask_b = _as_mask_bhw(mask, B=B, device=device, dtype=dtype)

    # Convert y to complex [B,C,H,W]
    y_c = ch2_to_complex(y_2ch)  # [B,C,H,W] complex
    y_c = y_c * mask_b.unsqueeze(1)  # ensure zeros on unmeasured

    sense_op = SenseOp(mask=mask_b, mps=mps)

    # b = A^H y
    b = sense_op.adjoint(y_c)  # [B,1,H,W] complex

    # Initial guess
    if x0_2ch is not None:
        x = ch2_to_complex(x0_2ch.to(device))
    else:
        x = sense_op.adjoint(y_c)

    lam_t = torch.tensor(float(lam), device=device, dtype=torch.float32)

    def A(img_c: Tensor) -> Tensor:
        k = sense_op.forward(img_c)
        out = sense_op.adjoint(k)
        if lam > 0:
            out = out + lam_t.to(out.dtype) * img_c
        return out

    r = b - A(x)
    p = r.clone()
    rr = _inner_prod(r, r).real
    bnorm = torch.sqrt(_inner_prod(b, b).real).clamp_min(1e-12)

    res_hist = []
    rel_hist = []

    for it in range(int(max_iter)):
        Ap = A(p)
        pAp = _inner_prod(p, Ap).real.clamp_min(1e-12)
        alpha = rr / pAp

        x = x + alpha * p
        r = r - alpha * Ap

        rr_new = _inner_prod(r, r).real
        res_norm = torch.sqrt(rr_new).item()
        rel = (torch.sqrt(rr_new) / bnorm).item()
        res_hist.append(float(res_norm))
        rel_hist.append(float(rel))

        if verbose:
            print(f"[CG] it={it:03d}  ||r||={res_norm:.3e}  rel={rel:.3e}")

        if rel < tol:
            rr = rr_new
            break

        beta = rr_new / rr.clamp_min(1e-12)
        p = r + beta * p
        rr = rr_new

    x_2ch = complex_to_2ch(x)  # [B,2,H,W]

    info: Dict[str, Any] = {
        "max_iter": int(max_iter),
        "tol": float(tol),
        "lam": float(lam),
        "iters": int(len(res_hist)),
        "residual_norm": res_hist,
        "residual_rel": rel_hist,
    }

    if x_ref_2ch is not None:
        info["metrics_mag"] = compute_basic_metrics_mag(x_hat_2ch=x_2ch, x_ref_2ch=x_ref_2ch)

    return x_2ch, info


def compute_basic_metrics_mag(*, x_hat_2ch: Tensor, x_ref_2ch: Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Compute basic metrics on magnitude images."""
    xh = ch2_to_complex(x_hat_2ch)
    xr = ch2_to_complex(x_ref_2ch)

    xh_mag = torch.abs(xh)
    xr_mag = torch.abs(xr)
    if xh_mag.ndim == 4 and xh_mag.shape[1] == 1:
        xh_mag = xh_mag[:, 0]
    if xr_mag.ndim == 4 and xr_mag.shape[1] == 1:
        xr_mag = xr_mag[:, 0]

    diff = xh_mag - xr_mag
    mse = torch.mean(diff * diff, dim=(-2, -1))
    rmse = torch.sqrt(mse + eps)
    peak = torch.amax(xr_mag, dim=(-2, -1)).clamp_min(eps)
    psnr = 20.0 * torch.log10(peak / rmse)

    num = torch.sum(diff * diff, dim=(-2, -1))
    den = torch.sum(xr_mag * xr_mag, dim=(-2, -1)).clamp_min(eps)
    nmse = num / den

    return {
        "psnr_mean": float(psnr.mean().item()),
        "psnr_min": float(psnr.min().item()),
        "psnr_max": float(psnr.max().item()),
        "nmse_mean": float(nmse.mean().item()),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CG-SENSE baseline using encoding.SenseOp")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_pt", type=str, default=None, help="Load y/mask/(x_ref) from a .pt dict")
    src.add_argument("--root", type=str, default=None, help="Dataset root (folder containing AccFactorXX)")

    p.add_argument("--acc", type=str, default="AccFactor04", help="Acceleration folder (dataset mode)")
    p.add_argument("--index", type=int, default=0, help="Dataset index (dataset mode)")

    # bundle keys (pt mode)
    p.add_argument("--y_key", type=str, default="y_2ch", help="Key for y in input_pt")
    p.add_argument("--mask_key", type=str, default="mask", help="Key for mask in input_pt")
    p.add_argument("--ref_key", type=str, default="x_target_2ch", help="Key for reference image in input_pt")

    # CG
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--max_iter", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--verbose", action="store_true")

    # output
    p.add_argument("--out_dir", type=str, default="./outputs/cg_sense")
    return p.parse_args()


def _load_from_dataset(root: str, acc: str, index: int) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    if _Dataset is None:
        raise ImportError("Could not import DimoKspaceDataset. Run inside your repo or adjust imports.")

    acc_root = Path(root) / acc
    case_dirs = sorted([p for p in acc_root.glob("P*") if p.is_dir()])
    if not case_dirs:
        raise FileNotFoundError(f"No P* case dirs found under: {acc_root}")

    acc_factor = "".join([c for c in acc if c.isdigit()])[-2:]
    ds = _Dataset(case_dirs=case_dirs, acc_factor=acc_factor, multi_coil=False, use_full_as_target=True)
    sample: Dict[str, Tensor] = ds[index]
    y = sample.get("kspace_und_2ch", sample.get("y_2ch"))
    m = sample.get("mask")
    ref = sample.get("img_target_2ch", sample.get("x0_2ch", None))
    if y.ndim == 3:
        y = y.unsqueeze(0)
    if ref is not None and isinstance(ref, torch.Tensor) and ref.ndim == 3:
        ref = ref.unsqueeze(0)
    return y, m, ref


def _save_preview_png(out_png: Path, x_zf_2ch: Tensor, x_cg_2ch: Tensor, x_ref_2ch: Optional[Tensor]) -> None:
    import matplotlib.pyplot as plt

    def mag(x2: Tensor) -> Tensor:
        xc = ch2_to_complex(x2)
        m = torch.abs(xc)
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0]
        return m

    zf = mag(x_zf_2ch)[0].detach().cpu().numpy()
    cg = mag(x_cg_2ch)[0].detach().cpu().numpy()

    panels = [("ZF", zf), ("CG-SENSE", cg)]
    if x_ref_2ch is not None:
        ref = mag(x_ref_2ch)[0].detach().cpu().numpy()
        panels.extend([("REF", ref), ("|CG-REF|", abs(cg - ref))])

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_pt is not None:
        obj = torch.load(args.input_pt, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError("--input_pt must point to a dict saved by torch.save")
        y = obj[args.y_key]
        mask = obj[args.mask_key]
        ref = obj.get(args.ref_key, None)
        if y.ndim == 3:
            y = y.unsqueeze(0)
        if ref is not None and isinstance(ref, torch.Tensor) and ref.ndim == 3:
            ref = ref.unsqueeze(0)
    else:
        y, mask, ref = _load_from_dataset(args.root, args.acc, args.index)

    # zf for preview
    mask_b = _as_mask_bhw(mask, B=int(y.shape[0]), device=device, dtype=y.dtype)
    y_c = ch2_to_complex(y.to(device))
    y_c = y_c * mask_b.unsqueeze(1)
    zf_c = SenseOp(mask=mask_b).adjoint(y_c)
    zf_2ch = complex_to_2ch(zf_c)

    x_cg_2ch, info = cg_sense_recon(
        y_2ch=y,
        mask=mask,
        lam=args.lam,
        max_iter=args.max_iter,
        tol=args.tol,
        device=device,
        verbose=args.verbose,
        x_ref_2ch=ref,
    )

    torch.save({"x_cg_2ch": x_cg_2ch.cpu(), "info": info}, out_dir / "cg_sense_output.pt")
    with open(out_dir / "cg_sense_metrics.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    _save_preview_png(out_dir / "cg_sense_preview.png", zf_2ch, x_cg_2ch, ref)
    print(f"Saved CG-SENSE outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
