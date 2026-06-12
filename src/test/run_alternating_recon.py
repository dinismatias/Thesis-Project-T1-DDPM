# src/test/run_alternating_recon.py
"""
Alternating prior–data-consistency reconstruction for the T1_DDPM / DiMo pipeline.

Algorithm
---------
x = zero-filled reconstruction

for cycle in 1..N:
    1) prior step:
         x_prior = conditional DDIM/img2img denoise(x)
       using the same trained DiMo prior and the same fixed conditioning
       zf or zf+mask used during training.

    2) Stage-2 DDIM + data consistency inside the reverse trajectory:
         x = DDIM_with_DC(x_init=x_prior, measured k-space)
       so DC is applied at every DDIM reverse step, not just once after
       the prior step. This is the important correction versus the earlier
       alternating script.

    3) save:
         cycle_NN.pt, cycle_NN.png, metrics.json, final alternating_recon_output.pt

Important
---------
This script keeps tensors in memory. It does NOT repeatedly shell out to
run_denoise.py and run_sampling.py.

It intentionally reuses internal helpers from:
    src.test.run_sampling
and DC utility from:
    src.recon.dimo_sample

Expected placement
------------------
Save as:
    src/test/run_alternating_recon.py

Example
-------
python -m src.test.run_alternating_recon ^
  --acc_root "C:/Users/Admin/Tese/T1_DDPM_Project/ChallengeData/SingleCoil/Mapping/TrainingSet" ^
  --acc_factor 04 ^
  --index 0 ^
  --ckpt "checkpoints/dimo_cond_r4/epoch_0020.pt" ^
  --cond_mode zf_mask ^
  --cycles 3 ^
  --prior_strength 0.10 ^
  --prior_steps 25 ^
  --dc_mode replace ^
  --scale_mode auto ^
  --save_png ^
  --log_residuals ^
  --out_dir "outputs/AF4_case0_alternating_c3_s010"

For GPU:
  add --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


# -----------------------------------------------------------------------------
# Reuse current project internals
# -----------------------------------------------------------------------------

try:
    # Reuse the canonical run_sampling helpers, so checkpoint/dataset/model handling
    # stays consistent with the scripts you already run successfully.
    from src.test.run_sampling import (
        _load_state_and_config,
        _infer_timesteps,
        _infer_cond_mode,
        _infer_cond_ch,
        _infer_acc_factor,
        _make_dataset,
        _build_model,
        _ensure_bchw_2ch,
        _get_first,
        _mask_to_bhw,
        _scale_from_2ch,
        _build_cond,
        _ddim_img2img_no_dc,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import required helpers from src.test.run_sampling. "
        "Make sure your canonical run_sampling.py is installed as src/test/run_sampling.py."
    ) from exc

try:
    from src.recon.encoding import SenseOp
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import SenseOp from src.recon.encoding") from exc

try:
    from src.recon.dimo_sample import dc_prox_cg, ddim_with_dc_from_model
except Exception:  # pragma: no cover
    dc_prox_cg = None  # type: ignore
    ddim_with_dc_from_model = None  # type: ignore

try:
    from src.utils.complex_ops import twoch_to_complex as ch2_to_complex
except Exception:  # pragma: no cover
    try:
        from src.utils.complex_ops import ch2_to_complex  # type: ignore
    except Exception as exc:
        raise ImportError("Could not import two-channel -> complex converter") from exc

try:
    from src.utils.complex_ops import complex_to_twoch as complex_to_2ch
except Exception:  # pragma: no cover
    try:
        from src.utils.complex_ops import complex_to_2ch  # type: ignore
    except Exception as exc:
        raise ImportError("Could not import complex -> two-channel converter") from exc


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return f"Tensor{tuple(x.shape)}"
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def _adjoint(sense_op: Any, kspace_complex: torch.Tensor, mask_bhw: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compatibility wrapper for SenseOp.adjoint."""
    try:
        return sense_op.adjoint(kspace_complex)
    except TypeError:
        if mask_bhw is None:
            raise
        return sense_op.adjoint(kspace_complex, mask_bhw)


def _forward(sense_op: Any, img_complex: torch.Tensor) -> torch.Tensor:
    """Compatibility wrapper for SenseOp.forward."""
    return sense_op.forward(img_complex)


def _make_sense_op(mask_bhw: torch.Tensor) -> Any:
    """Construct SenseOp robustly across your historical versions."""
    try:
        return SenseOp(mask=mask_bhw)
    except TypeError:
        return SenseOp(mask_bhw)


def _mask_like_kspace(mask_bhw: torch.Tensor, kspace_complex: torch.Tensor) -> torch.Tensor:
    """Return mask broadcastable to k-space complex tensor."""
    mask = mask_bhw.to(device=kspace_complex.device, dtype=kspace_complex.real.dtype)
    while mask.ndim < kspace_complex.ndim:
        # common cases:
        #   kspace [B,H,W], mask [B,H,W] -> no change
        #   kspace [B,C,H,W], mask [B,H,W] -> [B,1,H,W]
        mask = mask.unsqueeze(1)
    return mask


def _dc_residual(
    *,
    x_2ch: torch.Tensor,
    y_2ch: torch.Tensor,
    sense_op: Any,
    mask_bhw: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Relative measured-k-space residual: ||M(Ax-y)|| / ||M y||."""
    x_c = ch2_to_complex(_ensure_bchw_2ch(x_2ch, name="x_2ch"))
    y_c = ch2_to_complex(_ensure_bchw_2ch(y_2ch, name="y_2ch"))
    k = _forward(sense_op, x_c)
    mask = _mask_like_kspace(mask_bhw, k)
    r = (k - y_c) * mask
    y_m = y_c * mask
    num = torch.linalg.vector_norm(r.reshape(r.shape[0], -1), dim=1)
    den = torch.linalg.vector_norm(y_m.reshape(y_m.shape[0], -1), dim=1).clamp_min(eps)
    return float((num / den).mean().detach().cpu())


@torch.no_grad()
def _dc_replace(
    *,
    x_pred_2ch: torch.Tensor,
    y_2ch: torch.Tensor,
    sense_op: Any,
    mask_bhw: torch.Tensor,
) -> torch.Tensor:
    """Hard data consistency: replace measured k-space samples exactly."""
    x_pred_2ch = _ensure_bchw_2ch(x_pred_2ch, name="x_pred_2ch")
    y_2ch = _ensure_bchw_2ch(y_2ch, name="y_2ch")

    x_c = ch2_to_complex(x_pred_2ch)
    y_c = ch2_to_complex(y_2ch)
    k_pred = _forward(sense_op, x_c)
    mask = _mask_like_kspace(mask_bhw, k_pred)

    k_dc = k_pred * (1.0 - mask) + y_c * mask
    x_dc_c = _adjoint(sense_op, k_dc, mask_bhw)
    x_dc_2ch = complex_to_2ch(x_dc_c)
    return _ensure_bchw_2ch(x_dc_2ch, name="x_dc_2ch")


@torch.no_grad()
def _dc_grad(
    *,
    x_pred_2ch: torch.Tensor,
    y_2ch: torch.Tensor,
    sense_op: Any,
    mask_bhw: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Soft gradient data consistency: x <- x - lam A^H M(Ax-y)."""
    x_pred_2ch = _ensure_bchw_2ch(x_pred_2ch, name="x_pred_2ch")
    y_2ch = _ensure_bchw_2ch(y_2ch, name="y_2ch")

    x_c = ch2_to_complex(x_pred_2ch)
    y_c = ch2_to_complex(y_2ch)
    k_pred = _forward(sense_op, x_c)
    mask = _mask_like_kspace(mask_bhw, k_pred)
    resid_k = (k_pred - y_c) * mask
    grad_c = _adjoint(sense_op, resid_k, mask_bhw)
    grad_2ch = _ensure_bchw_2ch(complex_to_2ch(grad_c), name="grad_2ch")
    return x_pred_2ch - float(lam) * grad_2ch


@torch.no_grad()
def _apply_dc(
    *,
    x_prior_2ch: torch.Tensor,
    y_2ch: torch.Tensor,
    sense_op: Any,
    mask_bhw: torch.Tensor,
    dc_mode: str,
    dc_lam: float,
    dc_cg_iter: int,
    dc_cg_tol: float,
) -> torch.Tensor:
    mode = str(dc_mode).lower()
    if mode in {"none", "off"}:
        return x_prior_2ch
    if mode in {"replace", "hard"}:
        return _dc_replace(
            x_pred_2ch=x_prior_2ch,
            y_2ch=y_2ch,
            sense_op=sense_op,
            mask_bhw=mask_bhw,
        )
    if mode in {"grad", "soft"}:
        return _dc_grad(
            x_pred_2ch=x_prior_2ch,
            y_2ch=y_2ch,
            sense_op=sense_op,
            mask_bhw=mask_bhw,
            lam=float(dc_lam),
        )
    if mode == "cg":
        if dc_prox_cg is None:
            raise ImportError("dc_mode='cg' requires src.recon.dimo_sample.dc_prox_cg")
        return dc_prox_cg(
            x_pred_2ch=x_prior_2ch,
            y_meas_2ch=y_2ch,
            sense_op=sense_op,
            lam=float(dc_lam),
            max_iter=int(dc_cg_iter),
            tol=float(dc_cg_tol),
        )
    raise ValueError("dc_mode must be one of: replace | hard | grad | soft | cg | none")


def _mag_np(x_2ch: torch.Tensor):
    x = _ensure_bchw_2ch(x_2ch.detach().cpu(), name="panel_tensor")
    mag = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    return mag[0].numpy()


def _safe_metrics(x_2ch: torch.Tensor, target_2ch: Optional[torch.Tensor]) -> Dict[str, Optional[float]]:
    if target_2ch is None:
        return {"nmse_mag": None, "nrmse_mag": None, "psnr_mag": None}

    x = torch.as_tensor(_ensure_bchw_2ch(x_2ch, name="x_metrics")).detach().float().cpu()
    y = torch.as_tensor(_ensure_bchw_2ch(target_2ch, name="target_metrics")).detach().float().cpu()
    xmag = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    ymag = torch.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)

    err2 = torch.sum((xmag - ymag) ** 2)
    ref2 = torch.sum(ymag ** 2).clamp_min(1e-12)
    mse = torch.mean((xmag - ymag) ** 2).clamp_min(1e-12)
    peak = torch.amax(ymag).clamp_min(1e-12)

    nmse = err2 / ref2
    nrmse = torch.sqrt(nmse)
    psnr = 20.0 * torch.log10(peak / torch.sqrt(mse))

    return {
        "nmse_mag": float(nmse),
        "nrmse_mag": float(nrmse),
        "psnr_mag": float(psnr),
    }


def _save_png_panel(
    *,
    out_png: Path,
    zf_2ch: torch.Tensor,
    current_2ch: torch.Tensor,
    prior_2ch: Optional[torch.Tensor],
    target_2ch: Optional[torch.Tensor],
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not save PNG because matplotlib import failed: {exc}")
        return

    panels: List[Tuple[str, Any]] = [("zf/input", _mag_np(zf_2ch))]
    if prior_2ch is not None:
        panels.append(("prior", _mag_np(prior_2ch)))
    panels.append(("after DC", _mag_np(current_2ch)))
    if target_2ch is not None:
        panels.append(("target", _mag_np(target_2ch)))

    # Shared display scaling based on target if available, otherwise input/current.
    ref = panels[-1][1] if target_2ch is not None else panels[0][1]
    vmax = float(ref.max()) if float(ref.max()) > 0 else None

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _build_args_for_run_sampling(args: argparse.Namespace) -> argparse.Namespace:
    """Create a Namespace containing the attributes expected by run_sampling._make_dataset/_build_model."""
    # _make_dataset in your current run_sampling.py expects several optional attrs.
    ns = argparse.Namespace()
    for k, v in vars(args).items():
        setattr(ns, k, v)

    # Legacy compatibility attrs expected by helper functions.
    if not hasattr(ns, "root"):
        ns.root = None
    if not hasattr(ns, "acc"):
        ns.acc = None

    # Dataset simulation attrs expected by _make_dataset.
    for name, default in [
        ("simulate_mask", False),
        ("sim_accel", None),
        ("sim_mask_type", "random_1D"),
        ("sim_seed", 0),
        ("sim_center", True),
        ("sim_vary_per_slice", False),
    ]:
        if not hasattr(ns, name):
            setattr(ns, name, default)

    if not hasattr(ns, "data_ch"):
        ns.data_ch = 2

    return ns


def _parse_float_schedule(schedule: Optional[str], *, default: float, cycles: int, name: str) -> List[float]:
    """Parse comma-separated per-cycle floats. If shorter than cycles, repeat last value."""
    if schedule is None or str(schedule).strip() == "":
        values = [float(default)]
    else:
        raw = [x.strip() for x in str(schedule).replace(";", ",").split(",") if x.strip()]
        if not raw:
            raise ValueError(f"--{name} was provided but no values were parsed")
        values = [float(x) for x in raw]
    if len(values) < int(cycles):
        values.extend([values[-1]] * (int(cycles) - len(values)))
    return values[: int(cycles)]


def _parse_int_schedule(schedule: Optional[str], *, default: int, cycles: int, name: str) -> List[int]:
    """Parse comma-separated per-cycle ints. If shorter than cycles, repeat last value."""
    if schedule is None or str(schedule).strip() == "":
        values = [int(default)]
    else:
        raw = [x.strip() for x in str(schedule).replace(";", ",").split(",") if x.strip()]
        if not raw:
            raise ValueError(f"--{name} was provided but no values were parsed")
        values = [int(x) for x in raw]
    if len(values) < int(cycles):
        values.extend([values[-1]] * (int(cycles) - len(values)))
    return values[: int(cycles)]


def _normalize_stage2_residual_log(residual_log: Any) -> List[Dict[str, float]]:
    """Normalize residual logs returned by historical dimo_sample variants."""
    if residual_log is None:
        return []

    if hasattr(residual_log, "rows"):
        residual_log = getattr(residual_log, "rows")
    elif hasattr(residual_log, "residuals"):
        residual_log = getattr(residual_log, "residuals")
    elif isinstance(residual_log, dict) and "residual_log" in residual_log:
        residual_log = residual_log.get("residual_log")

    rows: List[Dict[str, float]] = []
    if isinstance(residual_log, list):
        for i, row in enumerate(residual_log):
            if isinstance(row, dict):
                clean: Dict[str, float] = {}
                for k, v in row.items():
                    try:
                        clean[str(k)] = float(v)
                    except Exception:
                        # Keep logs numeric because they go to JSON and tables.
                        continue
                if "iter" not in clean:
                    clean["iter"] = float(i)
                rows.append(clean)
    return rows


@torch.no_grad()
def _run_stage2_ddim_with_dc(
    *,
    model: Any,
    x_init_2ch: torch.Tensor,
    cond: Optional[torch.Tensor],
    y_2ch: torch.Tensor,
    mask_bhw: torch.Tensor,
    sense_op: Any,
    strength: float,
    num_steps: int,
    dc_mode: str,
    dc_lam: float,
    dc_cg_iter: int,
    dc_cg_tol: float,
    log_residuals: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """Run Stage-2 DDIM sampling with DC inside every reverse step.

    This is the central correction: the earlier alternating script applied a
    single DC projection after the prior. Here we reuse dimo_sample's
    ``ddim_with_dc_from_model`` so the trajectory is:

        DDIM reverse step -> DC -> DDIM reverse step -> DC -> ...

    initialized from the Stage-1/prior output ``x_init_2ch``.
    """
    if ddim_with_dc_from_model is None:
        raise ImportError("Stage-2 DDIM+DC requires src.recon.dimo_sample.ddim_with_dc_from_model")

    kwargs = dict(
        model=model,
        sense_op=sense_op,
        y_k_2ch=y_2ch,
        mask=mask_bhw,
        x_init_2ch=x_init_2ch,
        cond=cond,
        strength=float(strength),
        num_steps=int(num_steps),
        dc_mode=dc_mode,
        dc_lam=float(dc_lam),
        dc_cg_iter=int(dc_cg_iter),
        dc_cg_tol=float(dc_cg_tol),
        log_residuals=bool(log_residuals),
        return_residuals=True,
        device=device,
    )

    try:
        out = ddim_with_dc_from_model(**kwargs)
    except TypeError as exc:
        # Some historical versions used log_residuals but not return_residuals.
        if "return_residuals" not in str(exc):
            raise
        kwargs.pop("return_residuals", None)
        out = ddim_with_dc_from_model(**kwargs)

    if isinstance(out, tuple):
        x_out, residual_log = out[0], out[1] if len(out) > 1 else None
    else:
        x_out, residual_log = out, None

    x_out = _ensure_bchw_2ch(x_out, name="x_stage2_ddim_dc_2ch")
    return x_out, _normalize_stage2_residual_log(residual_log)


def _residual_summary_from_rows(rows: List[Dict[str, float]], fallback_before: float, fallback_after: float) -> Dict[str, float]:
    """Small summary for metrics.json while keeping the full residual log too."""
    if not rows:
        return {
            "stage2_first_before_dc": float(fallback_before),
            "stage2_last_before_dc": float(fallback_before),
            "stage2_final_after_dc": float(fallback_after),
        }

    first_before = rows[0].get("before", fallback_before)
    last_before = rows[-1].get("before", first_before)
    final_after = rows[-1].get("after", fallback_after)
    return {
        "stage2_first_before_dc": float(first_before),
        "stage2_last_before_dc": float(last_before),
        "stage2_final_after_dc": float(final_after),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Alternating diffusion-prior / data-consistency reconstruction loop")

    # Data / checkpoint.
    p.add_argument("--acc_root", required=True, help="AccFactor folder or broader TrainingSet folder accepted by current DimoKspaceDataset")
    p.add_argument("--acc_factor", default=None, help="Acceleration id: 04, 08, 10. Inferred if omitted.")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--ckpt", required=True)

    # Model / conditioning.
    p.add_argument("--target_mode", default=None, choices=[None, "complex", "rss"])
    p.add_argument("--cond_mode", default="auto", choices=["auto", "none", "zf", "zf_mask"])
    p.add_argument("--cond_ch", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--data_ch", type=int, default=2)

    # Alternating loop.
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--prior_strength", type=float, default=0.10, help="Fallback DDIM img2img denoising strength per prior step")
    p.add_argument("--prior_strength_schedule", default=None, help="Comma-separated per-cycle strengths, e.g. '0.30,0.10,0.05'. If shorter than --cycles, the last value is repeated.")
    p.add_argument("--prior_steps", type=int, default=25, help="Fallback DDIM steps inside each prior step")
    p.add_argument("--prior_steps_schedule", default=None, help="Optional comma-separated per-cycle prior steps, e.g. '50,25,15'. If shorter than --cycles, the last value is repeated.")

    # Stage-2 sampling stage. This is separate from the prior step: after the
    # prior denoise, DDIM sampling is run with DC inside every reverse step.
    p.add_argument("--stage2_strength", type=float, default=0.10, help="Fallback img2img strength for the Stage-2 DDIM+DC trajectory")
    p.add_argument("--stage2_strength_schedule", default=None, help="Comma-separated per-cycle Stage-2 strengths, e.g. '0.20,0.10,0.05'. If shorter than --cycles, the last value is repeated.")
    p.add_argument("--stage2_steps", type=int, default=50, help="Fallback DDIM steps for the Stage-2 DDIM+DC trajectory")
    p.add_argument("--stage2_steps_schedule", default=None, help="Optional comma-separated per-cycle Stage-2 steps, e.g. '50,50,25'. If shorter than --cycles, the last value is repeated.")

    p.add_argument("--dc_mode", default="replace", choices=["replace", "hard", "grad", "soft", "cg", "none"])
    p.add_argument("--dc_lam", type=float, default=0.1)
    p.add_argument("--dc_cg_iter", type=int, default=10)
    p.add_argument("--dc_cg_tol", type=float, default=1e-6)

    # Scaling.
    p.add_argument("--scale_mode", default="auto", choices=["auto", "zf", "target", "none"])
    p.add_argument("--scale_min", type=float, default=1e-8)

    # Runtime/output.
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_png", action="store_true")
    p.add_argument("--log_residuals", action="store_true")
    p.add_argument("--out_dir", default="outputs/alternating_recon")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    rs_args = _build_args_for_run_sampling(args)

    # Load checkpoint/model metadata using the same compatibility logic as run_sampling.py.
    state, cfg, legacy_args, raw_ckpt = _load_state_and_config(args.ckpt)

    timesteps = _infer_timesteps(state, cfg, legacy_args, fallback=args.timesteps)
    cond_mode = _infer_cond_mode(args.cond_mode, cfg, legacy_args)
    cond_ch = _infer_cond_ch(cond_mode, args.cond_ch, cfg, legacy_args, state)
    acc_factor = _infer_acc_factor(args.acc_root, None, cfg, legacy_args, args.acc_factor)
    target_mode = args.target_mode or cfg.get("target_mode") or legacy_args.get("target_mode") or "complex"

    rs_args.timesteps = timesteps
    rs_args.cond_ch = cond_ch
    rs_args.cond_mode = cond_mode
    rs_args.acc_factor = acc_factor
    rs_args.target_mode = target_mode

    model = _build_model(state, rs_args, timesteps, cond_ch, device)
    model.eval().to(device)

    prior_strength_schedule = _parse_float_schedule(
        args.prior_strength_schedule,
        default=float(args.prior_strength),
        cycles=int(args.cycles),
        name="prior_strength_schedule",
    )
    prior_steps_schedule = _parse_int_schedule(
        args.prior_steps_schedule,
        default=int(args.prior_steps),
        cycles=int(args.cycles),
        name="prior_steps_schedule",
    )
    stage2_strength_schedule = _parse_float_schedule(
        args.stage2_strength_schedule,
        default=float(args.stage2_strength),
        cycles=int(args.cycles),
        name="stage2_strength_schedule",
    )
    stage2_steps_schedule = _parse_int_schedule(
        args.stage2_steps_schedule,
        default=int(args.stage2_steps),
        cycles=int(args.cycles),
        name="stage2_steps_schedule",
    )

    print(f"[INFO] Loaded checkpoint: {args.ckpt}")
    print(f"[INFO] timesteps={timesteps} cond_mode={cond_mode} cond_ch={cond_ch} target_mode={target_mode} acc_factor={acc_factor}")
    print(f"[INFO] cycles={args.cycles} dc_mode={args.dc_mode}")
    print(f"[INFO] prior_strength_schedule={prior_strength_schedule}")
    print(f"[INFO] prior_steps_schedule={prior_steps_schedule}")
    print(f"[INFO] stage2_strength_schedule={stage2_strength_schedule}")
    print(f"[INFO] stage2_steps_schedule={stage2_steps_schedule}")

    # Build dataset through current run_sampling compatibility helper.
    ds = _make_dataset(rs_args, cfg, legacy_args, acc_factor, target_mode, cond_mode)
    sample = ds[int(args.index)]

    # Extract sample tensors.
    y_2ch = _ensure_bchw_2ch(
        _get_first(sample, ["kspace_und_2ch", "y_2ch", "kspace_sub_2ch", "k_und_2ch"]),
        name="kspace_und_2ch",
    ).to(device)

    zf_2ch = _ensure_bchw_2ch(
        _get_first(sample, ["zf_2ch", "x_in_2ch", "img_zf_2ch"]),
        name="zf_2ch",
    ).to(device)

    x_target = _get_first(sample, ["img_target_2ch", "x_target_2ch", "target_2ch"], required=False)
    if x_target is not None:
        x_target = _ensure_bchw_2ch(x_target, name="x_target").to(device)

    mask_raw = _get_first(sample, ["mask", "mask_hw", "sampling_mask"])
    mask_bhw = _mask_to_bhw(mask_raw, batch_size=zf_2ch.shape[0], device=device)

    B, _, H, W = zf_2ch.shape
    sense_op = _make_sense_op(mask_bhw)

    # Choose scale in the same spirit as run_sampling --scale_mode auto.
    scale_mode = args.scale_mode
    if scale_mode == "auto":
        scale_mode = "zf"
    if scale_mode == "none":
        scale_2ch = torch.ones((B, 1, 1, 1), device=device, dtype=zf_2ch.dtype)
    elif scale_mode == "target":
        if x_target is None:
            print("[WARN] scale_mode='target' requested but target is unavailable; falling back to zf.")
            scale_mode = "zf"
            scale_2ch = _scale_from_2ch(zf_2ch, scale_min=args.scale_min).to(device)
        else:
            scale_2ch = _scale_from_2ch(x_target, scale_min=args.scale_min).to(device)
    elif scale_mode == "zf":
        scale_2ch = _scale_from_2ch(zf_2ch, scale_min=args.scale_min).to(device)
    else:
        raise ValueError(f"Unsupported scale_mode: {args.scale_mode}")

    y_norm = y_2ch / scale_2ch
    zf_norm = zf_2ch / scale_2ch
    target_norm = x_target / scale_2ch if x_target is not None else None

    # Fixed conditioning: this should remain the original ZF+mask, because that is
    # what the conditional model was trained to use.
    cond = _build_cond(cond_mode, zf_2ch_norm=zf_norm, mask_bhw=mask_bhw, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[Dict[str, Any]] = []

    # Initial state: normalized zero-filled reconstruction.
    x_norm = zf_norm.clone()

    init_residual = _dc_residual(x_2ch=x_norm, y_2ch=y_norm, sense_op=sense_op, mask_bhw=mask_bhw)
    init_metrics = _safe_metrics(x_norm * scale_2ch, x_target)
    metrics.append({
        "cycle": 0,
        "stage": "initial_zf",
        "dc_residual": init_residual,
        **init_metrics,
    })

    if args.log_residuals:
        print("\n[Alternating prior-DC residuals] internal normalized units")
        print("cycle  prior_s prior_n stage2_s stage2_n   first_before     final_after       nmse_mag       psnr_mag")

    if args.save_png:
        _save_png_panel(
            out_png=out_dir / "cycle_00_initial.png",
            zf_2ch=zf_2ch,
            current_2ch=zf_2ch,
            prior_2ch=None,
            target_2ch=x_target,
            title="cycle 00: initial ZF",
        )

    for cycle in range(1, int(args.cycles) + 1):
        cycle_prior_strength = float(prior_strength_schedule[cycle - 1])
        cycle_prior_steps = int(prior_steps_schedule[cycle - 1])
        cycle_stage2_strength = float(stage2_strength_schedule[cycle - 1])
        cycle_stage2_steps = int(stage2_steps_schedule[cycle - 1])

        # 1) Diffusion prior / denoising step in normalized units.
        x_prior_norm = _ddim_img2img_no_dc(
            model,
            x_norm,
            cond=cond,
            strength=cycle_prior_strength,
            num_steps=cycle_prior_steps,
            device=device,
        )

        prior_residual = _dc_residual(
            x_2ch=x_prior_norm,
            y_2ch=y_norm,
            sense_op=sense_op,
            mask_bhw=mask_bhw,
        )

        # 2) Correct Stage-2: DDIM sampling initialized from the prior output,
        # with data consistency applied inside every reverse step.
        x_next_norm, stage2_residual_log = _run_stage2_ddim_with_dc(
            model=model,
            x_init_2ch=x_prior_norm,
            cond=cond,
            y_2ch=y_norm,
            mask_bhw=mask_bhw,
            sense_op=sense_op,
            strength=cycle_stage2_strength,
            num_steps=cycle_stage2_steps,
            dc_mode=args.dc_mode,
            dc_lam=args.dc_lam,
            dc_cg_iter=args.dc_cg_iter,
            dc_cg_tol=args.dc_cg_tol,
            log_residuals=bool(args.log_residuals),
            device=device,
        )

        after_dc = _dc_residual(
            x_2ch=x_next_norm,
            y_2ch=y_norm,
            sense_op=sense_op,
            mask_bhw=mask_bhw,
        )
        residual_summary = _residual_summary_from_rows(
            stage2_residual_log,
            fallback_before=prior_residual,
            fallback_after=after_dc,
        )
        before_dc = residual_summary["stage2_first_before_dc"]
        after_dc = residual_summary["stage2_final_after_dc"]

        # Original units for saving/visualization/metrics.
        x_prior_orig = x_prior_norm * scale_2ch
        x_next_orig = x_next_norm * scale_2ch

        m = _safe_metrics(x_next_orig, x_target)
        row = {
            "cycle": cycle,
            "stage": "prior_then_stage2_ddim_dc",
            "prior_strength": cycle_prior_strength,
            "prior_steps": cycle_prior_steps,
            "stage2_strength": cycle_stage2_strength,
            "stage2_steps": cycle_stage2_steps,
            "prior_residual_before_stage2": prior_residual,
            "before_dc": before_dc,
            "after_dc": after_dc,
            "dc_residual": after_dc,
            **residual_summary,
            "stage2_residual_log": stage2_residual_log,
            **m,
        }
        metrics.append(row)

        if args.log_residuals:
            nmse_str = "nan" if m["nmse_mag"] is None else f"{m['nmse_mag']:.6e}"
            psnr_str = "nan" if m["psnr_mag"] is None else f"{m['psnr_mag']:.3f}"
            print(f"{cycle:5d} {cycle_prior_strength:8.4f} {cycle_prior_steps:7d} {cycle_stage2_strength:8.4f} {cycle_stage2_steps:8d} {before_dc:14.6e} {after_dc:14.6e} {nmse_str:>14} {psnr_str:>10}")

        cycle_payload = {
            "x_rec_2ch": x_next_orig.detach().cpu(),
            "x_prior_2ch": x_prior_orig.detach().cpu(),
            "x_rec_norm_2ch": x_next_norm.detach().cpu(),
            "x_prior_norm_2ch": x_prior_norm.detach().cpu(),
            "x_in_2ch": zf_2ch.detach().cpu(),
            "zf_2ch": zf_2ch.detach().cpu(),
            "x_target_2ch": (x_target.detach().cpu() if x_target is not None else None),
            "y_2ch": y_2ch.detach().cpu(),
            "mask": mask_bhw.detach().cpu(),
            "scale_2ch": scale_2ch.detach().cpu(),
            "metrics": row,
            "stage2_residual_log": stage2_residual_log,
            "meta": {
                "ckpt": args.ckpt,
                "acc_root": args.acc_root,
                "acc_factor": acc_factor,
                "index": int(args.index),
                "cycle": cycle,
                "cycles": int(args.cycles),
                "cond_mode": cond_mode,
                "cond_ch": int(cond_ch),
                "target_mode": target_mode,
                "scale_mode_requested": args.scale_mode,
                "scale_mode_effective": scale_mode,
                "prior_strength": cycle_prior_strength,
                "prior_steps": cycle_prior_steps,
                "prior_strength_schedule": prior_strength_schedule,
                "prior_steps_schedule": prior_steps_schedule,
                "stage2_strength": cycle_stage2_strength,
                "stage2_steps": cycle_stage2_steps,
                "stage2_strength_schedule": stage2_strength_schedule,
                "stage2_steps_schedule": stage2_steps_schedule,
                "dc_mode": args.dc_mode,
                "dc_lam": float(args.dc_lam),
                "dc_cg_iter": int(args.dc_cg_iter),
                "dc_cg_tol": float(args.dc_cg_tol),
            },
        }

        torch.save(cycle_payload, out_dir / f"cycle_{cycle:02d}.pt")
        with open(out_dir / f"cycle_{cycle:02d}_stage2_residuals.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(stage2_residual_log), f, indent=2)

        if args.save_png:
            _save_png_panel(
                out_png=out_dir / f"cycle_{cycle:02d}.png",
                zf_2ch=zf_2ch,
                current_2ch=x_next_orig,
                prior_2ch=x_prior_orig,
                target_2ch=x_target,
                title=f"cycle {cycle:02d}: prior -> Stage-2 DDIM+{args.dc_mode} DC",
            )

        # Update for next alternation.
        x_norm = x_next_norm

    final_payload = {
        "x_rec_2ch": (x_norm * scale_2ch).detach().cpu(),
        "x_rec_norm_2ch": x_norm.detach().cpu(),
        "x_in_2ch": zf_2ch.detach().cpu(),
        "zf_2ch": zf_2ch.detach().cpu(),
        "x_target_2ch": (x_target.detach().cpu() if x_target is not None else None),
        "y_2ch": y_2ch.detach().cpu(),
        "mask": mask_bhw.detach().cpu(),
        "scale_2ch": scale_2ch.detach().cpu(),
        "metrics": metrics,
        "meta": {
            "ckpt": args.ckpt,
            "acc_root": args.acc_root,
            "acc_factor": acc_factor,
            "index": int(args.index),
            "cycles": int(args.cycles),
            "timesteps": int(timesteps),
            "cond_mode": cond_mode,
            "cond_ch": int(cond_ch),
            "target_mode": target_mode,
            "scale_mode_requested": args.scale_mode,
            "scale_mode_effective": scale_mode,
            "prior_strength": float(args.prior_strength),
            "prior_strength_schedule": prior_strength_schedule,
            "prior_steps": int(args.prior_steps),
            "prior_steps_schedule": prior_steps_schedule,
            "stage2_strength": float(args.stage2_strength),
            "stage2_strength_schedule": stage2_strength_schedule,
            "stage2_steps": int(args.stage2_steps),
            "stage2_steps_schedule": stage2_steps_schedule,
            "dc_mode": args.dc_mode,
            "dc_lam": float(args.dc_lam),
            "dc_cg_iter": int(args.dc_cg_iter),
            "dc_cg_tol": float(args.dc_cg_tol),
            "seed": int(args.seed),
        },
    }

    torch.save(final_payload, out_dir / "alternating_recon_output.pt")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2)

    if args.save_png:
        _save_png_panel(
            out_png=out_dir / "alternating_recon_final.png",
            zf_2ch=zf_2ch,
            current_2ch=final_payload["x_rec_2ch"],
            prior_2ch=None,
            target_2ch=x_target,
            title=f"final alternating output: {args.cycles} cycles",
        )

    print(f"\n[INFO] Saved outputs to: {out_dir}")
    print(f"[INFO] Final file: {out_dir / 'alternating_recon_output.pt'}")
    print(f"[INFO] Metrics: {out_dir / 'metrics.json'}")
    print(f"[INFO] scale_mode={scale_mode} scale_mean={float(scale_2ch.mean().detach().cpu()):.6g}")


if __name__ == "__main__":
    main()
