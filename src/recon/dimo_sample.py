# src/recon/dimo_sample.py
"""DDIM sampling and MRI data-consistency utilities for DiMo.

This file merges the capabilities of the historical ``dimo_sample`` versions:

- legacy wrapper ``ddim_like_sampling_with_dc``;
- current keyword API ``ddim_with_dc_from_model``;
- conditional model calls via ``cond=...``;
- image-to-image / Stage-1 -> Stage-2 initialization via ``x_init_2ch``;
- DC modes: ``replace``/``hard``, ``grad``/``soft``, ``cg`` and ``none``;
- residual logging before/after DC;
- robust extraction of model schedule from ``model.alpha_bars`` or
  ``model.schedule.alpha_bars``.

The scientific default for your current thesis experiments should still be
``dc_mode='replace'`` while you are validating the learned prior, because it
exactly enforces the measured k-space samples.  ``dc_mode='cg'`` is a proximal
step; with small ``dc_lam`` it is expected to leave a much larger measured-data
residual than hard replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    import src.utils.complex_ops as _complex_ops
except Exception:  # pragma: no cover
    try:
        import complex_ops as _complex_ops  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("Could not import complex_ops from src.utils or local path") from exc

try:
    from src.recon.encoding import SenseOp
except Exception:  # pragma: no cover
    try:
        from encoding import SenseOp  # type: ignore
    except Exception:  # pragma: no cover
        SenseOp = Any  # type: ignore

Tensor = torch.Tensor


def ch2_to_complex(x: Tensor) -> Tensor:
    """Compatibility wrapper for 2-channel -> complex conversion."""
    for name in ("twoch_to_complex", "ch2_to_complex", "to_complex"):
        fn = getattr(_complex_ops, name, None)
        if fn is not None:
            return fn(x)
    raise AttributeError("complex_ops must expose twoch_to_complex or ch2_to_complex")


def complex_to_2ch(x: Tensor) -> Tensor:
    """Compatibility wrapper for complex -> 2-channel conversion."""
    for name in ("complex_to_twoch", "complex_to_2ch", "from_complex"):
        fn = getattr(_complex_ops, name, None)
        if fn is not None:
            return fn(x)
    raise AttributeError("complex_ops must expose complex_to_twoch or complex_to_2ch")


@dataclass
class ResidualLog:
    timesteps: List[int]
    before_dc: List[float]
    after_dc: List[float]

    def to_dict(self) -> Dict[str, List[Union[int, float]]]:
        return {"timesteps": self.timesteps, "before_dc": self.before_dc, "after_dc": self.after_dc}

    def to_rows(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for i, (t, b, a) in enumerate(zip(self.timesteps, self.before_dc, self.after_dc)):
            rows.append({"iter": i, "t": int(t), "before": float(b), "after": float(a)})
        return rows


# -----------------------------------------------------------------------------
# Basic shape/schedule helpers
# -----------------------------------------------------------------------------


def _ensure_bchw_2ch(x: Tensor, *, name: str) -> Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim == 3 and x.shape[0] == 2:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"{name} must be [B,2,H,W] or [2,H,W], got {tuple(x.shape)}")
    return x.float()


def _mask_to_bhw(mask: Optional[Tensor], *, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=device, dtype=dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim != 3:
        raise ValueError(f"mask must be [H,W], [B,H,W], or [B,1,H,W], got {tuple(mask.shape)}")
    if mask.shape[-2:] != (H, W):
        raise ValueError(f"mask spatial shape {tuple(mask.shape[-2:])} does not match {(H, W)}")
    if mask.shape[0] == 1 and B > 1:
        mask = mask.expand(B, -1, -1)
    if mask.shape[0] != B:
        raise ValueError(f"mask batch size {mask.shape[0]} does not match {B}")
    return mask


def _get_alpha_bars_and_T(model: Any, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, int]:
    if hasattr(model, "alpha_bars"):
        alpha_bars = getattr(model, "alpha_bars")
        T = int(getattr(model, "T", getattr(model, "timesteps", len(alpha_bars))))
        return alpha_bars.to(device=device, dtype=dtype), T
    if hasattr(model, "schedule") and hasattr(model.schedule, "alpha_bars"):
        alpha_bars = model.schedule.alpha_bars
        T = int(getattr(model, "T", getattr(model, "timesteps", getattr(model.schedule, "T", len(alpha_bars)))))
        return alpha_bars.to(device=device, dtype=dtype), T
    raise AttributeError("Model must expose model.alpha_bars or model.schedule.alpha_bars")


def _make_ddim_timesteps(t_start: int, num_steps: int, *, device: Optional[torch.device] = None) -> List[int]:
    t_start = int(max(t_start, 0))
    if num_steps <= 1 or t_start == 0:
        return [t_start]
    ts = torch.linspace(t_start, 0, steps=int(num_steps), device=device)
    ts = torch.round(ts).long().detach().cpu().tolist()
    out: List[int] = []
    for t in ts:
        if not out or int(t) != out[-1]:
            out.append(int(t))
    if out[-1] != 0:
        out.append(0)
    return out


def _q_sample_from_x0(x0_2ch: Tensor, t: int, alpha_bars: Tensor, noise_2ch: Optional[Tensor] = None) -> Tensor:
    if noise_2ch is None:
        noise_2ch = torch.randn_like(x0_2ch)
    a_bar = alpha_bars[int(t)].to(device=x0_2ch.device, dtype=x0_2ch.dtype).view(1, 1, 1, 1)
    return torch.sqrt(a_bar) * x0_2ch + torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0)) * noise_2ch


# -----------------------------------------------------------------------------
# Data consistency helpers
# -----------------------------------------------------------------------------


def _sense_mask(sense_op: Any, fallback_mask: Optional[Tensor], *, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
    mask = getattr(sense_op, "mask", None)
    if mask is None:
        mask = fallback_mask
    return _mask_to_bhw(mask, B=B, H=H, W=W, device=device, dtype=dtype) if mask is not None else None


def _dc_residual(
    *,
    x_2ch: Tensor,
    y_2ch: Tensor,
    sense_op: Any,
    mask: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> Tensor:
    """Relative measured k-space residual per batch element."""
    x_2ch = _ensure_bchw_2ch(x_2ch, name="x_2ch")
    y_2ch = _ensure_bchw_2ch(y_2ch, name="y_2ch")
    B, _, H, W = x_2ch.shape
    mask_bhw = _sense_mask(sense_op, mask, B=B, H=H, W=W, device=x_2ch.device, dtype=x_2ch.dtype)
    y = ch2_to_complex(y_2ch)
    k = sense_op.forward(ch2_to_complex(x_2ch))
    if mask_bhw is not None:
        r = (k - y) * mask_bhw
        denom_k = y * mask_bhw
    else:
        r = k - y
        denom_k = y
    num = torch.linalg.vector_norm(r.reshape(B, -1), dim=1)
    den = torch.linalg.vector_norm(denom_k.reshape(B, -1), dim=1).clamp_min(eps)
    return num / den


def _call_data_consistency(
    *,
    sense_op: Any,
    x_pred_2ch: Tensor,
    y_2ch: Tensor,
    mask: Optional[Tensor],
    mode: str,
    lam: float,
) -> Tensor:
    """Call SenseOp.data_consistency with several historical signatures."""
    x_c = ch2_to_complex(x_pred_2ch)
    y_c = ch2_to_complex(y_2ch)
    mode = "replace" if mode == "hard" else mode
    mode = "grad" if mode == "soft" else mode

    # Try the newest style first.
    attempts = [
        lambda: sense_op.data_consistency(x_c, y=y_c, mode=mode, lam=lam),
        lambda: sense_op.data_consistency(x_c, y_c, mode=mode, lam=lam),
        lambda: sense_op.data_consistency(x_c, y_c, lam=lam, mask=mask, mode=mode),
        lambda: sense_op.data_consistency(x_c, y_c, mask, lam, mode),
    ]
    last_err: Optional[BaseException] = None
    for attempt in attempts:
        try:
            return complex_to_2ch(attempt())
        except TypeError as exc:
            last_err = exc
            continue
    raise TypeError(f"Could not call sense_op.data_consistency with a known signature. Last error: {last_err}")


def _cg_solve(mv, b: Tensor, x0: Tensor, *, max_iter: int = 10, tol: float = 1e-6) -> Tensor:
    """Conjugate gradient on a real 2-channel tensor [2,H,W]."""
    x = x0.clone()
    r = b - mv(x)
    p = r.clone()
    rsold = torch.sum(r * r)
    if torch.sqrt(rsold).item() < tol:
        return x
    tiny = torch.tensor(1e-12, device=b.device, dtype=b.dtype)
    for _ in range(int(max_iter)):
        Ap = mv(p)
        denom = torch.sum(p * Ap)
        if torch.abs(denom).item() < float(tiny):
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.sum(r * r)
        if torch.sqrt(rsnew).item() < tol:
            break
        p = r + (rsnew / rsold.clamp_min(tiny)) * p
        rsold = rsnew
    return x


def dc_prox_cg(
    *,
    x_pred_2ch: Tensor,
    y_meas_2ch: Tensor,
    sense_op: Any,
    lam: float,
    max_iter: int = 10,
    tol: float = 1e-6,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Proximal DC: ``(I + lam A^H A)x = x_pred + lam A^H y``.

    ``A`` is whatever your ``SenseOp.forward``/``adjoint`` pair implements.  If
    it includes the sampling mask, this is the masked MRI operator.  The measured
    k-space is multiplied by the mask before ``A^H y`` when a mask is available.
    """
    if lam <= 0:
        return x_pred_2ch

    x_pred_2ch = _ensure_bchw_2ch(x_pred_2ch, name="x_pred_2ch")
    y_meas_2ch = _ensure_bchw_2ch(y_meas_2ch, name="y_meas_2ch")
    B, _, H, W = x_pred_2ch.shape
    mask_bhw = _sense_mask(sense_op, mask, B=B, H=H, W=W, device=x_pred_2ch.device, dtype=x_pred_2ch.dtype)

    y = ch2_to_complex(y_meas_2ch)
    if mask_bhw is not None:
        y = y * mask_bhw
    Ah_y_2ch = complex_to_2ch(sense_op.adjoint(y))
    b = x_pred_2ch + float(lam) * Ah_y_2ch

    out: List[Tensor] = []
    for i in range(B):
        def mv(v_2ch: Tensor) -> Tensor:
            v = v_2ch.unsqueeze(0)
            Av = sense_op.forward(ch2_to_complex(v))
            AhAv = sense_op.adjoint(Av)
            return v_2ch + float(lam) * complex_to_2ch(AhAv)[0]

        out.append(_cg_solve(mv, b[i], x_pred_2ch[i], max_iter=max_iter, tol=tol))
    return torch.stack(out, dim=0)


# -----------------------------------------------------------------------------
# DDIM sampler
# -----------------------------------------------------------------------------


@torch.no_grad()
def ddim_with_dc_from_model(
    *,
    model: Any,
    sense_op: Any,
    y_k_2ch: Tensor,
    mask: Optional[Tensor] = None,
    x_init_2ch: Optional[Tensor] = None,
    cond: Optional[Tensor] = None,
    strength: Optional[float] = None,
    init_strength: Optional[float] = None,
    t_start: Optional[int] = None,
    num_steps: int = 50,
    dc_mode: str = "replace",
    dc_lam: Optional[float] = None,
    dc_lambda: Optional[float] = None,
    dc_cg_iter: int = 10,
    dc_cg_tol: float = 1e-6,
    log_residuals: bool = False,
    return_residuals: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[Tensor, Tuple[Tensor, List[Dict[str, float]]]]:
    """DDIM-like reverse diffusion with optional MRI data consistency.

    Parameters use both old and new names:
    - ``strength`` and ``init_strength`` are aliases;
    - ``dc_lam`` and ``dc_lambda`` are aliases;
    - ``log_residuals`` and ``return_residuals`` are aliases.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if return_residuals is None:
        return_residuals = bool(log_residuals)
    else:
        log_residuals = bool(return_residuals)

    if strength is None:
        strength = 0.2 if init_strength is None else float(init_strength)
    if dc_lam is None:
        dc_lam = 0.1 if dc_lambda is None else float(dc_lambda)

    dc_mode = str(dc_mode).lower()
    if dc_mode == "hard":
        dc_mode = "replace"
    if dc_mode == "soft":
        dc_mode = "grad"
    if dc_mode not in {"replace", "grad", "cg", "none", "off"}:
        raise ValueError("dc_mode must be one of: replace | hard | grad | soft | cg | none")

    model.eval().to(device)
    y_k_2ch = _ensure_bchw_2ch(y_k_2ch, name="y_k_2ch").to(device)
    B, _, H, W = y_k_2ch.shape
    dtype = y_k_2ch.dtype
    alpha_bars, T = _get_alpha_bars_and_T(model, device=device, dtype=dtype)

    mask_bhw = _mask_to_bhw(mask, B=B, H=H, W=W, device=device, dtype=dtype) if mask is not None else None

    if cond is not None:
        cond = cond.to(device=device, dtype=dtype)
        if cond.ndim != 4 or cond.shape[0] != B or cond.shape[-2:] != (H, W):
            raise ValueError(f"cond must be [B,C,H,W] matching y; got {tuple(cond.shape)}")

    if t_start is None:
        if x_init_2ch is None:
            # Full prior sampling starts from max noise.
            t_start = T - 1
        else:
            t_start = int(round(float(strength) * (T - 1)))
    t_start = int(max(0, min(T - 1, int(t_start))))

    if x_init_2ch is None:
        x_t = torch.randn((B, 2, H, W), device=device, dtype=dtype)
    else:
        x0 = _ensure_bchw_2ch(x_init_2ch, name="x_init_2ch").to(device=device, dtype=dtype)
        if x0.shape != (B, 2, H, W):
            raise ValueError(f"x_init_2ch must have shape {(B, 2, H, W)}, got {tuple(x0.shape)}")
        x_t = _q_sample_from_x0(x0, t_start, alpha_bars)

    t_seq = _make_ddim_timesteps(t_start, int(num_steps), device=device)
    residual_rows: List[Dict[str, float]] = []

    if len(t_seq) == 1:
        # strength=0 case.  Optionally still project once to measured data.
        if x_init_2ch is not None and dc_mode not in {"none", "off"}:
            before = _dc_residual(x_2ch=x_t, y_2ch=y_k_2ch, sense_op=sense_op, mask=mask_bhw).mean().item()
            if dc_mode == "cg":
                x_t = dc_prox_cg(x_pred_2ch=x_t, y_meas_2ch=y_k_2ch, sense_op=sense_op, lam=float(dc_lam), max_iter=dc_cg_iter, tol=dc_cg_tol, mask=mask_bhw)
            else:
                x_t = _call_data_consistency(sense_op=sense_op, x_pred_2ch=x_t, y_2ch=y_k_2ch, mask=mask_bhw, mode=dc_mode, lam=float(dc_lam))
            after = _dc_residual(x_2ch=x_t, y_2ch=y_k_2ch, sense_op=sense_op, mask=mask_bhw).mean().item()
            if log_residuals:
                residual_rows.append({"iter": 0, "t": 0, "before": float(before), "after": float(after)})
        return (x_t, residual_rows) if return_residuals else x_t

    for i in range(len(t_seq) - 1):
        t = int(t_seq[i])
        t_prev = int(t_seq[i + 1])
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        eps_pred = model(x_t, t_batch, cond=cond)

        a_t = alpha_bars[t].view(1, 1, 1, 1)
        a_prev = alpha_bars[t_prev].view(1, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(torch.clamp(1.0 - a_t, min=0.0)) * eps_pred) / torch.sqrt(a_t.clamp_min(1e-12))
        x_pred = torch.sqrt(a_prev) * x0_pred + torch.sqrt(torch.clamp(1.0 - a_prev, min=0.0)) * eps_pred

        if dc_mode in {"none", "off"}:
            x_dc = x_pred
            before = after = float("nan")
        else:
            before = _dc_residual(x_2ch=x_pred, y_2ch=y_k_2ch, sense_op=sense_op, mask=mask_bhw).mean().item()
            if dc_mode == "replace":
                x_dc = _call_data_consistency(sense_op=sense_op, x_pred_2ch=x_pred, y_2ch=y_k_2ch, mask=mask_bhw, mode="replace", lam=float(dc_lam))
            elif dc_mode == "grad":
                x_dc = _call_data_consistency(sense_op=sense_op, x_pred_2ch=x_pred, y_2ch=y_k_2ch, mask=mask_bhw, mode="grad", lam=float(dc_lam))
            elif dc_mode == "cg":
                x_dc = dc_prox_cg(x_pred_2ch=x_pred, y_meas_2ch=y_k_2ch, sense_op=sense_op, lam=float(dc_lam), max_iter=dc_cg_iter, tol=dc_cg_tol, mask=mask_bhw)
            else:  # pragma: no cover
                raise ValueError(dc_mode)
            after = _dc_residual(x_2ch=x_dc, y_2ch=y_k_2ch, sense_op=sense_op, mask=mask_bhw).mean().item()

        if log_residuals:
            residual_rows.append({"iter": i, "t": t, "t_prev": t_prev, "before": float(before), "after": float(after)})
        x_t = x_dc

    return (x_t, residual_rows) if return_residuals else x_t


@torch.no_grad()
def ddim_like_sampling_with_dc(
    model: Any,
    y_k_2ch: Tensor,
    mask: Tensor,
    sense_op: Any,
    num_steps: int = 50,
    dc_mode: str = "replace",
    dc_lambda: float = 1.0,
    x_init_2ch: Optional[Tensor] = None,
    init_strength: float = 1.0,
    t_start: Optional[int] = None,
    return_residuals: bool = False,
    cond: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, ResidualLog]]:
    """Legacy positional wrapper.

    Returns the old ``ResidualLog`` dataclass when ``return_residuals=True``.
    """
    out = ddim_with_dc_from_model(
        model=model,
        sense_op=sense_op,
        y_k_2ch=y_k_2ch,
        mask=mask,
        x_init_2ch=x_init_2ch,
        cond=cond,
        init_strength=init_strength,
        t_start=t_start,
        num_steps=num_steps,
        dc_mode=dc_mode,
        dc_lambda=dc_lambda,
        return_residuals=return_residuals,
        device=y_k_2ch.device,
    )
    if not return_residuals:
        return out  # type: ignore[return-value]
    x, rows = out  # type: ignore[misc]
    log = ResidualLog(
        timesteps=[int(r.get("t_prev", r.get("t", 0))) for r in rows],
        before_dc=[float(r["before"]) for r in rows],
        after_dc=[float(r["after"]) for r in rows],
    )
    return x, log


__all__ = [
    "ResidualLog",
    "ch2_to_complex",
    "complex_to_2ch",
    "ddim_like_sampling_with_dc",
    "ddim_with_dc_from_model",
    "dc_prox_cg",
]
