# src/utils/mask_utils.py
"""
Canonical mask utilities for the T1_DDPM / DiMo reconstruction pipeline.

This file merges the useful behavior from mask_utils.py and mask_utils_v2.py:
- robust import of generate_sampling_1D_mask.py;
- friendly aliases such as random_1D, inner_1D, full;
- conversion between [H,W], [B,H,W], [B,1,H,W];
- application of masks to 2-channel k-space tensors;
- empirical acceleration reporting.

The project convention is:
- image/k-space tensors: [B,2,H,W] for real/imag single-coil tensors;
- masks: [H,W] or [B,H,W], converted to [B,1,H,W] for SenseOp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Literal, Optional, Union
import importlib.util
import warnings

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Robust reference-generator import
# -----------------------------------------------------------------------------

_generate_sampling_1D_mask = None

try:  # common when running from src/utils
    from generate_sampling_1D_mask import generate_sampling_1D_mask as _generate_sampling_1D_mask  # type: ignore
except Exception:
    _generate_sampling_1D_mask = None

if _generate_sampling_1D_mask is None:  # sibling file next to this module
    _here = _Path(__file__).resolve().parent
    _cand = _here / "generate_sampling_1D_mask.py"
    if _cand.exists():
        spec = importlib.util.spec_from_file_location("generate_sampling_1D_mask", str(_cand))
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            _generate_sampling_1D_mask = getattr(mod, "generate_sampling_1D_mask", None)

if _generate_sampling_1D_mask is None:  # package layout
    try:
        from src.utils.generate_sampling_1D_mask import generate_sampling_1D_mask as _generate_sampling_1D_mask  # type: ignore
    except Exception:
        _generate_sampling_1D_mask = None

# Expose the name expected by older code. It may be None; generate_mask_1d has a
# safe fallback so the whole module does not hard-crash if the reference file is
# temporarily missing.
generate_sampling_1D_mask = _generate_sampling_1D_mask

MaskType = Literal[
    # Reference / historical names
    "out_r_1D",
    "out_c_1D",
    "out_r_1D_unif",
    "out_c_1D_unif",
    "out_r_1D_rand",
    "out_c_1D_rand",
    "out_c_1D_rand_center",
    "on_in_1D",
    "full_sam",
    # Friendly aliases
    "random_1D",
    "inner_1D",
    "full",
]


@dataclass(frozen=True)
class MaskSpec:
    accel: int
    mask_type: MaskType = "out_r_1D"
    center: bool = True
    nt: int = 1
    seed: Optional[int] = None


def _with_numpy_seed(seed: Optional[int]):
    """Temporarily set NumPy global RNG seed, restoring it afterwards."""

    class _Ctx:
        def __init__(self, seed_: Optional[int]):
            self.seed = seed_
            self.state = None

        def __enter__(self):
            if self.seed is not None:
                self.state = np.random.get_state()
                np.random.seed(int(self.seed))

        def __exit__(self, exc_type, exc, tb):
            if self.state is not None:
                np.random.set_state(self.state)

    return _Ctx(seed)


def normalize_mask_type(mask_type: str) -> str:
    """Map friendly aliases to generator-compatible names."""
    mt = str(mask_type)
    aliases = {
        "full": "full_sam",
        "full_sampling": "full_sam",
        "random_1D": "out_r_1D",
        "random": "out_r_1D",
        "inner_1D": "on_in_1D",
        "inner": "on_in_1D",
    }
    return aliases.get(mt, mt)


def _fallback_variable_density_mask(H: int, W: int, accel: int, *, center: bool, seed: Optional[int]) -> np.ndarray:
    """Simple deterministic fallback when generate_sampling_1D_mask.py is unavailable.

    It samples full phase-encode columns, keeps a small central calibration band,
    and randomly chooses the remaining columns. This is not the official challenge
    mask generator, but it keeps debugging scripts usable.
    """
    rng = np.random.default_rng(seed)
    target_cols = max(1, int(round(W / max(1, accel))))
    mask_cols = np.zeros(W, dtype=np.float32)

    if center:
        center_cols = max(2, min(W, int(round(W * 0.06))))
        center_cols = min(center_cols, target_cols)
        c0 = W // 2 - center_cols // 2
        mask_cols[c0 : c0 + center_cols] = 1.0
    else:
        center_cols = 0

    remaining = max(0, target_cols - int(mask_cols.sum()))
    candidates = np.flatnonzero(mask_cols < 0.5)
    if remaining > 0 and candidates.size > 0:
        chosen = rng.choice(candidates, size=min(remaining, candidates.size), replace=False)
        mask_cols[chosen] = 1.0

    return np.tile(mask_cols[None, :], (H, 1)).astype(np.float32)


def generate_mask_1d(
    H: int,
    W: int,
    *,
    accel: int,
    mask_type: MaskType = "out_r_1D",
    center: bool = True,
    nt: int = 1,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a 1D phase-encode undersampling mask.

    Returns [H,W] or [B,H,W] if batch_size is provided. Values are exactly 0/1.
    """
    H, W, accel, nt = int(H), int(W), int(accel), int(nt)
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")
    if accel <= 0:
        raise ValueError("accel must be positive")
    if nt <= 0:
        raise ValueError("nt must be >= 1")

    mt = normalize_mask_type(str(mask_type))

    if mt == "full_sam":
        m = np.ones((H, W), dtype=np.float32)
    elif generate_sampling_1D_mask is not None:
        dim = [H, W, nt]
        with _with_numpy_seed(seed):
            try:
                m_np = generate_sampling_1D_mask(dim, accel, mt, center=center)  # type: ignore[misc]
            except TypeError:
                # Some reference copies do not accept keyword center.
                m_np = generate_sampling_1D_mask(dim, accel, mt)  # type: ignore[misc]
        if not isinstance(m_np, np.ndarray):
            m_np = np.asarray(m_np)
        if m_np.ndim == 2:
            m = m_np.astype(np.float32)
        elif m_np.ndim == 3:
            if m_np.shape[0] == H and m_np.shape[1] == W:
                m = m_np[:, :, 0].astype(np.float32)
            elif m_np.shape[-2:] == (H, W):
                m = m_np[0].astype(np.float32)
            else:
                raise RuntimeError(f"Unexpected reference mask shape {m_np.shape}; expected H/W {(H, W)}")
        else:
            raise RuntimeError(f"Unexpected reference mask shape {m_np.shape}; expected 2D or 3D")
    else:
        warnings.warn(
            "generate_sampling_1D_mask.py not found; using a simple fallback random 1D mask. "
            "Use the official generator for final experiments.",
            RuntimeWarning,
        )
        m = _fallback_variable_density_mask(H, W, accel, center=center, seed=seed)

    if m.shape != (H, W):
        raise RuntimeError(f"Mask shape {m.shape} does not match expected {(H, W)}")

    mask = torch.from_numpy((m > 0.5).astype(np.float32))
    if device is not None:
        mask = mask.to(device=device)
    mask = mask.to(dtype=dtype)

    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return mask


def ensure_mask_bhw(mask: torch.Tensor, batch_size: int, *, H: Optional[int] = None, W: Optional[int] = None) -> torch.Tensor:
    """Convert [H,W], [1,H,W], [B,H,W], or [B,1,H,W] to [B,H,W]."""
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)

    if mask.ndim == 2:
        m = mask.unsqueeze(0)
    elif mask.ndim == 3:
        m = mask
    elif mask.ndim == 4 and mask.shape[1] == 1:
        m = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        # Rare MATLAB-like [B,H,W,1]
        m = mask[..., 0]
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    if H is not None and W is not None and tuple(m.shape[-2:]) != (int(H), int(W)):
        raise ValueError(f"Mask spatial shape {tuple(m.shape[-2:])} does not match {(int(H), int(W))}")

    batch_size = int(batch_size)
    if m.shape[0] == 1 and batch_size != 1:
        m = m.expand(batch_size, -1, -1)
    if m.shape[0] != batch_size:
        raise ValueError(f"Mask batch {m.shape[0]} does not match batch_size {batch_size}")

    return (m > 0.5).to(dtype=mask.dtype, device=mask.device)


def mask_to_senseop(mask: torch.Tensor, batch_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert any supported mask representation to [B,1,H,W]."""
    return ensure_mask_bhw(mask, batch_size=batch_size).to(dtype=dtype).unsqueeze(1)


def estimate_acceleration(mask: torch.Tensor) -> float:
    """Empirical acceleration = total points / sampled points on first mask."""
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask2d = mask[0, 0]
    elif mask.ndim == 3:
        mask2d = mask[0]
    elif mask.ndim == 2:
        mask2d = mask
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")
    sampled = float((mask2d > 0.5).sum().item())
    if sampled <= 0:
        return float("inf")
    return float(mask2d.numel()) / sampled


def sampled_fraction(mask: torch.Tensor) -> float:
    """Fraction of sampled k-space points in the first mask."""
    acc = estimate_acceleration(mask)
    if not np.isfinite(acc) or acc <= 0:
        return 0.0
    return 1.0 / acc


def apply_mask_kspace_2ch(y_2ch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a binary mask to [B,2,H,W] or [2,H,W] real/imag k-space."""
    if y_2ch.ndim == 3:
        y = y_2ch.unsqueeze(0)
        squeeze = True
    elif y_2ch.ndim == 4:
        y = y_2ch
        squeeze = False
    else:
        raise ValueError(f"Expected y_2ch [2,H,W] or [B,2,H,W], got {tuple(y_2ch.shape)}")
    if y.shape[1] != 2:
        raise ValueError(f"Expected channel dim=2, got {tuple(y.shape)}")
    B, _, H, W = y.shape
    m = ensure_mask_bhw(mask, batch_size=B, H=H, W=W).to(device=y.device, dtype=y.dtype).unsqueeze(1)
    out = y * m
    return out.squeeze(0) if squeeze else out


__all__ = [
    "MaskSpec",
    "MaskType",
    "generate_sampling_1D_mask",
    "normalize_mask_type",
    "generate_mask_1d",
    "ensure_mask_bhw",
    "mask_to_senseop",
    "estimate_acceleration",
    "sampled_fraction",
    "apply_mask_kspace_2ch",
]
