# src/utils/mask_utils_v2.py

"""
mask_utils.py

Small utilities for generating and handling k-space sampling masks in a way that
matches this project’s conventions.

Why this exists
---------------
You already have `generate_sampling_1D_mask.py` (ported from the reference repo).
That script generates NumPy masks with shape [Nx, Ny, Nt]. In *your* pipeline:

- k-space tensors are typically [B, 2, H, W] (single-coil 2ch real/imag), or
  [B, C, H, W] complex (multi-coil) inside `SenseOp`.
- masks are typically [H, W] or [B, H, W] (and `SenseOp` canonizes them to
  [B, 1, H, W]).

So this module wraps the reference mask generator and returns torch tensors in
the expected shapes.

Main API
--------
- generate_mask_1d(...): returns a 1D undersampling mask as torch.Tensor
  with shape [H, W] or [B, H, W], values in {0,1}.
- ensure_mask_bhw(...): convert various mask shapes to [B, H, W].
- mask_to_senseop(...): convert [H,W] or [B,H,W] to [B,1,H,W] for SenseOp.
- estimate_acceleration(...): compute empirical acceleration = HW / (#samples)

Notes
-----
- The underlying `generate_sampling_1D_mask` uses NumPy's *global* RNG
  (np.random.*). If you pass `seed=...`, we temporarily set and restore the RNG
  state to keep runs reproducible without permanently changing global state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

import importlib.util
from pathlib import Path as _Path

# -----------------------------------------------------------------------------
# Import the reference generator robustly (works whether or not /mnt/data is on PYTHONPATH)
# -----------------------------------------------------------------------------
_generate_sampling_1D_mask = None

# 1) Try standard import (works when module is on PYTHONPATH)
try:  # pragma: no cover
    from generate_sampling_1D_mask import generate_sampling_1D_mask as _generate_sampling_1D_mask  # type: ignore
except Exception:  # pragma: no cover
    _generate_sampling_1D_mask = None

# 2) Try to load from a sibling file next to this module
if _generate_sampling_1D_mask is None:  # pragma: no cover
    _here = _Path(__file__).resolve().parent
    _cand = _here / "generate_sampling_1D_mask.py"
    if _cand.exists():
        spec = importlib.util.spec_from_file_location("generate_sampling_1D_mask", str(_cand))
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
        _generate_sampling_1D_mask = getattr(mod, "generate_sampling_1D_mask", None)

# 3) Optional src.* layout
if _generate_sampling_1D_mask is None:  # pragma: no cover
    try:
        from src.utils.generate_sampling_1D_mask import generate_sampling_1D_mask as _generate_sampling_1D_mask  # type: ignore
    except Exception:
        _generate_sampling_1D_mask = None

if _generate_sampling_1D_mask is None:
    raise ImportError(
        "Could not import generate_sampling_1D_mask. Ensure generate_sampling_1D_mask.py is available "
        "next to mask_utils.py or on your PYTHONPATH."
    )

# Expose under the expected name
generate_sampling_1D_mask = _generate_sampling_1D_mask



# NOTE: your current `generate_sampling_1D_mask.py` (per your screenshot)
# supports only these mask types:
#   - 'out_r_1D'
#   - 'on_in_1D'
#   - 'full_sam'
#
# We also accept a few friendly aliases and map them onto the supported set.
MaskType = Literal[
    "out_r_1D",
    "on_in_1D",
    "full_sam",
    # aliases
    "random_1D",
    "inner_1D",
    "full",
]


@dataclass
class MaskSpec:
    """Lightweight config for a 1D mask."""
    accel: int
    mask_type: MaskType = "out_r_1D"
    center: bool = True
    # number of time frames / repetitions (useful for dynamic sequences)
    nt: int = 1
    seed: Optional[int] = None


def _with_numpy_seed(seed: Optional[int]):
    """Context manager-ish helper to temporarily set NumPy global RNG seed."""
    class _Ctx:
        def __init__(self, seed_):
            self.seed = seed_
            self.state = None

        def __enter__(self):
            if self.seed is None:
                return
            self.state = np.random.get_state()
            np.random.seed(self.seed)

        def __exit__(self, exc_type, exc, tb):
            if self.state is not None:
                np.random.set_state(self.state)

    return _Ctx(seed)


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
    """
    Generate a 1D undersampling mask compatible with your pipeline.

    Parameters
    ----------
    H, W:
        Spatial size of k-space (H rows, W columns). The generated mask samples
        *entire columns* (phase-encode direction), i.e. it is constant along H.
    accel:
        Intended acceleration factor (e.g. 4, 8, 10).
    mask_type:
        Passed through to `generate_sampling_1D_mask`. Common choice: "out_r_1D".
    center:
        If True, ensures a fully-sampled central region (depends on mask_type).
    nt:
        Time dimension used by the reference generator (default 1). If nt>1,
        this returns one mask per time frame (then you can pick a frame or
        broadcast).
    batch_size:
        If provided, returns shape [B,H,W] by repeating the same mask for the
        whole batch. (If you want *different* masks per batch item, call this
        repeatedly with different seeds.)
    seed:
        Optional random seed for reproducibility.
    device, dtype:
        Torch placement and dtype.

    Returns
    -------
    mask:
        torch.Tensor of shape [H,W] (or [B,H,W] if batch_size is set),
        with values 0/1 in `dtype`.
    """
    if accel <= 0:
        raise ValueError("accel must be a positive integer")
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive integers")
    if nt <= 0:
        raise ValueError("nt must be >= 1")

    if mask_type == "full":
        # alias for full sampling
        mask_type = "full_sam"

    # Map aliases to the supported generator types
    if mask_type == "random_1D":
        mask_type = "out_r_1D"
    if mask_type == "inner_1D":
        mask_type = "on_in_1D"

    if mask_type == "full_sam":
        m = np.ones((H, W), dtype=np.float32)
    else:
        # reference generator expects dim=[Nx, Ny, Nt]
        dim = [int(H), int(W), int(nt)]
        with _with_numpy_seed(seed):
            m_np = generate_sampling_1D_mask(dim, int(accel), str(mask_type), center=center)
        # m_np is [H,W,nt]; take first frame by default if nt>1
        if m_np.ndim != 3 or m_np.shape[0] != H or m_np.shape[1] != W:
            raise RuntimeError(f"Unexpected mask shape from generator: {m_np.shape}, expected {(H,W,nt)}")
        m = m_np[:, :, 0].astype(np.float32)

    mask = torch.from_numpy(m).to(device=device) if device is not None else torch.from_numpy(m)
    mask = mask.to(dtype=dtype)

    # Ensure strict binary (some generators could return floats close to 0/1)
    mask = (mask > 0.5).to(dtype)

    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        mask = mask.unsqueeze(0).repeat(int(batch_size), 1, 1)  # [B,H,W]

    return mask


def ensure_mask_bhw(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Convert masks of shape [H,W], [1,H,W], [B,H,W] into [B,H,W].

    This matches how you handle masks in `run_sampling.py` before passing them
    into `SenseOp`.
    """
    if mask.ndim == 2:
        mask_b = mask.unsqueeze(0)  # [1,H,W]
    elif mask.ndim == 3:
        mask_b = mask
    elif mask.ndim == 4 and mask.shape[1] == 1:
        # [B,1,H,W] -> [B,H,W]
        mask_b = mask[:, 0]
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    if mask_b.shape[0] == 1 and batch_size != 1:
        mask_b = mask_b.repeat(batch_size, 1, 1)

    if mask_b.shape[0] != batch_size:
        raise ValueError(f"Mask batch ({mask_b.shape[0]}) does not match batch_size ({batch_size})")

    # Ensure 0/1
    mask_b = (mask_b > 0.5).to(mask_b.dtype)
    return mask_b


def mask_to_senseop(mask: torch.Tensor, batch_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert [H,W] or [B,H,W] -> [B,1,H,W], which is what `SenseOp` will canonize to.
    """
    m = ensure_mask_bhw(mask, batch_size=batch_size).to(dtype=dtype)
    return m.unsqueeze(1)  # [B,1,H,W]


def estimate_acceleration(mask: torch.Tensor) -> float:
    """
    Empirical acceleration = (H*W)/(#sampled points), computed on a single mask.

    For a 1D mask sampled by full columns, this reflects your actual sampling
    density (central fully-sampled band will reduce accel slightly).
    """
    if mask.ndim == 3:
        # take first in batch
        mask2d = mask[0]
    elif mask.ndim == 2:
        mask2d = mask
    elif mask.ndim == 4 and mask.shape[1] == 1:
        mask2d = mask[0, 0]
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    H, W = int(mask2d.shape[0]), int(mask2d.shape[1])
    sampled = float((mask2d > 0.5).sum().item())
    total = float(H * W)
    if sampled <= 0:
        return float("inf")
    return total / sampled


def apply_mask_kspace_2ch(y_2ch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a binary mask to 2ch k-space.

    y_2ch: [B,2,H,W] or [2,H,W]
    mask:  [H,W] or [B,H,W] or [B,1,H,W]
    returns: same shape as y_2ch
    """
    if y_2ch.ndim == 3:
        y = y_2ch.unsqueeze(0)
        squeeze_back = True
    else:
        y = y_2ch
        squeeze_back = False

    if y.ndim != 4 or y.shape[1] != 2:
        raise ValueError(f"Expected y_2ch shape [B,2,H,W] (or [2,H,W]), got {tuple(y_2ch.shape)}")

    B, _, H, W = y.shape
    m = ensure_mask_bhw(mask, batch_size=B).to(device=y.device, dtype=y.dtype)  # [B,H,W]
    m = m.unsqueeze(1)  # [B,1,H,W]
    y_masked = y * m

    if squeeze_back:
        return y_masked.squeeze(0)
    return y_masked
