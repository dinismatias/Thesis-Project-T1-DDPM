# src/utils/metrics.py
"""Shared magnitude-image quality metrics for the T1_DDPM / DiMo pipeline.

Pure-PyTorch implementations (no scikit-image / SciPy dependency) so the exact
same numbers are produced on Windows, on the Linux VM, and on CPU or GPU.

All metrics operate on the *magnitude* of the reconstruction relative to a
reference. Inputs may be:
  - 2-channel complex images  [B,2,H,W]  (real, imag)  -> magnitude computed here
  - already-magnitude maps    [B,H,W] or [B,1,H,W] or [H,W]

Provided metrics
----------------
- nmse  : normalized mean squared error            (lower is better)
- nrmse : sqrt(nmse)                                (lower is better)
- psnr  : peak SNR in dB w.r.t. the reference peak  (higher is better)
- ssim  : structural similarity, Gaussian window,
          single scale, data_range = reference peak (higher is better)
- hfen  : High-Frequency Error Norm, relative; the L2 norm of the difference of
          Laplacian-of-Gaussian filtered images, normalized by the LoG-filtered
          reference norm so it is comparable across intensity scales
          (lower is better)

Primary entry point
-------------------
``compute_image_metrics(x, ref, suffix="_mag")`` returns a flat dict with keys
``nmse_mag``, ``nrmse_mag``, ``psnr_mag``, ``ssim_mag``, ``hfen_mag`` (the
suffix is configurable). If ``ref`` is None, every value is None so callers can
emit a stable schema even when no target is available.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Shape / magnitude helpers
# -----------------------------------------------------------------------------


def _to_magnitude(x: Tensor) -> Tensor:
    """Return a magnitude batch [B,H,W] from a variety of input layouts."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().float()

    if x.ndim == 2:
        # [H,W] real magnitude
        return x.unsqueeze(0)
    if x.ndim == 3:
        if x.shape[0] == 2:
            # [2,H,W] complex -> magnitude
            return torch.sqrt(x[0] ** 2 + x[1] ** 2).unsqueeze(0)
        # [B,H,W] magnitude batch
        return x
    if x.ndim == 4:
        if x.shape[1] == 2:
            # [B,2,H,W] complex -> magnitude
            return torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        if x.shape[1] == 1:
            # [B,1,H,W] magnitude
            return x[:, 0]
    raise ValueError(f"Unsupported tensor shape for magnitude metric: {tuple(x.shape)}")


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    win2d = torch.outer(g, g)
    return win2d.view(1, 1, window_size, window_size)


def _log_kernel(size: int, sigma: float, device, dtype) -> Tensor:
    """Zero-mean Laplacian-of-Gaussian kernel used for HFEN."""
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    r2 = xx ** 2 + yy ** 2
    g = torch.exp(-r2 / (2.0 * sigma ** 2))
    log = (r2 - 2.0 * sigma ** 2) / (sigma ** 4) * g
    log = log - log.mean()  # enforce zero DC response
    return log.view(1, 1, size, size)


# -----------------------------------------------------------------------------
# Individual metrics (operate on magnitude batches [B,H,W])
# -----------------------------------------------------------------------------


def _nmse(x: Tensor, ref: Tensor, eps: float) -> Tensor:
    num = torch.sum((x - ref) ** 2, dim=(-2, -1))
    den = torch.sum(ref ** 2, dim=(-2, -1)).clamp_min(eps)
    return num / den


def _psnr(x: Tensor, ref: Tensor, data_range: Tensor, eps: float) -> Tensor:
    mse = torch.mean((x - ref) ** 2, dim=(-2, -1)).clamp_min(eps)
    return 20.0 * torch.log10(data_range.clamp_min(eps)) - 10.0 * torch.log10(mse)


def _ssim(
    x: Tensor,
    ref: Tensor,
    data_range: Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Single-scale Gaussian-window SSIM, returned per batch element."""
    device, dtype = x.device, x.dtype
    win = _gaussian_window(window_size, sigma, device, dtype)
    pad = window_size // 2

    xb = x.unsqueeze(1)
    yb = ref.unsqueeze(1)

    mu_x = F.conv2d(xb, win, padding=pad)
    mu_y = F.conv2d(yb, win, padding=pad)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(xb * xb, win, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(yb * yb, win, padding=pad) - mu_y2
    sigma_xy = F.conv2d(xb * yb, win, padding=pad) - mu_xy

    dr = data_range.view(-1, 1, 1, 1)
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean(dim=(-3, -2, -1))


def _hfen(
    x: Tensor,
    ref: Tensor,
    *,
    size: int = 15,
    sigma: float = 1.5,
    eps: float = 1e-12,
) -> Tensor:
    """Relative High-Frequency Error Norm, returned per batch element."""
    device, dtype = x.device, x.dtype
    k = _log_kernel(size, sigma, device, dtype)
    pad = size // 2

    lx = F.conv2d(x.unsqueeze(1), k, padding=pad)
    ly = F.conv2d(ref.unsqueeze(1), k, padding=pad)

    num = torch.sqrt(torch.sum((lx - ly) ** 2, dim=(-3, -2, -1)))
    den = torch.sqrt(torch.sum(ly ** 2, dim=(-3, -2, -1))).clamp_min(eps)
    return num / den


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def compute_image_metrics(
    x,
    ref,
    *,
    suffix: str = "",
    data_range: Optional[float] = None,
    eps: float = 1e-12,
) -> Dict[str, Optional[float]]:
    """Compute NMSE / NRMSE / PSNR / SSIM / HFEN on magnitude images.

    Parameters
    ----------
    x, ref:
        Reconstruction and reference. Accept [B,2,H,W] complex, [B,1,H,W],
        [B,H,W] or [H,W] magnitude. If ``ref`` is None, all metrics are None.
    suffix:
        Appended to each metric key, e.g. ``"_mag"`` -> ``"nmse_mag"``.
    data_range:
        Peak value for PSNR/SSIM. Defaults to the reference magnitude max
        (per the whole batch), matching the historical PSNR convention here.
    """
    keys = ["nmse", "nrmse", "psnr", "ssim", "hfen"]
    if ref is None:
        return {f"{k}{suffix}": None for k in keys}

    x_mag = _to_magnitude(x)
    ref_mag = _to_magnitude(ref)
    if x_mag.shape != ref_mag.shape:
        raise ValueError(
            f"Reconstruction and reference magnitudes differ in shape: "
            f"{tuple(x_mag.shape)} vs {tuple(ref_mag.shape)}"
        )

    if data_range is None:
        dr = ref_mag.amax(dim=(-2, -1)).clamp_min(eps)
    else:
        dr = torch.full(
            (ref_mag.shape[0],), float(data_range), device=ref_mag.device, dtype=ref_mag.dtype
        ).clamp_min(eps)

    nmse = _nmse(x_mag, ref_mag, eps)
    nrmse = torch.sqrt(nmse)
    psnr = _psnr(x_mag, ref_mag, dr, eps)
    ssim = _ssim(x_mag, ref_mag, dr)
    hfen = _hfen(x_mag, ref_mag, eps=eps)

    return {
        f"nmse{suffix}": float(nmse.mean().item()),
        f"nrmse{suffix}": float(nrmse.mean().item()),
        f"psnr{suffix}": float(psnr.mean().item()),
        f"ssim{suffix}": float(ssim.mean().item()),
        f"hfen{suffix}": float(hfen.mean().item()),
    }


__all__ = ["compute_image_metrics"]
