"""
Centered FFT utilities and a SENSE-like encoding operator for MRI reconstruction.

This module is the "shape contract" for reconstruction and data-consistency:

- Image space: complex [B, 1, H, W]   (single-coil) or [B, 1, H, W] (SENSE image)
- Coil sensitivities (optional): complex [B, C, H, W]
- k-space: complex [B, C, H, W]
- Mask: float/bool [B, 1, H, W] (broadcastable to [B, C, H, W])

Your diffusion model works on 2-channel real tensors [B, 2, H, W].
Use src.utils.complex_ops.{ch2_to_complex, complex_to_2ch} to convert
between representations when calling SenseOp.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _fftshift(x: torch.Tensor, dim=(-2, -1)) -> torch.Tensor:
    for d in dim:
        n = x.size(d)
        p2 = n - n // 2
        x = torch.roll(x, shifts=p2, dims=d)
    return x


def _ifftshift(x: torch.Tensor, dim=(-2, -1)) -> torch.Tensor:
    for d in dim:
        n = x.size(d)
        p2 = n // 2
        x = torch.roll(x, shifts=p2, dims=d)
    return x


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D FFT with orthonormal normalization.

    Accepts complex tensors with shape [..., H, W] and returns complex tensors
    of the same shape.
    """
    if not torch.is_complex(x):
        raise TypeError(f"fft2c expected complex input, got dtype={x.dtype}")
    x = _ifftshift(x, dim=(-2, -1))
    X = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    X = _fftshift(X, dim=(-2, -1))
    return X


def ifft2c(X: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT with orthonormal normalization."""
    if not torch.is_complex(X):
        raise TypeError(f"ifft2c expected complex input, got dtype={X.dtype}")
    X = _ifftshift(X, dim=(-2, -1))
    x = torch.fft.ifft2(X, dim=(-2, -1), norm="ortho")
    x = _fftshift(x, dim=(-2, -1))
    return x


@dataclass
class SenseOp:
    """
    SENSE-like forward model:

      A(x) = mask ⊙ F(mps ⊙ x)
      A^H(y) = Σ_c conj(mps_c) ⊙ F^{-1}(mask ⊙ y_c)

    For single-coil:
      mps = None (treated as 1)
      C = 1
    """
    mask: torch.Tensor  # [B, 1, H, W] or [1, H, W] broadcastable
    mps: Optional[torch.Tensor] = None  # [B, C, H, W] complex (optional)

    def _canon_mask(self, mask: Optional[torch.Tensor], B: int, C: int, H: int, W: int, device, dtype) -> torch.Tensor:
        m = self.mask if mask is None else mask
        # Accept [H,W], [1,H,W], [B,1,H,W], [B,C,H,W]
        if m.ndim == 2:
            m = m[None, None, ...]
        elif m.ndim == 3:
            # could be [1,H,W] or [B,H,W]
            if m.shape[0] == B:
                m = m[:, None, ...]
            else:
                m = m[None, ...]  # [1,H,W] -> [1,1,H,W]
        elif m.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported mask shape: {tuple(m.shape)}")

        m = m.to(device=device)
        # keep mask real
        if m.dtype not in (torch.float16, torch.float32, torch.float64, torch.bool):
            m = m.float()
        if m.dtype == torch.bool:
            m = m.to(torch.float32)
        else:
            m = m.to(torch.float32)

        # Broadcast to [B, C, H, W]
        if m.shape[0] == 1 and B > 1:
            m = m.expand(B, -1, -1, -1)
        if m.shape[1] == 1 and C > 1:
            m = m.expand(-1, C, -1, -1)
        if m.shape[1] != C:
            raise ValueError(f"Mask channel dim {m.shape[1]} not compatible with C={C}")
        if m.shape[-2:] != (H, W):
            raise ValueError(f"Mask spatial {tuple(m.shape[-2:])} != {(H,W)}")
        return m

    def forward(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward encoding.

        img: complex [B, 1, H, W]
        returns: complex [B, C, H, W]
        """
        if not torch.is_complex(img):
            raise TypeError(f"SenseOp.forward expects complex img, got dtype={img.dtype}")
        if img.ndim != 4 or img.shape[1] != 1:
            raise ValueError(f"img must be [B,1,H,W], got {tuple(img.shape)}")

        B, _, H, W = img.shape
        if self.mps is None:
            coil_img = img  # [B,1,H,W]
            C = 1
        else:
            if not torch.is_complex(self.mps):
                raise TypeError(f"mps must be complex, got {self.mps.dtype}")
            if self.mps.ndim != 4:
                raise ValueError(f"mps must be [B,C,H,W], got {tuple(self.mps.shape)}")
            if self.mps.shape[0] not in (1, B):
                raise ValueError(f"mps batch {self.mps.shape[0]} not compatible with B={B}")
            if self.mps.shape[-2:] != (H, W):
                raise ValueError(f"mps spatial {tuple(self.mps.shape[-2:])} != {(H,W)}")
            if self.mps.shape[0] == 1 and B > 1:
                mps = self.mps.expand(B, -1, -1, -1)
            else:
                mps = self.mps
            C = mps.shape[1]
            coil_img = mps * img  # [B,C,H,W]

        kspace = fft2c(coil_img)
        m = self._canon_mask(mask, B, C, H, W, device=kspace.device, dtype=torch.float32)
        return kspace * m

    def adjoint(self, kspace: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adjoint encoding.

        kspace: complex [B, C, H, W]
        returns: complex [B, 1, H, W]
        """
        if not torch.is_complex(kspace):
            raise TypeError(f"SenseOp.adjoint expects complex kspace, got dtype={kspace.dtype}")
        if kspace.ndim != 4:
            raise ValueError(f"kspace must be [B,C,H,W], got {tuple(kspace.shape)}")

        B, C, H, W = kspace.shape
        m = self._canon_mask(mask, B, C, H, W, device=kspace.device, dtype=torch.float32)
        coil_img = ifft2c(kspace * m)  # [B,C,H,W]

        if self.mps is None:
            # single coil: just return the image (ensure keepdim=1)
            if C != 1:
                # if caller passed multi-coil without mps, fall back to sum-of-coils
                return coil_img.sum(dim=1, keepdim=True)
            return coil_img
        else:
            mps = self.mps
            if mps.shape[0] == 1 and B > 1:
                mps = mps.expand(B, -1, -1, -1)
            if mps.shape[1] != C:
                raise ValueError(f"mps coils {mps.shape[1]} != kspace coils {C}")
            return (coil_img * torch.conj(mps)).sum(dim=1, keepdim=True)

    def data_consistency(
        self,
        img: torch.Tensor,
        y: torch.Tensor,
        lam: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        mode: str = "grad",
    ) -> torch.Tensor:
        """
        One data-consistency update step.

        img: complex [B,1,H,W] (current image estimate)
        y:   complex [B,C,H,W] (measured undersampled k-space; zeros where missing)
        lam: step size

        mode:
          - "grad": gradient step: x <- x + lam * A^H(y - A(x))
          - "replace": (single-coil only) hard k-space replacement
        """
        if not torch.is_complex(img) or not torch.is_complex(y):
            raise TypeError("data_consistency expects complex img and y")
        if img.ndim != 4 or img.shape[1] != 1:
            raise ValueError(f"img must be [B,1,H,W], got {tuple(img.shape)}")
        if y.ndim != 4:
            raise ValueError(f"y must be [B,C,H,W], got {tuple(y.shape)}")

        B, _, H, W = img.shape
        C = y.shape[1]
        m = self._canon_mask(mask, B, C, H, W, device=y.device, dtype=torch.float32)

        if mode == "replace" and self.mps is None and C == 1:
            # exact hard replacement for single-coil
            k_pred = fft2c(img)  # [B,1,H,W]
            k_new = k_pred * (1.0 - m) + y * m
            return ifft2c(k_new)

        # gradient step (works for single & multi-coil)
        Ax = self.forward(img, mask=m)  # already masked
        resid = (y * m) - Ax            # ensure only measured points contribute
        grad = self.adjoint(resid, mask=m)
        return img + lam * grad
