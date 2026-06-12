"""
Utilities for converting between complex tensors and 2-channel (real/imag) tensors.

Conventions used throughout this project:

- Complex tensors use torch.complex64/complex128, with shape:
    [B, C, H, W]  (C = coil channels OR 1 for single-coil image space)
  or without batch: [C, H, W] / [H, W].

- 2-channel tensors represent complex values by concatenating real and imaginary
  parts along the channel dimension:
    complex [B, C, H, W] -> 2ch [B, 2*C, H, W]
    complex [C, H, W]    -> 2ch [2*C, H, W]
    complex [H, W]       -> 2ch [2, H, W]

This matches the shapes used by your dataset2 loader:
- single-coil k-space is returned as [2, H, W]
- single-coil image targets are returned as [2, H, W]
"""
from __future__ import annotations

import torch


def complex_to_2ch(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to a real-valued 2-channel representation."""
    if not torch.is_complex(x):
        raise TypeError(f"complex_to_2ch expected complex tensor, got dtype={x.dtype}")

    if x.ndim == 4:  # [B, C, H, W]
        return torch.cat([x.real, x.imag], dim=1)
    if x.ndim == 3:  # [C, H, W]
        return torch.cat([x.real, x.imag], dim=0)
    if x.ndim == 2:  # [H, W]
        return torch.stack([x.real, x.imag], dim=0)

    raise ValueError(f"Unsupported shape for complex_to_2ch: {tuple(x.shape)}")


def ch2_to_complex(x2: torch.Tensor) -> torch.Tensor:
    """
    Convert a 2-channel (real/imag) tensor back to complex.

    Accepts:
      - [B, 2*C, H, W] -> [B, C, H, W] complex
      - [2*C, H, W]    -> [C, H, W] complex
      - [2, H, W]      -> [1, H, W] complex
    """
    if torch.is_complex(x2):
        return x2

    if x2.ndim == 4:  # [B, 2*C, H, W]
        if x2.shape[1] % 2 != 0:
            raise ValueError(f"Channel dim must be even for ch2_to_complex, got {x2.shape[1]}")
        C = x2.shape[1] // 2
        real = x2[:, :C, ...]
        imag = x2[:, C:, ...]
        return torch.complex(real, imag)

    if x2.ndim == 3:  # [2*C, H, W]
        if x2.shape[0] % 2 != 0:
            raise ValueError(f"Channel dim must be even for ch2_to_complex, got {x2.shape[0]}")
        C = x2.shape[0] // 2
        real = x2[:C, ...]
        imag = x2[C:, ...]
        return torch.complex(real, imag)

    raise ValueError(f"Unsupported shape for ch2_to_complex: {tuple(x2.shape)}")


# -----------------------------------------------------------------------------
# Backward-compatible aliases
# -----------------------------------------------------------------------------
# Some scripts (or earlier iterations of this repo) use slightly different
# naming conventions. These aliases keep those scripts working.


def complex_to_twoch(x: torch.Tensor) -> torch.Tensor:
    return complex_to_2ch(x)


def twoch_to_complex(x2: torch.Tensor) -> torch.Tensor:
    return ch2_to_complex(x2)


def complex_to_ch2(x: torch.Tensor) -> torch.Tensor:
    return complex_to_2ch(x)


def ch2_to_complex64(x2: torch.Tensor) -> torch.Tensor:
    """Explicit name used in some codebases; returns complex tensor."""
    return ch2_to_complex(x2)

