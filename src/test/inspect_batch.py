# src/recon/inspect_batch.py
"""
Quick sanity check: plot zero-filled reconstruction vs DDPM target.

This catches most silent bugs (axis permutations, scaling, complex<->2ch mistakes).
It uses the same dataset arguments as train_dimo.py.

Example:
  python -m src.recon.inspect_batch --acc_root "C:/.../AccFactor04" --acc_factor 04 --index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.recon.dimo_dataset import DimoKspaceDataset


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """Centered 2D IFFT with ortho norm. k: [..., H, W] complex."""
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    x = torch.fft.ifft2(k, dim=(-2, -1), norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def twoch_to_complex(x2: torch.Tensor) -> torch.Tensor:
    """
    Convert 2ch tensor [B, 2*C, H, W] (real then imag) -> complex [B, C, H, W].
    """
    if x2.ndim != 4:
        raise ValueError(f"Expected 4D [B,2*C,H,W], got {tuple(x2.shape)}")
    ch = x2.shape[1]
    if ch % 2 != 0:
        raise ValueError(f"Channel dim must be even, got {ch}")
    C = ch // 2
    real = x2[:, :C, ...]
    imag = x2[:, C:, ...]
    return torch.complex(real, imag)


def magnitude(xc: torch.Tensor) -> torch.Tensor:
    return torch.abs(xc)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect a dataset batch (zf vs target).")
    p.add_argument("--acc_root", type=str, required=True)
    p.add_argument("--acc_factor", type=str, default="04")
    p.add_argument("--multi_coil", action="store_true")
    p.add_argument("--use_full_as_target", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--index", type=int, default=0, help="Dataset index to visualize (uses a non-shuffled loader).")
    return p


def main(args: argparse.Namespace) -> None:
    acc_root = Path(args.acc_root)
    case_dirs = sorted([p for p in acc_root.glob("P*") if p.is_dir()])
    if not case_dirs:
        raise FileNotFoundError(f"No P* dirs under {acc_root}")

    ds = DimoKspaceDataset(
        case_dirs=case_dirs,
        acc_factor=args.acc_factor,
        multi_coil=args.multi_coil,
        use_full_as_target=args.use_full_as_target,
    )

    # Fetch a specific index deterministically
    sample = ds[args.index]
    k2 = sample["kspace_und"].unsqueeze(0).float()  # [1,2*C,H,W] real-valued 2ch
    tgt2 = sample["img_target"].unsqueeze(0).float()  # [1,2,H,W]
    mask = sample["mask"].squeeze(0).cpu().numpy()  # [H,W]

    k_c = twoch_to_complex(k2)  # [1,C,H,W]
    img_c = ifft2c(k_c)         # [1,C,H,W]

    if img_c.shape[1] > 1:
        img_zf = torch.sqrt(torch.sum(torch.abs(img_c) ** 2, dim=1, keepdim=True))  # [1,1,H,W]
    else:
        img_zf = torch.abs(img_c)  # [1,1,H,W]

    tgt_c = twoch_to_complex(tgt2)      # [1,1,H,W]
    img_tgt = torch.abs(tgt_c)          # [1,1,H,W]

    zf_np = img_zf[0, 0].cpu().numpy()
    tgt_np = img_tgt[0, 0].cpu().numpy()
    diff_np = np.abs(zf_np - tgt_np)

    vmin = float(np.percentile(tgt_np, 1))
    vmax = float(np.percentile(tgt_np, 99))

    plt.figure()
    plt.imshow(zf_np, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Zero-filled magnitude")
    plt.axis("off")

    plt.figure()
    plt.imshow(tgt_np, cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Target magnitude (img_target)")
    plt.axis("off")

    plt.figure()
    plt.imshow(diff_np, cmap="gray")
    plt.title("|ZF - target|")
    plt.axis("off")

    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.title("Sampling mask")
    plt.axis("off")

    plt.show()

def save_pt_bundle(out_dir, fname, x_in_2ch, x_target_2ch=None, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "x_in_2ch": x_in_2ch.detach().cpu(),   # [1,2,H,W]
    }
    if x_target_2ch is not None:
        payload["x_target_2ch"] = x_target_2ch.detach().cpu()

    if extra is not None:
        # extra can contain scalars/strings or small tensors
        payload.update(extra)

    torch.save(payload, out_dir / fname)
    print(f"[save] wrote: {(out_dir / fname).as_posix()}")
    for k, v in payload.items():
        if torch.is_tensor(v):
            print(f"  - {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  - {k}: {type(v).__name__}")



if __name__ == "__main__":
    main(build_argparser().parse_args())
