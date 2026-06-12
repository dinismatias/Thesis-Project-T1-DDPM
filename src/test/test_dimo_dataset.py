# tests/test_dimo_dataset.py

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.recon.dimo_dataset import DimoKspaceDataset


def main():
    # TODO: change this to the real path on your machine
    # Example for MultiCoil / AccFactor04:
    dataset_root = Path(r"C:\Users\Admin\Tese\T1_DDPM_Project\ChallengeData\SingleCoil\Mapping\TrainingSet")
    acc = "04"

    acc_dir = dataset_root / f"AccFactor{acc}"
    # Collect all case dirs: P001, P002, ...
    case_dirs = sorted(
        [p for p in acc_dir.iterdir() if p.is_dir()]
    )

    print(f"Found {len(case_dirs)} case directories under {acc_dir}")
    for p in case_dirs[:5]:
        print("  ", p)

    # Create dataset
    ds = DimoKspaceDataset(
        case_dirs=case_dirs,
        acc_factor=acc,      # "04", "08", "10" or "full"
        multi_coil=False,     # False if you want SingleCoil
        use_full_as_target=True,
    )

    print(f"Dataset length (num (case, slice, TI) combinations): {len(ds)}")

    # Try to fetch one sample
    sample = ds[0]
    print("\nFirst sample keys:", sample.keys())

    print("  kspace_und shape :", sample["kspace_und"].shape)
    print("  mask shape       :", sample["mask"].shape)
    print("  img_target shape :", sample["img_target"].shape)

    if "TI_ms" in sample:
        print("  TI_ms            :", float(sample["TI_ms"]))
    print("  case_idx         :", int(sample["case_idx"]))
    print("  slice_idx        :", int(sample["slice_idx"]))
    print("  weight_idx       :", int(sample["weight_idx"]))

    # Optional: test with DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print("\nBatch shapes:")
    print("  kspace_und :", batch["kspace_und"].shape)   # [B, 2*C, H, W]
    print("  mask       :", batch["mask"].shape)         # [B, 1, H, W]
    print("  img_target :", batch["img_target"].shape)   # [B, 2, H, W]

    if "TI_ms" in batch:
        print("  TI_ms     :", batch["TI_ms"].shape)      # [B]


if __name__ == "__main__":
    main()
