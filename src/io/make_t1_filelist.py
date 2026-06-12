# src/io/make_t1_filelist.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def collect_t1_paths(
    root: Path,
    coil_type: str = "MultiCoil",
    task_type: str = "Mapping",
    split: str = "TrainingSet",
    img_type: str = "Fullsample",

) -> List[Path]:
    """
    Collect all T1map.mat paths for a given coil type and split.

    Expected layout (CMRxRecon-style):

        root/
          MultiCoil/
            Mapping/
              TrainingSet/
                AccFactor04/
                  P001/T1map.mat
                  P002/T1map.mat
                AccFactor08/
                  ...
                AccFactor10/
                  ...
              FullSample/
                P001/T1map.mat
                ...

        (Similarly for SingleCoil/Mapping/...)

    Parameters
    ----------
    root : Path
        Root directory of the challenge data (e.g. /path/to/Challenge).
    coil_type : {"MultiCoil", "SingleCoil"}
        Which branch to scan.
    split : {"TrainingSet", "ValidationSet", "TestSet", "FullSample"}
        Which split under Mapping/ to scan.

    Returns
    -------
    paths : list of Path
        All existing T1map.mat files under that subtree.
    """
    base = root / coil_type / task_type / split / img_type

    if not base.exists():
        print(f"[WARN] Base directory does not exist: {base}")
        return []

    # For TrainingSet: we typically have AccFactorXX folders
    # For FullSample: we may have P001, P002 directly.
    paths: List[Path] = []

    # First, handle possible AccFactorXX folders
    for acc_dir in sorted(base.glob("AccFactor*")):
        if not acc_dir.is_dir():
            continue

        for subj_dir in sorted(acc_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            t1_file = subj_dir / "T1map.mat"
            if t1_file.exists():
                paths.append(t1_file)

    # Also handle T1map.mat directly under PXXX for splits like FullSample/
    for subj_dir in sorted(base.iterdir()):
        if not subj_dir.is_dir():
            continue
        t1_file = subj_dir / "T1map.mat"
        if t1_file.exists() and t1_file not in paths:
            paths.append(t1_file)

    return paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a train_t1_list.txt file for CMRxRecon T1 data."
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the CMRxRecon data (e.g. /path/to/Challenge).",
    )
    parser.add_argument(
        "--coil_type",
        type=str,
        default="SingleCoil",
        choices=["MultiCoil", "SingleCoil"],
        help="Which branch to scan (MultiCoil or SingleCoil).",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="Mapping",
        choices=["Mapping", "Cine"],
        help="Which task type to scan (Mapping or Cine).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="TrainingSet",
        choices=["TrainingSet", "ValidationSet"],
        help="Which split under Mapping/ to scan (e.g. TrainingSet, ValidationSet).",
    )
    parser.add_argument(
        "--img_type",
        type=str,
        default="FullSample",
        choices=["AccFactor04", "AccFactor08", "AccFactor10", "FullSample", "SegmentROI"],
        help="Which image type to scan (e.g. AccFactor04, FullSample).",
    )    
    parser.add_argument(
        "--output",
        type=str,
        default="configs/train_t1_list.txt",
        help="Output text file to write paths into.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help=(
            "If set, write paths relative to --root. "
            "Otherwise, write absolute paths."
        ),
    )

    return parser


def main(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    out_path = Path(args.output)

    print(f"[INFO] Scanning root: {root}")
    print(f"[INFO] Coil type: {args.coil_type} | Task type: {args.task_type}  | Split: {args.split} | Image type: {args.img_type} ")

    t1_paths = collect_t1_paths(root=root, coil_type=args.coil_type, task_type=args.task_type, split=args.split, img_type=args.img_type)
    print(f"[INFO] Found {len(t1_paths)} T1map.mat files.")

    if not t1_paths:
        print("[WARN] No T1map.mat files found. Check your --root / layout.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for p in t1_paths:
            if args.relative:
                rel = p.relative_to(root)
                f.write(str(rel).replace("\\", "/") + "\n")
            else:
                f.write(str(p) + "\n")

    print(f"[INFO] Wrote file list to: {out_path}")


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
