# src/tools/sweep_alternating.py
"""Parameterized alternating-reconstruction sweep (checkpoint- and OS-agnostic).

Unlike the older sweep_alternating_cpu_vscode*.py drivers (hard-coded Windows
paths and a single output folder), this one takes the checkpoint, dataset root,
and output root as arguments. That lets old and new checkpoints write to
*separate* output roots (e.g. outputs/sweeps_cpu_alt vs outputs/sweeps_gpu_b2_alt),
which avoids the skip-bug where a fresh checkpoint silently reuses another
checkpoint's finished folders.

It drives ``python -m src.test.run_alternating_recon`` (the canonical script with
best-cycle support) and reuses the standard run-folder naming
``AF{af}_case{idx}_{schedule}`` so aggregate_results.py and make_panels.py work
unchanged. A run is skipped when its metrics.json already exists (unless
--force_rerun).

Default plan (per the roadmap): AF04, cases 0/1/2, schedules
{safe, stage2_heavier, aggressive_equal}. Expand to AF08/AF10 only once AF04 is
shown to improve.

Examples
--------
New GPU checkpoint, AF4 only, on the VM:
    python -m src.tools.sweep_alternating \
        --ckpt checkpoints/dimo_cond_r4_gpu_b2/epoch_0040.pt \
        --acc_root "$HOME/T1_DDPM_Project/ChallengeData/SingleCoil/Mapping/TrainingSet" \
        --out_root outputs/sweeps_gpu_b2_alt \
        --acc_factors 04 --indices 0 1 2 --device cuda

Reproduce the old checkpoint sweep (CPU):
    python -m src.tools.sweep_alternating \
        --ckpt checkpoints/dimo_cond_r4/epoch_0020.pt \
        --acc_root ".../TrainingSet" \
        --out_root outputs/sweeps_cpu_alt \
        --acc_factors 04 08 10 --indices 0 1 2 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# name -> (prior_strength, prior_steps, stage2_strength, stage2_steps)
SCHEDULES: Dict[str, Dict[str, str]] = {
    "safe": {
        "prior_strength_schedule": "0.10,0.07,0.05",
        "prior_steps_schedule": "25,20,15",
        "stage2_strength_schedule": "0.20,0.10,0.05",
        "stage2_steps_schedule": "50,50,50",
    },
    "stage2_heavier": {
        "prior_strength_schedule": "0.20,0.15,0.10",
        "prior_steps_schedule": "25,20,15",
        "stage2_strength_schedule": "0.30,0.25,0.15",
        "stage2_steps_schedule": "50,50,50",
    },
    "aggressive_equal": {
        "prior_strength_schedule": "0.30,0.30,0.20",
        "prior_steps_schedule": "25,20,15",
        "stage2_strength_schedule": "0.30,0.30,0.20",
        "stage2_steps_schedule": "50,50,50",
    },
}

MODULE = "src.test.run_alternating_recon"


def _final_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Light status helper: last cycle's NMSE/PSNR for the per-run status CSV.

    The full comparison table is produced by aggregate_results.py; this is only a
    quick at-a-glance summary alongside the run logs.
    """
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = data if isinstance(data, list) else data.get("metrics", []) if isinstance(data, dict) else []
    if not rows:
        return {}
    last = rows[-1]
    if not isinstance(last, dict):
        return {}
    return {
        "final_cycle": last.get("cycle"),
        "final_nmse_mag": last.get("nmse_mag"),
        "final_psnr_mag": last.get("psnr_mag"),
        "final_ssim_mag": last.get("ssim_mag"),
        "final_hfen_mag": last.get("hfen_mag"),
    }


def build_command(args: argparse.Namespace, *, acc_factor: str, index: int,
                  sched: Dict[str, str], out_dir: Path) -> List[str]:
    cmd = [
        sys.executable, "-m", MODULE,
        "--acc_root", str(args.acc_root),
        "--acc_factor", acc_factor,
        "--index", str(index),
        "--ckpt", str(args.ckpt),
        "--cond_mode", args.cond_mode,
        "--cycles", str(args.cycles),
        "--prior_strength_schedule", sched["prior_strength_schedule"],
        "--prior_steps_schedule", sched["prior_steps_schedule"],
        "--stage2_strength_schedule", sched["stage2_strength_schedule"],
        "--stage2_steps_schedule", sched["stage2_steps_schedule"],
        "--dc_mode", args.dc_mode,
        "--scale_mode", args.scale_mode,
        "--best_metric", args.best_metric,
        "--best_metric_mode", args.best_metric_mode,
        "--device", args.device,
        "--out_dir", str(out_dir),
    ]
    if args.log_residuals:
        cmd.append("--log_residuals")
    if args.save_png:
        cmd.append("--save_png")
    return cmd


def run_one(args: argparse.Namespace, *, acc_factor: str, index: int, schedule_name: str,
            out_root: Path, log_root: Path) -> Dict[str, Any]:
    sched = SCHEDULES[schedule_name]
    out_dir = out_root / f"AF{acc_factor}_case{index}_{schedule_name}"
    log_file = log_root / f"AF{acc_factor}_case{index}_{schedule_name}.log"
    metrics_path = out_dir / "metrics.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    base: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "acc_factor": acc_factor,
        "case_index": index,
        "schedule": schedule_name,
        "out_dir": str(out_dir),
        "log_file": str(log_file),
    }

    if metrics_path.exists() and not args.force_rerun:
        print(f"[SKIP] Already finished: {out_dir}")
        return {**base, "status": "skipped_existing", "returncode": 0, **_final_metrics(metrics_path)}

    cmd = build_command(args, acc_factor=acc_factor, index=index, sched=sched, out_dir=out_dir)

    print("\n" + "=" * 96)
    print(f"Running AF={acc_factor} case={index} schedule={schedule_name}")
    print(f"prior_strength  = {sched['prior_strength_schedule']}")
    print(f"stage2_strength = {sched['stage2_strength_schedule']}")
    print(f"out_dir         = {out_dir}")
    print("[CMD] " + " ".join(cmd))
    print("=" * 96)

    if args.dry_run:
        return {**base, "status": "dry_run", "returncode": 0}

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.project) + os.pathsep + env.get("PYTHONPATH", "")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(args.cpu_threads)

    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(args.project), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        rc = proc.wait()

    status = "ok" if rc == 0 else "failed"
    row = {**base, "status": status, "returncode": rc, **_final_metrics(metrics_path)}
    if rc != 0:
        row["error"] = f"Command failed (rc={rc}); see log: {log_file}"
        print(f"[FAIL] AF={acc_factor} case={index} {schedule_name}; see {log_file}")
    else:
        print(f"[OK] AF={acc_factor} case={index} {schedule_name} | "
              f"final NMSE={row.get('final_nmse_mag')} PSNR={row.get('final_psnr_mag')}")
    return row


def write_status_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["timestamp", "status", "returncode", "acc_factor", "case_index", "schedule",
            "final_cycle", "final_nmse_mag", "final_psnr_mag", "final_ssim_mag", "final_hfen_mag",
            "out_dir", "log_file", "error"]
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parameterized alternating-recon sweep.")
    p.add_argument("--ckpt", required=True, help="Checkpoint .pt to evaluate.")
    p.add_argument("--acc_root", required=True, help="TrainingSet / AccFactor root for the dataset.")
    p.add_argument("--out_root", required=True,
                   help="Output root for this checkpoint (use a distinct root per checkpoint).")
    p.add_argument("--log_root", default=None, help="Log root. Default: <out_root>/logs")
    p.add_argument("--project", default=".", help="Project dir used as cwd / PYTHONPATH.")

    p.add_argument("--acc_factors", nargs="+", default=["04"], help="e.g. 04 08 10")
    p.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--schedules", nargs="+", default=list(SCHEDULES.keys()),
                   choices=list(SCHEDULES.keys()))

    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--cond_mode", default="zf_mask")
    p.add_argument("--dc_mode", default="replace")
    p.add_argument("--scale_mode", default="auto")
    p.add_argument("--best_metric", default="nmse_mag")
    p.add_argument("--best_metric_mode", default="auto", choices=["auto", "min", "max"])
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument("--cpu_threads", default="2", help="Thread cap for BLAS/OMP when on CPU.")

    p.add_argument("--no_save_png", action="store_false", dest="save_png")
    p.add_argument("--no_log_residuals", action="store_false", dest="log_residuals")
    p.set_defaults(save_png=True, log_residuals=True)
    p.add_argument("--force_rerun", action="store_true", help="Re-run even if metrics.json exists.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.acc_factors = [str(a).zfill(2) for a in args.acc_factors]

    project = Path(args.project).resolve()
    if not Path(args.ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not Path(args.acc_root).exists():
        raise FileNotFoundError(f"acc_root not found: {args.acc_root}")
    args.project = project

    out_root = Path(args.out_root)
    log_root = Path(args.log_root) if args.log_root else out_root / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    status_csv = out_root / "sweep_status.csv"

    print(f"[INFO] ckpt       : {args.ckpt}")
    print(f"[INFO] acc_root   : {args.acc_root}")
    print(f"[INFO] out_root   : {out_root}")
    print(f"[INFO] acc_factors: {args.acc_factors}  indices: {args.indices}")
    print(f"[INFO] schedules  : {args.schedules}  device: {args.device}")

    rows: List[Dict[str, Any]] = []
    ok = skipped = failed = 0
    for af in args.acc_factors:
        for idx in args.indices:
            for name in args.schedules:
                row = run_one(args, acc_factor=af, index=int(idx), schedule_name=name,
                              out_root=out_root, log_root=log_root)
                rows.append(row)
                ok += row["status"] == "ok"
                skipped += row["status"] == "skipped_existing"
                failed += row["status"] == "failed"
                write_status_csv(rows, status_csv)

    print("\n" + "=" * 96)
    print(f"[ALL FINISHED] total={len(rows)} ok={ok} skipped={skipped} failed={failed}")
    print(f"Status CSV : {status_csv}")
    print(f"Outputs    : {out_root}")
    print(f"Next       : python -m src.tools.aggregate_results --root {out_root}")
    print("=" * 96)


if __name__ == "__main__":
    main()
