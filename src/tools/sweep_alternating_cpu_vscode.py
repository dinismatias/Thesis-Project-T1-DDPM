#sweep_alternating_cpu_vscode.py
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIG
# Edit only this section first.
# ============================================================

PROJECT = Path(r"C:\Users\Admin\Tese\T1_DDPM_Project")
CKPT = PROJECT / "checkpoints" / "dimo_cond_r4" / "epoch_0020.pt"

ACC_ROOT = PROJECT / "ChallengeData" / "SingleCoil" / "Mapping" / "TrainingSet"

# Start small. Use [0] first. Later change to [0, 1, 2].
CASE_INDICES = [0, 1, 2]

# Start with AF4. Later change to ["04", "08", "10"].
ACC_FACTORS = ["04", "08", "10"]

CYCLES = 3
DEVICE = "cpu"

# Useful schedules for comparing Stage-1 prior strength and Stage-2 DDIM+DC strength.
# name, prior_strength_schedule, prior_steps_schedule, stage2_strength_schedule, stage2_steps_schedule
SCHEDULES = [
    (
        "safe",
        "0.10,0.07,0.05",
        "25,20,15",
        "0.20,0.10,0.05",
        "50,50,50",
    ),
    (
        "stage2_heavier",
        "0.20,0.15,0.10",
        "25,20,15",
        "0.30,0.25,0.15",
        "50,50,50",
    ),
    (
        "aggressive_equal",
        "0.30,0.30,0.20",
        "25,20,15",
        "0.30,0.30,0.20",
        "50,50,50",
    ),
]

OUT_ROOT = PROJECT / "outputs" / "sweeps_cpu_alt"
LOG_ROOT = PROJECT / "logs" / "sweeps_cpu_alt"
SUMMARY_CSV = OUT_ROOT / "summary.csv"


# ============================================================
# Helpers
# ============================================================

def final_metrics_from_json(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}

    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if isinstance(data, list) and data:
        final = data[-1]
        if isinstance(final, dict):
            return final

    if isinstance(data, dict):
        return data

    return {}


def run_one(
    *,
    acc_factor: str,
    case_index: int,
    schedule_name: str,
    prior_strength_schedule: str,
    prior_steps_schedule: str,
    stage2_strength_schedule: str,
    stage2_steps_schedule: str,
) -> dict:
    out_dir = OUT_ROOT / f"AF{acc_factor}_case{case_index}_{schedule_name}"
    log_file = LOG_ROOT / f"AF{acc_factor}_case{case_index}_{schedule_name}.log"
    metrics_path = out_dir / "metrics.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    if metrics_path.exists():
        print(f"[SKIP] Already finished: {out_dir}")
        metrics = final_metrics_from_json(metrics_path)
        return {
            "acc_factor": acc_factor,
            "case_index": case_index,
            "schedule": schedule_name,
            "out_dir": str(out_dir),
            "status": "skipped_existing",
            **metrics,
        }

    cmd = [
        sys.executable,
        "-m",
        "src.test.run_alternating_recon",
        "--acc_root",
        str(ACC_ROOT),
        "--acc_factor",
        acc_factor,
        "--index",
        str(case_index),
        "--ckpt",
        str(CKPT),
        "--cond_mode",
        "zf_mask",
        "--cycles",
        str(CYCLES),
        "--prior_strength_schedule",
        prior_strength_schedule,
        "--prior_steps_schedule",
        prior_steps_schedule,
        "--stage2_strength_schedule",
        stage2_strength_schedule,
        "--stage2_steps_schedule",
        stage2_steps_schedule,
        "--dc_mode",
        "replace",
        "--scale_mode",
        "auto",
        "--log_residuals",
        "--save_png",
        "--device",
        DEVICE,
        "--out_dir",
        str(out_dir),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")

    # Keep CPU controlled while GPU training is running.
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"] = "2"

    print()
    print("=" * 90)
    print(f"Running AF={acc_factor} case={case_index} schedule={schedule_name}")
    print(f"prior_strength_schedule  = {prior_strength_schedule}")
    print(f"stage2_strength_schedule = {stage2_strength_schedule}")
    print(f"out_dir = {out_dir}")
    print("=" * 90)

    with log_file.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            f.write(line)

        return_code = process.wait()

    metrics = final_metrics_from_json(metrics_path)

    status = "ok" if return_code == 0 else f"failed_return_code_{return_code}"

    return {
        "acc_factor": acc_factor,
        "case_index": case_index,
        "schedule": schedule_name,
        "out_dir": str(out_dir),
        "log_file": str(log_file),
        "status": status,
        **metrics,
    }


def write_summary(rows: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_keys = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"[DONE] Summary written to:")
    print(SUMMARY_CSV)


def main() -> None:
    if not PROJECT.exists():
        raise FileNotFoundError(f"PROJECT does not exist: {PROJECT}")

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {CKPT}")

    if not ACC_ROOT.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {ACC_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []

    for acc_factor in ACC_FACTORS:
        for case_index in CASE_INDICES:
            for (
                schedule_name,
                prior_strength_schedule,
                prior_steps_schedule,
                stage2_strength_schedule,
                stage2_steps_schedule,
            ) in SCHEDULES:
                row = run_one(
                    acc_factor=acc_factor,
                    case_index=case_index,
                    schedule_name=schedule_name,
                    prior_strength_schedule=prior_strength_schedule,
                    prior_steps_schedule=prior_steps_schedule,
                    stage2_strength_schedule=stage2_strength_schedule,
                    stage2_steps_schedule=stage2_steps_schedule,
                )
                rows.append(row)
                write_summary(rows)

    print()
    print("[ALL FINISHED]")
    print(f"Outputs: {OUT_ROOT}")
    print(f"Logs:    {LOG_ROOT}")
    print(f"CSV:     {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
