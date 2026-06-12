#sweep_alternating_cpu_vscode_ultimate.py
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIG
# ============================================================

PROJECT = Path(r"C:\Users\Admin\Tese\T1_DDPM_Project")
CKPT = PROJECT / "checkpoints" / "dimo_cond_r4" / "epoch_0020.pt"

# Use the working canonical script, not the broken _v1 module.
MODULE = "src.test.run_alternating_recon"

ACC_ROOT = PROJECT / "ChallengeData" / "SingleCoil" / "Mapping" / "TrainingSet"

ACC_FACTORS = ["04", "08", "10"]
CASE_INDICES = [0, 1, 2]

CYCLES = 3
DEVICE = "cpu"
COND_MODE = "zf_mask"
DC_MODE = "replace"
SCALE_MODE = "auto"
SAVE_PNG = True
LOG_RESIDUALS = True

# Best-cycle reporting.
BEST_METRIC = "nmse_mag"
BEST_METRIC_MODE = "auto"
SAVE_BEST_CYCLE = True

# Runtime controls.
DRY_RUN = False
FORCE_RERUN = False

# CPU throttling while GPU training is active.
CPU_THREADS = "2"

# Keep these names because they match your current completed experiments.
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

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def metric_mode(metric_name: str, requested: str = "auto") -> str:
    requested = str(requested).lower()

    if requested in {"min", "max"}:
        return requested

    metric = str(metric_name).lower()

    if "psnr" in metric or "ssim" in metric:
        return "max"

    return "min"


def is_better(candidate: Any, current_best: Any, mode: str) -> bool:
    c = safe_float(candidate)
    b = safe_float(current_best)

    if c is None:
        return False

    if b is None:
        return True

    if mode == "max":
        return c > b

    return c < b


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics_from_run(out_dir: Path) -> Dict[str, Any]:
    """
    Reads either:
      - metrics_summary.json from the updated run_alternating_recon.py
      - or old metrics.json from previous completed runs

    Returns flat fields for CSV.
    """
    metrics_summary_path = out_dir / "metrics_summary.json"
    metrics_path = out_dir / "metrics.json"

    result: Dict[str, Any] = {}

    # Preferred new format.
    if metrics_summary_path.exists():
        try:
            obj = load_json(metrics_summary_path)

            initial = obj.get("initial") or {}
            final_cycle = obj.get("final_cycle") or {}
            best = obj.get("best_cycle") or {}
            best_metrics = best.get("best_cycle_metrics") or {}

            initial_nmse = safe_float(initial.get("nmse_mag"))
            initial_psnr = safe_float(initial.get("psnr_mag"))

            final_nmse = safe_float(final_cycle.get("nmse_mag"))
            final_psnr = safe_float(final_cycle.get("psnr_mag"))

            best_nmse = safe_float(best_metrics.get("nmse_mag"))
            best_psnr = safe_float(best_metrics.get("psnr_mag"))

            result.update(
                {
                    "initial_cycle": initial.get("cycle"),
                    "initial_nmse_mag": initial_nmse,
                    "initial_psnr_mag": initial_psnr,
                    "final_cycle": final_cycle.get("cycle"),
                    "final_nmse_mag": final_nmse,
                    "final_psnr_mag": final_psnr,
                    "final_nrmse_mag": final_cycle.get("nrmse_mag"),
                    "final_dc_residual": final_cycle.get("dc_residual"),
                    "best_metric": best.get("best_metric"),
                    "best_metric_mode": best.get("best_metric_mode"),
                    "best_metric_value": best.get("best_metric_value"),
                    "best_cycle": best.get("best_cycle"),
                    "best_nmse_mag": best_nmse,
                    "best_psnr_mag": best_psnr,
                    "best_nrmse_mag": best_metrics.get("nrmse_mag"),
                    "best_dc_residual": best_metrics.get("dc_residual"),
                    "metrics_source": "metrics_summary.json",
                }
            )

            if initial_nmse is not None and final_nmse is not None and initial_nmse != 0:
                result["final_nmse_reduction_pct"] = 100.0 * (initial_nmse - final_nmse) / initial_nmse

            if initial_nmse is not None and best_nmse is not None and initial_nmse != 0:
                result["best_nmse_reduction_pct"] = 100.0 * (initial_nmse - best_nmse) / initial_nmse

            if initial_psnr is not None and final_psnr is not None:
                result["final_psnr_gain_db"] = final_psnr - initial_psnr

            if initial_psnr is not None and best_psnr is not None:
                result["best_psnr_gain_db"] = best_psnr - initial_psnr

            return result

        except Exception as e:
            result["metrics_summary_error"] = str(e)

    # Backward-compatible old format.
    if metrics_path.exists():
        try:
            obj = load_json(metrics_path)

            if isinstance(obj, list):
                rows = [r for r in obj if isinstance(r, dict)]
            elif isinstance(obj, dict):
                maybe_rows = obj.get("metrics") or obj.get("rows") or obj.get("history")
                if isinstance(maybe_rows, list):
                    rows = [r for r in maybe_rows if isinstance(r, dict)]
                else:
                    rows = [obj]
            else:
                rows = []

            if not rows:
                result["metrics_error"] = "No metric rows found"
                return result

            initial = rows[0]
            cycle_rows = [r for r in rows if r.get("cycle", 0) != 0]
            final = cycle_rows[-1] if cycle_rows else rows[-1]

            mode = metric_mode(BEST_METRIC, BEST_METRIC_MODE)

            best_row = None
            best_value = None

            for r in cycle_rows:
                v = r.get(BEST_METRIC)
                if is_better(v, best_value, mode):
                    best_value = safe_float(v)
                    best_row = r

            if best_row is None:
                best_row = final

            initial_nmse = safe_float(initial.get("nmse_mag"))
            initial_psnr = safe_float(initial.get("psnr_mag"))

            final_nmse = safe_float(final.get("nmse_mag"))
            final_psnr = safe_float(final.get("psnr_mag"))

            best_nmse = safe_float(best_row.get("nmse_mag"))
            best_psnr = safe_float(best_row.get("psnr_mag"))

            result.update(
                {
                    "initial_cycle": initial.get("cycle"),
                    "initial_nmse_mag": initial_nmse,
                    "initial_psnr_mag": initial_psnr,
                    "final_cycle": final.get("cycle"),
                    "final_nmse_mag": final_nmse,
                    "final_psnr_mag": final_psnr,
                    "final_nrmse_mag": final.get("nrmse_mag"),
                    "final_dc_residual": final.get("dc_residual"),
                    "best_metric": BEST_METRIC,
                    "best_metric_mode": mode,
                    "best_metric_value": best_value,
                    "best_cycle": best_row.get("cycle"),
                    "best_nmse_mag": best_nmse,
                    "best_psnr_mag": best_psnr,
                    "best_nrmse_mag": best_row.get("nrmse_mag"),
                    "best_dc_residual": best_row.get("dc_residual"),
                    "metrics_source": "metrics.json",
                }
            )

            if initial_nmse is not None and final_nmse is not None and initial_nmse != 0:
                result["final_nmse_reduction_pct"] = 100.0 * (initial_nmse - final_nmse) / initial_nmse

            if initial_nmse is not None and best_nmse is not None and initial_nmse != 0:
                result["best_nmse_reduction_pct"] = 100.0 * (initial_nmse - best_nmse) / initial_nmse

            if initial_psnr is not None and final_psnr is not None:
                result["final_psnr_gain_db"] = final_psnr - initial_psnr

            if initial_psnr is not None and best_psnr is not None:
                result["best_psnr_gain_db"] = best_psnr - initial_psnr

            return result

        except Exception as e:
            result["metrics_error"] = str(e)
            return result

    result["metrics_error"] = "No metrics.json or metrics_summary.json found"
    return result


def write_summary(rows: List[Dict[str, Any]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    preferred = [
        "timestamp",
        "status",
        "returncode",
        "acc_factor",
        "case_index",
        "schedule",
        "prior_strength_schedule",
        "prior_steps_schedule",
        "stage2_strength_schedule",
        "stage2_steps_schedule",
        "out_dir",
        "log_file",
        "initial_nmse_mag",
        "initial_psnr_mag",
        "final_cycle",
        "final_nmse_mag",
        "final_psnr_mag",
        "final_nmse_reduction_pct",
        "final_psnr_gain_db",
        "best_cycle",
        "best_metric",
        "best_metric_value",
        "best_nmse_mag",
        "best_psnr_mag",
        "best_nmse_reduction_pct",
        "best_psnr_gain_db",
        "metrics_source",
        "error",
    ]

    all_keys: List[str] = []

    for key in preferred:
        if key not in all_keys:
            all_keys.append(key)

    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("[DONE] Summary written to:")
    print(SUMMARY_CSV)


def build_command(
    *,
    acc_factor: str,
    case_index: int,
    prior_strength_schedule: str,
    prior_steps_schedule: str,
    stage2_strength_schedule: str,
    stage2_steps_schedule: str,
    out_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        MODULE,
        "--acc_root",
        str(ACC_ROOT),
        "--acc_factor",
        acc_factor,
        "--index",
        str(case_index),
        "--ckpt",
        str(CKPT),
        "--cond_mode",
        COND_MODE,
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
        DC_MODE,
        "--scale_mode",
        SCALE_MODE,
        "--device",
        DEVICE,
        "--best_metric",
        BEST_METRIC,
        "--best_metric_mode",
        BEST_METRIC_MODE,
        "--out_dir",
        str(out_dir),
    ]

    if LOG_RESIDUALS:
        cmd.append("--log_residuals")

    if SAVE_PNG:
        cmd.append("--save_png")

    if SAVE_BEST_CYCLE:
        cmd.append("--save_best_cycle")
    else:
        cmd.append("--no_save_best_cycle")

    return cmd


def run_one(
    *,
    acc_factor: str,
    case_index: int,
    schedule_name: str,
    prior_strength_schedule: str,
    prior_steps_schedule: str,
    stage2_strength_schedule: str,
    stage2_steps_schedule: str,
) -> Dict[str, Any]:
    out_dir = OUT_ROOT / f"AF{acc_factor}_case{case_index}_{schedule_name}"
    log_file = LOG_ROOT / f"AF{acc_factor}_case{case_index}_{schedule_name}.log"
    metrics_path = out_dir / "metrics.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    base_row: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "acc_factor": acc_factor,
        "case_index": case_index,
        "schedule": schedule_name,
        "prior_strength_schedule": prior_strength_schedule,
        "prior_steps_schedule": prior_steps_schedule,
        "stage2_strength_schedule": stage2_strength_schedule,
        "stage2_steps_schedule": stage2_steps_schedule,
        "out_dir": str(out_dir),
        "log_file": str(log_file),
    }

    if metrics_path.exists() and not FORCE_RERUN:
        print(f"[SKIP] Already finished: {out_dir}")
        metrics = extract_metrics_from_run(out_dir)
        return {
            **base_row,
            "status": "skipped_existing",
            "returncode": 0,
            **metrics,
        }

    cmd = build_command(
        acc_factor=acc_factor,
        case_index=case_index,
        prior_strength_schedule=prior_strength_schedule,
        prior_steps_schedule=prior_steps_schedule,
        stage2_strength_schedule=stage2_strength_schedule,
        stage2_steps_schedule=stage2_steps_schedule,
        out_dir=out_dir,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = CPU_THREADS
    env["MKL_NUM_THREADS"] = CPU_THREADS
    env["OPENBLAS_NUM_THREADS"] = CPU_THREADS
    env["NUMEXPR_NUM_THREADS"] = CPU_THREADS

    print()
    print("=" * 100)
    print(f"Running AF={acc_factor} case={case_index} schedule={schedule_name}")
    print(f"prior_strength_schedule  = {prior_strength_schedule}")
    print(f"stage2_strength_schedule = {stage2_strength_schedule}")
    print(f"out_dir = {out_dir}")
    print("[CMD]")
    print(" ".join(cmd))
    print("=" * 100)

    if DRY_RUN:
        return {
            **base_row,
            "status": "dry_run",
            "returncode": 0,
        }

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

    metrics = extract_metrics_from_run(out_dir)
    status = "ok" if return_code == 0 else "failed"

    row = {
        **base_row,
        "status": status,
        "returncode": return_code,
        **metrics,
    }

    if return_code != 0:
        row["error"] = f"Command failed. Check log: {log_file}"

    nmse = row.get("final_nmse_mag", row.get("nmse_mag", "NA"))
    psnr = row.get("final_psnr_mag", row.get("psnr_mag", "NA"))
    best_cycle = row.get("best_cycle", "NA")
    best_nmse = row.get("best_nmse_mag", "NA")

    if return_code == 0:
        print(
            f"[OK] AF={acc_factor} case={case_index} {schedule_name} | "
            f"final NMSE={nmse} | final PSNR={psnr} | "
            f"best_cycle={best_cycle} | best_NMSE={best_nmse}"
        )
    else:
        print(f"[FAIL] AF={acc_factor} case={case_index} {schedule_name}. Check log: {log_file}")

    return row


def main() -> None:
    if not PROJECT.exists():
        raise FileNotFoundError(f"PROJECT does not exist: {PROJECT}")

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {CKPT}")

    if not ACC_ROOT.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {ACC_ROOT}")

    os.chdir(PROJECT)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    print("[INFO] Windows VS Code CPU sweep")
    print(f"[INFO] PROJECT: {PROJECT}")
    print(f"[INFO] CKPT: {CKPT}")
    print(f"[INFO] MODULE: {MODULE}")
    print(f"[INFO] ACC_ROOT: {ACC_ROOT}")
    print(f"[INFO] DEVICE: {DEVICE}")
    print(f"[INFO] ACC_FACTORS: {ACC_FACTORS}")
    print(f"[INFO] CASE_INDICES: {CASE_INDICES}")
    print(f"[INFO] SUMMARY_CSV: {SUMMARY_CSV}")
    print()

    rows: List[Dict[str, Any]] = []

    total = 0
    ok = 0
    failed = 0
    skipped = 0

    for acc_factor in ACC_FACTORS:
        for case_index in CASE_INDICES:
            for (
                schedule_name,
                prior_strength_schedule,
                prior_steps_schedule,
                stage2_strength_schedule,
                stage2_steps_schedule,
            ) in SCHEDULES:
                total += 1

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

                status = row.get("status")
                if status == "ok":
                    ok += 1
                elif status == "skipped_existing":
                    skipped += 1
                elif status == "failed":
                    failed += 1

                write_summary(rows)

    print()
    print("=" * 100)
    print("[ALL FINISHED]")
    print(f"total={total} ok={ok} skipped={skipped} failed={failed}")
    print(f"Outputs: {OUT_ROOT}")
    print(f"Logs:    {LOG_ROOT}")
    print(f"CSV:     {SUMMARY_CSV}")
    print("=" * 100)


if __name__ == "__main__":
    main()
