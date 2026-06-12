# src/tools/aggregate_results.py
"""Aggregate alternating-reconstruction runs into one tidy CSV.

This walks an output tree (e.g. ``outputs/sweeps_cpu_alt`` or
``outputs/sweeps_gpu_b2_alt``) and collects, for every run directory, the
initial / final-cycle / best-cycle metrics into a single comparison table.

It understands both result formats produced by the pipeline:
  - ``metrics_summary.json`` (preferred; written by the current
    run_alternating_recon.py, includes best-cycle selection);
  - ``metrics.json`` (a list of per-cycle rows; older runs). When only this is
    present, the best cycle is recomputed here using ``--best_metric``.

All five magnitude metrics are carried through when present:
``nmse_mag, nrmse_mag, psnr_mag, ssim_mag, hfen_mag``.

The run is keyed by (checkpoint, acc_factor, case_index, schedule). acc_factor,
case_index and schedule are taken from run metadata when available and otherwise
parsed from the conventional folder name ``AF{af}_case{idx}_{schedule}``.

Usage
-----
    python -m src.tools.aggregate_results --root outputs/sweeps_cpu_alt
    python -m src.tools.aggregate_results --root outputs/sweeps_gpu_b2_alt \
        --out_csv outputs/sweeps_gpu_b2_alt/aggregate_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Metrics carried through for initial / final / best.
_METRIC_KEYS = ["nmse_mag", "nrmse_mag", "psnr_mag", "ssim_mag", "hfen_mag"]

_FOLDER_RE = re.compile(r"AF0*?(?P<af>\d+)_case(?P<case>\d+)_(?P<schedule>.+)$")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _metric_mode(metric_name: str, requested: str = "auto") -> str:
    requested = str(requested).lower()
    if requested in {"min", "max"}:
        return requested
    metric = str(metric_name).lower()
    if "psnr" in metric or "ssim" in metric:
        return "max"
    return "min"


def _is_better(candidate: Any, current_best: Any, mode: str) -> bool:
    c = _safe_float(candidate)
    b = _safe_float(current_best)
    if c is None:
        return False
    if b is None:
        return True
    return c > b if mode == "max" else c < b


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_folder_name(name: str) -> Dict[str, Optional[str]]:
    m = _FOLDER_RE.match(name)
    if not m:
        return {"acc_factor": None, "case_index": None, "schedule": None}
    return {
        "acc_factor": m.group("af").zfill(2),
        "case_index": m.group("case"),
        "schedule": m.group("schedule"),
    }


def _reductions(row: Dict[str, Any], initial_nmse: Optional[float], initial_psnr: Optional[float],
                final_nmse: Optional[float], final_psnr: Optional[float],
                best_nmse: Optional[float], best_psnr: Optional[float]) -> None:
    if initial_nmse not in (None, 0) and final_nmse is not None:
        row["final_nmse_reduction_pct"] = 100.0 * (initial_nmse - final_nmse) / initial_nmse
    if initial_nmse not in (None, 0) and best_nmse is not None:
        row["best_nmse_reduction_pct"] = 100.0 * (initial_nmse - best_nmse) / initial_nmse
    if initial_psnr is not None and final_psnr is not None:
        row["final_psnr_gain_db"] = final_psnr - initial_psnr
    if initial_psnr is not None and best_psnr is not None:
        row["best_psnr_gain_db"] = best_psnr - initial_psnr


def _row_from_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    initial = obj.get("initial") or {}
    final_cycle = obj.get("final_cycle") or {}
    best = obj.get("best_cycle") or {}
    best_metrics = best.get("best_cycle_metrics") or {}
    meta = obj.get("meta") or {}

    row: Dict[str, Any] = {"metrics_source": "metrics_summary.json"}
    for k in _METRIC_KEYS:
        row[f"initial_{k}"] = _safe_float(initial.get(k))
        row[f"final_{k}"] = _safe_float(final_cycle.get(k))
        row[f"best_{k}"] = _safe_float(best_metrics.get(k))

    row["initial_cycle"] = initial.get("cycle")
    row["final_cycle"] = final_cycle.get("cycle")
    row["final_dc_residual"] = _safe_float(final_cycle.get("dc_residual"))
    row["best_cycle"] = best.get("best_cycle")
    row["best_metric"] = best.get("best_metric")
    row["best_metric_mode"] = best.get("best_metric_mode")
    row["best_metric_value"] = _safe_float(best.get("best_metric_value"))
    row["best_dc_residual"] = _safe_float(best_metrics.get("dc_residual"))

    _meta_into_row(row, meta)
    _reductions(
        row,
        row["initial_nmse_mag"], row["initial_psnr_mag"],
        row["final_nmse_mag"], row["final_psnr_mag"],
        row["best_nmse_mag"], row["best_psnr_mag"],
    )
    return row


def _row_from_metrics_list(obj: Any, best_metric: str, best_metric_mode: str) -> Dict[str, Any]:
    if isinstance(obj, dict):
        obj = obj.get("metrics") or obj.get("rows") or obj.get("history") or [obj]
    rows = [r for r in obj if isinstance(r, dict)] if isinstance(obj, list) else []
    if not rows:
        return {"metrics_source": "metrics.json", "error": "no metric rows"}

    initial = rows[0]
    cycle_rows = [r for r in rows if r.get("cycle", 0) not in (0, None)] or rows
    final = cycle_rows[-1]

    mode = _metric_mode(best_metric, best_metric_mode)
    best_row, best_val = None, None
    for r in cycle_rows:
        v = r.get(best_metric)
        if _is_better(v, best_val, mode):
            best_val, best_row = _safe_float(v), r
    if best_row is None:
        best_row = final

    row: Dict[str, Any] = {"metrics_source": "metrics.json"}
    for k in _METRIC_KEYS:
        row[f"initial_{k}"] = _safe_float(initial.get(k))
        row[f"final_{k}"] = _safe_float(final.get(k))
        row[f"best_{k}"] = _safe_float(best_row.get(k))

    row["initial_cycle"] = initial.get("cycle")
    row["final_cycle"] = final.get("cycle")
    row["final_dc_residual"] = _safe_float(final.get("dc_residual"))
    row["best_cycle"] = best_row.get("cycle")
    row["best_metric"] = best_metric
    row["best_metric_mode"] = mode
    row["best_metric_value"] = best_val
    row["best_dc_residual"] = _safe_float(best_row.get("dc_residual"))

    _reductions(
        row,
        row["initial_nmse_mag"], row["initial_psnr_mag"],
        row["final_nmse_mag"], row["final_psnr_mag"],
        row["best_nmse_mag"], row["best_psnr_mag"],
    )
    return row


def _meta_into_row(row: Dict[str, Any], meta: Dict[str, Any]) -> None:
    ckpt = meta.get("ckpt")
    row["checkpoint"] = ckpt
    row["checkpoint_name"] = Path(ckpt).name if ckpt else None
    if meta.get("acc_factor") is not None:
        row["acc_factor"] = str(meta.get("acc_factor")).zfill(2)
    if meta.get("index") is not None:
        row["case_index"] = meta.get("index")
    row["cycles"] = meta.get("cycles")
    row["cond_mode"] = meta.get("cond_mode")
    row["dc_mode"] = meta.get("dc_mode")
    row["scale_mode"] = meta.get("scale_mode_effective")


def extract_run(run_dir: Path, *, best_metric: str, best_metric_mode: str) -> Optional[Dict[str, Any]]:
    summary = run_dir / "metrics_summary.json"
    metrics = run_dir / "metrics.json"

    if summary.exists():
        try:
            row = _row_from_summary(_load_json(summary))
        except Exception as e:  # pragma: no cover
            row = {"metrics_source": "metrics_summary.json", "error": str(e)}
    elif metrics.exists():
        try:
            row = _row_from_metrics_list(_load_json(metrics), best_metric, best_metric_mode)
        except Exception as e:  # pragma: no cover
            row = {"metrics_source": "metrics.json", "error": str(e)}
    else:
        return None

    # Fill keys from the folder name when metadata did not provide them.
    parsed = _parse_folder_name(run_dir.name)
    for key in ("acc_factor", "case_index", "schedule"):
        if not row.get(key) and parsed.get(key):
            row[key] = parsed[key]
    row["run_name"] = run_dir.name
    row["run_dir"] = str(run_dir)
    return row


def find_runs(root: Path) -> List[Path]:
    seen = set()
    runs: List[Path] = []
    for marker in ("metrics_summary.json", "metrics.json"):
        for p in root.rglob(marker):
            d = p.parent
            if d not in seen:
                seen.add(d)
                runs.append(d)
    return sorted(runs, key=lambda p: str(p))


# Column order: identity first, then initial/final/best blocks, then deltas.
_PREFERRED_COLUMNS = [
    "run_name", "checkpoint_name", "acc_factor", "case_index", "schedule",
    "cycles", "cond_mode", "dc_mode", "scale_mode",
    "initial_nmse_mag", "initial_psnr_mag", "initial_ssim_mag", "initial_hfen_mag",
    "final_cycle", "final_nmse_mag", "final_psnr_mag", "final_ssim_mag", "final_hfen_mag",
    "final_dc_residual",
    "best_cycle", "best_metric", "best_metric_value",
    "best_nmse_mag", "best_psnr_mag", "best_ssim_mag", "best_hfen_mag", "best_dc_residual",
    "final_nmse_reduction_pct", "best_nmse_reduction_pct",
    "final_psnr_gain_db", "best_psnr_gain_db",
    "metrics_source", "checkpoint", "run_dir",
]


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = list(_PREFERRED_COLUMNS)
    for r in rows:
        for k in r.keys():
            if k not in columns:
                columns.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate alternating-recon runs into one tidy CSV.")
    p.add_argument("--root", required=True, help="Directory to walk recursively for run folders.")
    p.add_argument("--out_csv", default=None, help="Output CSV path. Default: <root>/aggregate_summary.csv")
    p.add_argument("--best_metric", default="nmse_mag",
                   help="Metric used to choose best cycle when only metrics.json exists.")
    p.add_argument("--best_metric_mode", default="auto", choices=["auto", "min", "max"])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"--root does not exist: {root}")

    out_csv = Path(args.out_csv) if args.out_csv else root / "aggregate_summary.csv"

    run_dirs = find_runs(root)
    if not run_dirs:
        print(f"[WARN] No runs (metrics_summary.json / metrics.json) found under {root}")
        return

    rows: List[Dict[str, Any]] = []
    for d in run_dirs:
        row = extract_run(d, best_metric=args.best_metric, best_metric_mode=args.best_metric_mode)
        if row is not None:
            rows.append(row)
            best = row.get("best_nmse_mag")
            best_str = f"{best:.4e}" if isinstance(best, float) else "NA"
            print(f"[OK] {row.get('run_name'):40s} best_nmse_mag={best_str} "
                  f"best_psnr_mag={row.get('best_psnr_mag')} src={row.get('metrics_source')}")

    write_csv(rows, out_csv)
    print(f"\n[DONE] Aggregated {len(rows)} runs -> {out_csv}")


if __name__ == "__main__":
    main()
