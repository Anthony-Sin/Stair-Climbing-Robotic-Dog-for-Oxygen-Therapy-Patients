#!/usr/bin/env python3
"""Aggregate a terrain-benchmark batch into perf_tracker + a benchmark summary.

Run host-side (the launcher calls it via WSL ``python3``), AFTER the warm Kit has
produced one run folder per terrain. For each terrain it reuses
``perf_tracker.update_table.extract_metrics`` and ingests the row via
``update_table.record_run`` (so it is archived in ``perf_tracker/data/archive.jsonl``
with a ``terrain_id`` column). Bench rows are categorised ``bench`` and stay OFF the
stair-climb leaderboard ``performance_table.csv`` -- a ramp's large ``max_x_m`` is not
comparable to a climb. The per-terrain results table is ``benchmark_summary.{md,csv,json}``
(written into the batch folder), alongside bench-only fields (PASS/FAIL, distance ratio,
traversal time, body-height stability).

Usage:
    python3 bench_metrics.py <batch_dir> [--git-branch <branch>]

``<batch_dir>`` (e.g. ``log/run_bench_<stamp>``) must contain ``manifest.json``:
    {"stamp": "...", "git_branch": "...",
     "terrains": [{"terrain_id": "ramp_10deg", "run_dir": "<abs path>"}, ...]}
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent                 # sim/isaac/terrain_bench
_REPO_ROOT = _THIS_DIR.parents[2]                           # repo root
_PERF_DIR = _REPO_ROOT / "perf_tracker"

sys.path.insert(0, str(_THIS_DIR))      # terrain_registry
sys.path.insert(0, str(_PERF_DIR))      # update_table, charts

from terrain_registry import get_battery, START_X_M  # noqa: E402
import update_table as ut  # noqa: E402  (extract_metrics / record_run / rebuild_table)


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _height_stats(run_dir: Path) -> Dict[str, Optional[float]]:
    """Re-read the fall_diag stream for the body-height (h) series -> min + std.

    extract_metrics only keeps final/max height, so the stability series is recomputed
    here from debug/isaac_env.jsonl (the same source extract_metrics parses)."""
    path = run_dir / "debug" / "isaac_env.jsonl"
    hs: List[float] = []
    if path.exists():
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                block = ev.get("event")
                if not (isinstance(block, dict) and block.get("action") == "fall_diag"):
                    continue
                h = _safe_float((ev.get("sim") or {}).get("h"))
                if h is not None:
                    hs.append(h)
    if not hs:
        return {"body_height_min_m": None, "body_height_std_m": None}
    mean = sum(hs) / len(hs)
    std = (sum((h - mean) ** 2 for h in hs) / len(hs)) ** 0.5
    return {"body_height_min_m": round(min(hs), 3), "body_height_std_m": round(std, 4)}


def _target_end_x(spec, row: Dict[str, Any]) -> Optional[float]:
    """The world X the robot had to reach to PASS this terrain.

    Ramps/flat carry an explicit target in the spec. Stairs use the geometry the sim
    actually logged (step_count * step_depth from stair_preset_configured), so the
    target tracks the real spawned staircase rather than a duplicated constant."""
    if spec.kind == "stairs":
        n = row.get("step_count")
        depth = row.get("step_depth_m")
        if n and depth:
            return round(float(spec.start_x_m) + float(n) * float(depth), 3)
        return None
    return float(spec.target_end_x_m) or None


def aggregate(batch_dir: Path, git_branch: Optional[str]) -> Dict[str, Any]:
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {batch_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    branch = git_branch or manifest.get("git_branch")
    by_id = {t.terrain_id: t for t in get_battery()}

    summaries: List[Dict[str, Any]] = []
    for entry in manifest.get("terrains", []):
        terrain_id = entry.get("terrain_id")
        run_dir = Path(entry.get("run_dir", ""))
        spec = by_id.get(terrain_id)
        if not run_dir.is_dir() or spec is None:
            print(f"[bench] SKIP {terrain_id}: missing run_dir or unknown terrain", flush=True)
            continue

        row = ut.extract_metrics(run_dir, git_branch=branch)
        row["terrain_id"] = terrain_id
        # terrain_id is set AFTER extract_metrics, so re-classify the row: a bench
        # run is archived (full history) but kept OFF the stair-climb leaderboard,
        # where a ramp's large max_x would otherwise outrank every real climb. The
        # per-terrain results live in benchmark_summary.{md,csv,json} below.
        row["run_category"] = ut.classify_run(row)
        ut.record_run(row, charts=False)  # archive + refresh table; charts drawn once at the end

        max_x = _safe_float(row.get("max_x_m"))
        target = _target_end_x(spec, row)
        traversed = None if max_x is None else round(max_x - START_X_M, 3)
        span = None if target is None else (target - START_X_M)
        distance_ratio = None
        if traversed is not None and span and span > 0:
            distance_ratio = round(max(0.0, traversed) / span, 3)
        success = bool(max_x is not None and target is not None and max_x >= target)

        s = {
            "terrain_id": terrain_id,
            "kind": spec.kind,
            "run_id": row.get("run_id"),
            "outcome": row.get("outcome"),
            "exit_reason": row.get("exit_reason"),
            "bench_success": success,
            "max_x_m": max_x,
            "target_end_x_m": target,
            "distance_ratio": distance_ratio,
            "time_to_traverse_sec": _safe_float(row.get("motion_elapsed_sec")),
            "fall_type": row.get("fall_type"),
            "max_abs_pitch_deg": _safe_float(row.get("max_abs_pitch_deg")),
            "max_abs_roll_deg": _safe_float(row.get("max_abs_roll_deg")),
            "mean_action_norm": _safe_float(row.get("mean_action_norm")),
            "max_action_norm": _safe_float(row.get("max_action_norm")),
        }
        s.update(_height_stats(run_dir))
        summaries.append(s)
        print(
            f"[bench] {terrain_id:22s} success={str(success):5s} "
            f"max_x={max_x}m ratio={distance_ratio} fall={row.get('fall_type')}",
            flush=True,
        )

    passed = sum(1 for s in summaries if s["bench_success"])
    total = len(summaries)
    result = {
        "stamp": manifest.get("stamp"),
        "git_branch": branch,
        "battery_pass_rate": (round(passed / total, 3) if total else None),
        "passed": passed,
        "total": total,
        "terrains": summaries,
    }

    _write_summary(batch_dir, result)

    # Draw the perf_tracker charts once, from the refreshed leaderboard.
    ut.rebuild_table(charts=True)

    print(f"[bench] archive    -> {ut.ARCHIVE_JSONL}", flush=True)
    print(f"[bench] summary    -> {batch_dir / 'benchmark_summary.md'}", flush=True)
    return result


_SUMMARY_COLS = [
    "terrain_id", "kind", "bench_success", "outcome", "exit_reason",
    "max_x_m", "target_end_x_m", "distance_ratio", "time_to_traverse_sec",
    "max_abs_pitch_deg", "max_abs_roll_deg", "mean_action_norm", "max_action_norm",
    "body_height_min_m", "body_height_std_m", "fall_type", "run_id",
]


def _write_summary(batch_dir: Path, result: Dict[str, Any]) -> None:
    (batch_dir / "benchmark_summary.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    with open(batch_dir / "benchmark_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_COLS, extrasaction="ignore")
        writer.writeheader()
        for s in result["terrains"]:
            writer.writerow(s)

    lines: List[str] = []
    lines.append(f"# Terrain Benchmark Summary -- {result.get('stamp')}")
    lines.append("")
    pr = result.get("battery_pass_rate")
    lines.append(
        f"Branch `{result.get('git_branch')}` | "
        f"PASS {result.get('passed')}/{result.get('total')}"
        + (f" ({pr * 100:.0f}%)" if pr is not None else "")
    )
    lines.append("")
    lines.append("| terrain | kind | PASS | max_x (m) | target | ratio | time (s) | "
                 "max&#124;pitch&#124; | max&#124;roll&#124; | mean a_norm | min h (m) | fall |")
    lines.append("|---|---|:--:|--:|--:|--:|--:|--:|--:|--:|--:|---|")

    def _c(v: Any) -> str:
        return "-" if v is None else str(v)

    for s in result["terrains"]:
        cells = [
            s["terrain_id"], s["kind"],
            ("PASS" if s["bench_success"] else "FAIL"),
            _c(s["max_x_m"]), _c(s["target_end_x_m"]), _c(s["distance_ratio"]),
            _c(s["time_to_traverse_sec"]), _c(s["max_abs_pitch_deg"]), _c(s["max_abs_roll_deg"]),
            _c(s["mean_action_norm"]), _c(s["body_height_min_m"]), _c(s["fall_type"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("PASS = robot's max forward X reached the terrain target without the "
                 "episode ending in a fall. `ratio` = fraction of the obstacle traversed "
                 "(graded progress even when PASS is false). Drive is a constant forward "
                 "command through the frozen parkour policy (no vision/Docker).")
    (batch_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate a terrain-benchmark batch.")
    ap.add_argument("batch_dir", help="Path to log/run_bench_<stamp> (must hold manifest.json)")
    ap.add_argument("--git-branch", default=None)
    ns = ap.parse_args()

    batch_dir = Path(ns.batch_dir).resolve()
    if not batch_dir.is_dir():
        print(f"[bench] ERROR: batch dir not found: {batch_dir}", file=sys.stderr)
        sys.exit(1)
    aggregate(batch_dir, ns.git_branch)


if __name__ == "__main__":
    main()
