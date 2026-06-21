"""Ingest a finished real run through the sim's public perf_tracker API.

A real run folder written by ``real.logging.real_telemetry`` is byte-compatible with
the sim's run-dir schema, so this is a thin adapter: ``extract_metrics(run_dir)`` ->
the row, then ``record_run(row)`` archives it + refreshes the leaderboard. No
sim-specific parsing and no duplicated metric logic -- the same code path the sim uses.

Run:  python -m real.verification.extract_real_run <run_dir> [--no-record]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


def ingest(run_dir: str, *, record: bool = True) -> Dict[str, Any]:
    """Extract the row for a real run folder; archive it unless record=False."""
    from perf_tracker.update_table import extract_metrics, record_run, classify_run

    row = extract_metrics(Path(run_dir))
    row["run_category"] = classify_run(row)
    if record:
        record_run(row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a real run into the perf_tracker leaderboard.")
    ap.add_argument("run_dir", help="Path to the real run folder (run_real_*)")
    ap.add_argument("--no-record", action="store_true", help="extract + print only; do not archive")
    args = ap.parse_args()

    row = ingest(args.run_dir, record=not args.no_record)
    print(f"run_id={row.get('run_id')} category={row.get('run_category')} "
          f"fall_diag_steps={row.get('fall_diag_steps')} final_x_m={row.get('final_x_m')} "
          f"worst_pitch={row.get('worst_pitch_deg')}")
    sys.exit(0)


if __name__ == "__main__":
    main()
