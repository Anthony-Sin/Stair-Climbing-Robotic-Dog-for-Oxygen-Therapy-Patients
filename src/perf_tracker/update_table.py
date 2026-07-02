#!/usr/bin/env python3
"""
update_table.py

Extract performance metrics from a single run_sim_* folder, archive the run,
and refresh a lean leaderboard. Three files live in perf_tracker/data/:

  archive.jsonl            -- full history: every run ever ingested (source of truth).
  performance_table.csv    -- lean leaderboard: top actionable runs + all successes,
  performance_table.jsonl     best max_x_m first. This is what charts + humans read.
  last_run.json            -- the previous run's row, so the next run can print a
                              current-vs-prior delta without re-running anything.

"Actionable" = a real follow+climb attempt worth comparing (see classify_run).
Open-loop self-tests, terrain-bench battery runs, and aborted boots are archived
but kept off the leaderboard so it does not fill with noise.

Usage (called by run_sim.ps1 via WSL):
    python3 perf_tracker/update_table.py <run_folder_path> [--git-branch <branch>]
    python3 perf_tracker/update_table.py --rebuild        # re-derive table from archive
    python3 perf_tracker/update_table.py --keep 40 ...    # widen the leaderboard

Data sources inside each run folder:
  logs/status.jsonl                            -- launcher timeline + docker command (all controller args)
  reports/stair_demo_report.json               -- exit_reason, motion time, final robot pose
  debug/isaac_env.jsonl                        -- fall_diag physics stream, stair preset, climb events
  debug/debug_trace/vision_main_trace.jsonl    -- per-frame controller timing, YOLO detections, latency

Structure note (Phase 2 refactor): this module is a facade. The row schema,
classification rules, extraction, persistence, and console summary now live in
sibling modules (columns / classification / extraction / persistence / summary);
they are re-exported below so every previously-importable name still resolves
from perf_tracker.update_table. The canonical data-file paths and the public
ingest API (record_run / rebuild_table / main) stay here because tests and other
tools mutate these module-level paths and expect the ingest functions to observe
the change.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = _SCRIPT_DIR / "data"

# Canonical data files (see module docstring). Other tools (e.g. the terrain
# bench) should ingest through record_run()/rebuild_table() rather than touching
# these directly.
ARCHIVE_JSONL = DATA_DIR / "archive.jsonl"
TABLE_CSV = DATA_DIR / "performance_table.csv"
TABLE_JSONL = DATA_DIR / "performance_table.jsonl"
LAST_RUN_JSON = DATA_DIR / "last_run.json"

sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Re-exports -- preserve the public surface after the Phase 2 split. Every name
# below was importable from perf_tracker.update_table before the split.
# ---------------------------------------------------------------------------
from columns import COLUMNS  # noqa: F401,E402

from classification import (  # noqa: F401,E402
    CATEGORY_REAL,
    CATEGORY_SELF_TEST,
    CATEGORY_BENCH,
    CATEGORY_INCOMPLETE,
    MIN_REAL_FALL_DIAG_STEPS,
    _INCOMPLETE_OUTCOMES,
    _SUCCESS_MARKERS,
    _safe_int,
    classify_run,
    is_actionable,
    is_success,
)

from extraction import (  # noqa: F401,E402
    _safe_float,
    _parse_arg,
    _flag_absent,
    _get_git_branch,
    _get_git_info,
    _extract_trace,
    extract_metrics,
)

from persistence import (  # noqa: F401,E402
    _sort_key,
    DEFAULT_TABLE_KEEP,
    _load_jsonl,
    _write_jsonl,
    _write_csv,
    upsert_archive,
    select_table_rows,
    _migrate_archive,
    _load_last_run,
    _write_last_run,
    _prior_run,
)

from summary import (  # noqa: F401,E402
    _fnum,
    _fmt,
    _delta,
    _truthy,
    _category_counts,
    format_summary,
)


def _generate_charts(table_jsonl: Path) -> None:
    try:
        from charts import generate_all
        generate_all(table_jsonl, DATA_DIR / "charts")
    except Exception as exc:
        print(f"[perf_table] Charts skipped: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Public ingest API -- the only entry points other tools should use
# ---------------------------------------------------------------------------

def rebuild_table(
    *, keep: int = DEFAULT_TABLE_KEEP, charts: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Re-derive the lean leaderboard (csv+jsonl, +charts) from the archive.

    Returns (archive_rows, table_rows). Does not ingest anything new.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_rows = _load_jsonl(ARCHIVE_JSONL)
    table_rows = select_table_rows(archive_rows, keep)
    _write_csv(TABLE_CSV, table_rows)
    _write_jsonl(TABLE_JSONL, table_rows)
    if charts:
        _generate_charts(TABLE_JSONL)
    return archive_rows, table_rows


def record_run(
    row: Dict[str, Any], *, keep: int = DEFAULT_TABLE_KEEP, charts: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Archive one run and refresh the lean leaderboard. The single public ingest
    point shared by the CLI and the terrain bench. Returns (archive_rows, table_rows).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_archive(ARCHIVE_JSONL, TABLE_JSONL)
    upsert_archive(ARCHIVE_JSONL, row)
    return rebuild_table(keep=keep, charts=charts)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract run metrics, archive the run, and refresh the lean leaderboard."
    )
    parser.add_argument("run_folder", nargs="?", default=None, help="Path to the run_sim_* folder")
    parser.add_argument("--git-branch", default=None, help="Git branch name (passed from launcher)")
    parser.add_argument(
        "--keep", type=int, default=DEFAULT_TABLE_KEEP,
        help=f"Actionable runs to keep on the leaderboard (default {DEFAULT_TABLE_KEEP}); "
             "successes are always kept. Archive retains every run.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-derive the leaderboard + charts from archive.jsonl without ingesting a new run.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_archive(ARCHIVE_JSONL, TABLE_JSONL)

    if args.rebuild:
        archive_rows, table_rows = rebuild_table(keep=args.keep, charts=True)
        counts = _category_counts(archive_rows)
        print(
            f"[perf_table] Rebuilt from {len(archive_rows)} archived runs "
            f"(real={counts.get(CATEGORY_REAL, 0)} self_test={counts.get(CATEGORY_SELF_TEST, 0)} "
            f"bench={counts.get(CATEGORY_BENCH, 0)} incomplete={counts.get(CATEGORY_INCOMPLETE, 0)})"
            f" -> leaderboard keeps {len(table_rows)}",
            flush=True,
        )
        return

    if not args.run_folder:
        parser.error("run_folder is required unless --rebuild is given")

    run_folder = Path(args.run_folder).resolve()
    if not run_folder.is_dir():
        print(f"[perf_table] ERROR: run folder not found: {run_folder}", file=sys.stderr)
        sys.exit(1)

    # Snapshot the prior run BEFORE we ingest the current one.
    prior = _prior_run(LAST_RUN_JSON, _load_jsonl(ARCHIVE_JSONL), current_run_id=run_folder.name)

    print(f"[perf_table] Extracting: {run_folder.name}", flush=True)
    row = extract_metrics(run_folder, git_branch=args.git_branch)

    archive_rows, table_rows = record_run(row, keep=args.keep)
    _write_last_run(LAST_RUN_JSON, row)

    print(format_summary(row, prior, table_rows), flush=True)

    counts = _category_counts(archive_rows)
    print(
        f"[perf_table] archive {len(archive_rows)} runs "
        f"(real={counts.get(CATEGORY_REAL, 0)} self_test={counts.get(CATEGORY_SELF_TEST, 0)} "
        f"bench={counts.get(CATEGORY_BENCH, 0)} incomplete={counts.get(CATEGORY_INCOMPLETE, 0)})"
        f"  |  leaderboard keeps {len(table_rows)}",
        flush=True,
    )
    print(f"[perf_table] table   -> {TABLE_CSV}", flush=True)
    print(f"[perf_table] archive -> {ARCHIVE_JSONL}", flush=True)


if __name__ == "__main__":
    main()
