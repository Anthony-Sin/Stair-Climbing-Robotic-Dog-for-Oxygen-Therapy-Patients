"""Archive / leaderboard persistence + selection helpers.

Split out of update_table.py (Phase 2 structural refactor). These functions are
pure with respect to the module-level data paths: every one takes the file path
(or rows) it operates on as an argument, so the canonical data-file globals stay
in update_table.py where the public ingest API (record_run/rebuild_table) can
own them as mutable state.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from columns import COLUMNS
from classification import classify_run, is_success, CATEGORY_REAL


def _sort_key(r: Dict) -> tuple:
    """Best runs first: max_x_m desc, then timestamp desc."""
    try:
        x = float(r.get("max_x_m") or 0)
    except (TypeError, ValueError):
        x = 0.0
    ts = str(r.get("timestamp") or "")
    return (-x, ts)


# Default number of actionable runs kept on the leaderboard. The archive keeps
# every run regardless; this only bounds the human-readable table + charts.
DEFAULT_TABLE_KEEP = 25


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {col: ("" if r.get(col) is None else str(r.get(col))) for col in COLUMNS}
            )


def upsert_archive(archive_path: Path, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Replace any prior row for this run_id and append the new one. Full history."""
    rows = [r for r in _load_jsonl(archive_path) if r.get("run_id") != row.get("run_id")]
    rows.append(row)
    _write_jsonl(archive_path, rows)
    return rows


def select_table_rows(archive_rows: List[Dict[str, Any]], keep: int) -> List[Dict[str, Any]]:
    """Derive the lean leaderboard from the full archive.

    Keeps the top `keep` actionable runs by max_x_m, plus EVERY clear success
    regardless of rank (we never want to drop a run that actually climbed).
    Categories are recomputed here so reclassification rules always win over any
    stale `run_category` stored on an old row.
    """
    actionable = [r for r in archive_rows if classify_run(r) == CATEGORY_REAL]
    actionable.sort(key=_sort_key)  # best first

    kept = list(actionable[: max(0, keep)])
    kept_ids = {r.get("run_id") for r in kept}
    for r in actionable:
        if r.get("run_id") not in kept_ids and is_success(r):
            kept.append(r)
            kept_ids.add(r.get("run_id"))

    kept.sort(key=_sort_key)
    for r in kept:
        r["run_category"] = classify_run(r)
    return kept


def _migrate_archive(archive_jsonl: Path, legacy_table_jsonl: Path) -> None:
    """One-time: seed archive.jsonl from the pre-existing (un-pruned) table jsonl."""
    if archive_jsonl.exists():
        return
    if legacy_table_jsonl.exists():
        rows = _load_jsonl(legacy_table_jsonl)
        _write_jsonl(archive_jsonl, rows)
        print(f"[perf_table] Migrated {len(rows)} existing runs -> archive.jsonl", flush=True)
    else:
        _write_jsonl(archive_jsonl, [])


def _load_last_run(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_last_run(path: Path, row: Dict[str, Any]) -> None:
    path.write_text(json.dumps(row, default=str, indent=2), encoding="utf-8")


def _prior_run(
    last_run_path: Path, archive_rows: List[Dict[str, Any]], current_run_id: str
) -> Optional[Dict[str, Any]]:
    """The run to compare the current one against.

    Normally last_run.json (the immediately-preceding invocation). Falls back to
    the most-recent archived run when last_run.json is missing or points at the
    current run (e.g. a re-extract of the same folder).
    """
    lr = _load_last_run(last_run_path)
    if lr and lr.get("run_id") and lr.get("run_id") != current_run_id:
        return lr
    candidates = [r for r in archive_rows if r.get("run_id") != current_run_id]
    candidates.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    return candidates[0] if candidates else None
