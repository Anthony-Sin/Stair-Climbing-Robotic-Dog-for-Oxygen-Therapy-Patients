"""Console summary -- the at-a-glance verdict for one run.

Split out of update_table.py (Phase 2 structural refactor). Pure formatting:
turns a run row (+ its prior run and the current leaderboard) into the boxed
text block the CLI prints, plus the small numeric formatters and the category
tally used by the CLI footer.
"""

from typing import Any, Dict, List, Optional

from classification import classify_run, is_actionable


def _fnum(v: Any) -> Optional[float]:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _fmt(v: Any, unit: str = "", nd: int = 2) -> str:
    n = _fnum(v)
    return f"{n:.{nd}f}{unit}" if n is not None else "--"


def _delta(cur: Any, prev: Any, unit: str = "", nd: int = 2) -> str:
    c, p = _fnum(cur), _fnum(prev)
    if c is None or p is None:
        return ""
    return f"  ({c - p:+.{nd}f}{unit} vs prior)"


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _category_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        cat = classify_run(r)
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def format_summary(
    row: Dict[str, Any], prior: Optional[Dict[str, Any]], table_rows: List[Dict[str, Any]]
) -> str:
    bar = "=" * 64
    cat = row.get("run_category") or classify_run(row)
    L = [
        bar,
        f" RUN  {row.get('run_id')}   [{cat}]",
        "-" * 64,
        f"  outcome        {row.get('outcome')}",
        f"  max forward    {_fmt(row.get('max_x_m'), ' m')}"
        f"{_delta(row.get('max_x_m'), prior.get('max_x_m') if prior else None, ' m')}",
        f"  reached stairs {'yes' if _truthy(row.get('stair_climb_reached')) else 'no'}",
        f"  worst pitch    {_fmt(row.get('max_abs_pitch_deg'), ' deg', 1)}",
        f"  worst roll     {_fmt(row.get('max_abs_roll_deg'), ' deg', 1)}",
        f"  fall_diag      {row.get('fall_diag_steps')} samples",
        f"  git            {row.get('git_commit_sha')}  {row.get('git_branch')}",
        "-" * 64,
    ]

    if is_actionable(row):
        ranked_ids = [r.get("run_id") for r in table_rows]
        if row.get("run_id") in ranked_ids:
            rank = ranked_ids.index(row.get("run_id")) + 1
            best = _fmt(table_rows[0].get("max_x_m"), " m") if table_rows else "--"
            L.append(f"  leaderboard    #{rank} of {len(table_rows)}   (best: {best})")
        else:
            L.append("  leaderboard    kept (actionable)")
    else:
        L.append(f"  leaderboard    archived only ({cat} -- not a tuning run)")

    if prior:
        L.append(
            f"  prior run      {prior.get('run_id')}  "
            f"max {_fmt(prior.get('max_x_m'), ' m')}  outcome {prior.get('outcome')}"
        )
    else:
        L.append("  prior run      (none recorded yet)")

    L.append(bar)
    return "\n".join(L)
