"""Run classification -- "actionable" vs archive-only noise.

Split out of update_table.py (Phase 2 structural refactor). Holds the category
constants and the pure predicates (classify_run / is_actionable / is_success)
that decide which runs land on the lean leaderboard. Also hosts the shared
``_safe_int`` numeric coercion used by both classification and extraction.
"""

from typing import Any, Dict, Optional


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Run classification -- "actionable" vs archive-only noise
#
# The table accumulates every run ever launched, but most rows do not help
# anyone tune the climb controller: aborted boots, open-loop self-tests, and
# terrain-bench battery runs all land here too. We tag each run with a single
# category so the leaderboard can keep only the runs worth comparing while the
# archive keeps the full history.
# ---------------------------------------------------------------------------

CATEGORY_REAL = "real"            # genuine follow+climb attempt with a usable outcome
CATEGORY_SELF_TEST = "self_test"  # open-loop walk/heading self-test, no patient follow
CATEGORY_BENCH = "bench"          # terrain_bench battery run (separate harness)
CATEGORY_INCOMPLETE = "incomplete"  # crashed/aborted/killed before a usable outcome

# A run must log at least this many physics samples to be judged a real attempt.
# Below this the dog barely moved before the run died, so the row is noise.
MIN_REAL_FALL_DIAG_STEPS = 50

# Outcomes that mean "no usable physics verdict was recorded".
# "loop_ended_without_evaluation" is isaac_env.py's teardown-time fallback exit_reason
# (main(), the after-loop evaluation call) for a render loop that ended WITHOUT any of
# the guarded evaluation_exit break sites firing -- e.g. an unguarded exception or Kit's
# is_running() silently going False (see CLAUDE.md incident: run_sim_20260711_211922_087
# died at handoff_crest headless with zero evaluation_exit event). It carries whatever
# trajectory happened to be recorded, which can exceed MIN_REAL_FALL_DIAG_STEPS, so
# without this entry such a run would misclassify as CATEGORY_REAL on the leaderboard.
_INCOMPLETE_OUTCOMES = {
    "", "none", "unknown", "not_recorded", "docker_failed",
    "loop_ended_without_evaluation",
}

# Substrings that mark a run as a clear success (kept in the table regardless of rank).
_SUCCESS_MARKERS = (
    "completed",
    "reached_top",
    "top_landing",
    "patient_destination",
    "climb_visible",
)


def classify_run(row: Dict[str, Any]) -> str:
    """Return one of the CATEGORY_* constants for a run row.

    Robust to string-valued fields (rows reloaded from CSV are all strings).
    Only CATEGORY_REAL rows are "actionable" -- see is_actionable().
    """
    if str(row.get("terrain_id") or "").strip():
        return CATEGORY_BENCH

    outcome = str(row.get("outcome") or "").strip().lower()
    exit_reason = str(row.get("exit_reason") or "").strip().lower()
    if "self_test" in outcome or "self_test" in exit_reason:
        return CATEGORY_SELF_TEST

    if outcome in _INCOMPLETE_OUTCOMES:
        return CATEGORY_INCOMPLETE

    steps = _safe_int(row.get("fall_diag_steps"), 0) or 0
    if steps < MIN_REAL_FALL_DIAG_STEPS:
        return CATEGORY_INCOMPLETE

    return CATEGORY_REAL


def is_actionable(row: Dict[str, Any]) -> bool:
    """True if this run belongs on the leaderboard (a real, evaluable attempt)."""
    return classify_run(row) == CATEGORY_REAL


def is_success(row: Dict[str, Any]) -> bool:
    """True if the robot/patient clearly succeeded (climbed / reached the top)."""
    blob = (str(row.get("outcome") or "") + " " + str(row.get("exit_reason") or "")).lower()
    return any(marker in blob for marker in _SUCCESS_MARKERS)
