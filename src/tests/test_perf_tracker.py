"""Host-side tests for the performance/cross tracker (perf_tracker/update_table.py).

No Isaac / torch / network needed -- pure run classification + leaderboard
selection + persistence. These lock in the rules that decide which runs are
"actionable" (kept on the lean leaderboard) versus archived-only noise.

Run directly (python tests/test_perf_tracker.py) or via pytest.
"""
import os
import sys
import tempfile
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERF = os.path.join(_REPO, "perf_tracker")
if _PERF not in sys.path:
    sys.path.insert(0, _PERF)

import update_table as ut  # noqa: E402


def _row(**kw):
    """A minimal run row with sensible 'real attempt' defaults; override per test."""
    r = {
        "run_id": kw.pop("run_id", "run_sim_x"),
        "outcome": None,
        "exit_reason": None,
        "terrain_id": None,
        "fall_diag_steps": 200,
        "max_x_m": 1.0,
        "timestamp": "t",
    }
    r.update(kw)
    return r


# --------------------------------------------------------------------------
# classify_run
# --------------------------------------------------------------------------

def test_classify_bench():
    r = _row(terrain_id="ramp_10deg", outcome="bench_terrain_complete")
    assert ut.classify_run(r) == ut.CATEGORY_BENCH, "a terrain_bench run must be categorised bench, not real"


def test_classify_self_test():
    r = _row(outcome="self_test_walk_complete", fall_diag_steps=199)
    assert ut.classify_run(r) == ut.CATEGORY_SELF_TEST


def test_classify_incomplete_outcomes():
    for o in ("unknown", "not_recorded", "", None, "docker_failed",
              "loop_ended_without_evaluation"):
        assert ut.classify_run(_row(outcome=o)) == ut.CATEGORY_INCOMPLETE, \
            f"outcome={o!r} has no usable verdict and must be archived-only"


def test_classify_incomplete_loop_ended_without_evaluation_even_with_many_samples():
    # A run whose render loop silently died (isaac_env.py's teardown fallback exit_reason,
    # see CLAUDE.md incident: run_sim_20260711_211922_087) can still have logged plenty of
    # fall_diag samples before it died -- MIN_REAL_FALL_DIAG_STEPS alone must not promote
    # it to CATEGORY_REAL just because the trajectory was long.
    r = _row(outcome="loop_ended_without_evaluation", fall_diag_steps=2000)
    assert ut.classify_run(r) == ut.CATEGORY_INCOMPLETE


def test_classify_incomplete_when_too_few_samples():
    r = _row(outcome="robot_fell", fall_diag_steps=ut.MIN_REAL_FALL_DIAG_STEPS - 1)
    assert ut.classify_run(r) == ut.CATEGORY_INCOMPLETE, \
        "a fall logged before MIN_REAL_FALL_DIAG_STEPS is too short to evaluate"


def test_classify_real_attempt():
    r = _row(outcome="robot_fell", fall_diag_steps=200)
    assert ut.classify_run(r) == ut.CATEGORY_REAL
    assert ut.is_actionable(r) is True


def test_classify_handles_string_fields():
    # Rows reloaded from CSV arrive as strings -- classification must still work.
    r = _row(outcome="robot_fell", fall_diag_steps="200", terrain_id="")
    assert ut.classify_run(r) == ut.CATEGORY_REAL, "string-valued steps/terrain must classify like numbers"


# --------------------------------------------------------------------------
# is_success
# --------------------------------------------------------------------------

def test_is_success_markers():
    assert ut.is_success(_row(outcome="patient_destination_and_robot_top_landing"))
    assert ut.is_success(_row(outcome="completed", exit_reason="patient_reached_top"))
    assert ut.is_success(_row(outcome="patient_destination_and_robot_stair_climb_visible"))
    assert not ut.is_success(_row(outcome="robot_fell")), "a fall is not a success"


# --------------------------------------------------------------------------
# select_table_rows -- the lean leaderboard
# --------------------------------------------------------------------------

def test_select_keeps_topN_plus_all_successes():
    rows = [
        _row(run_id=f"fall_{i}", outcome="robot_fell", max_x_m=3.0 - i * 0.1, fall_diag_steps=200)
        for i in range(30)
    ]
    # Two low-ranked successes that must survive the top-N cut anyway.
    rows.append(_row(run_id="win_a", outcome="patient_destination_and_robot_top_landing",
                     max_x_m=0.5, fall_diag_steps=200))
    rows.append(_row(run_id="win_b", outcome="completed", exit_reason="patient_reached_top",
                     max_x_m=0.2, fall_diag_steps=200))

    kept = ut.select_table_rows(rows, keep=5)
    ids = {r["run_id"] for r in kept}
    assert "win_a" in ids and "win_b" in ids, \
        "every success must be kept regardless of rank -- we never drop a real climb"
    fall_kept = [r for r in kept if r["outcome"] == "robot_fell"]
    assert len(fall_kept) == 5, f"keep=5 should retain exactly the 5 best falls, got {len(fall_kept)}"
    assert len(kept) == 7, f"5 top falls + 2 successes = 7, got {len(kept)}"


def test_select_excludes_noise_and_overrides_stale_category():
    rows = [
        _row(run_id="real1", outcome="robot_fell", max_x_m=2.0, fall_diag_steps=200),
        _row(run_id="bench1", terrain_id="flat", outcome="bench_terrain_complete",
             max_x_m=9.0, fall_diag_steps=200),
        # A noise row carrying a *stale* run_category='real' must still be excluded:
        # classification is recomputed from the live rules, not trusted blindly.
        _row(run_id="unk1", outcome="unknown", max_x_m=8.0, fall_diag_steps=500, run_category="real"),
    ]
    kept = ut.select_table_rows(rows, keep=25)
    ids = {r["run_id"] for r in kept}
    assert ids == {"real1"}, f"only the genuine attempt belongs on the leaderboard, got {ids}"


def test_select_sorts_best_first():
    rows = [
        _row(run_id="a", outcome="robot_fell", max_x_m=1.0, fall_diag_steps=200),
        _row(run_id="b", outcome="robot_fell", max_x_m=3.0, fall_diag_steps=200),
        _row(run_id="c", outcome="robot_fell", max_x_m=2.0, fall_diag_steps=200),
    ]
    kept = ut.select_table_rows(rows, keep=25)
    assert [r["run_id"] for r in kept] == ["b", "c", "a"], "leaderboard must be sorted by max_x_m descending"


# --------------------------------------------------------------------------
# Persistence: archive upsert, prior-run snapshot, migration
# --------------------------------------------------------------------------

def test_archive_upsert_is_idempotent():
    p = Path(tempfile.mkdtemp()) / "archive.jsonl"
    ut.upsert_archive(p, _row(run_id="dup", outcome="robot_fell", max_x_m=1.0))
    ut.upsert_archive(p, _row(run_id="dup", outcome="robot_fell", max_x_m=2.0))  # same id, newer data
    rows = ut._load_jsonl(p)
    assert len(rows) == 1, "re-ingesting the same run_id must replace, not duplicate"
    assert float(rows[0]["max_x_m"]) == 2.0, "the latest ingest should win"


def test_prior_run_prefers_last_run_then_falls_back():
    last = Path(tempfile.mkdtemp()) / "last_run.json"
    archive = [_row(run_id="old", timestamp="2026-06-01"), _row(run_id="older", timestamp="2026-05-01")]

    # No last_run.json yet -> newest archived run that isn't the current one.
    assert ut._prior_run(last, archive, current_run_id="cur")["run_id"] == "old"

    # last_run.json present -> it is the prior run.
    ut._write_last_run(last, _row(run_id="prev"))
    assert ut._prior_run(last, archive, current_run_id="cur")["run_id"] == "prev"

    # ...unless it points at the current run (a re-extract) -> fall back to archive.
    assert ut._prior_run(last, archive, current_run_id="prev")["run_id"] == "old"


def test_record_run_archives_bench_but_keeps_it_off_leaderboard():
    # Regression guard for the tracker<->terrain-bench contract: a bench run must
    # be archived (full history) yet excluded from the stair-climb leaderboard,
    # where a ramp's large max_x would otherwise outrank every real climb.
    d = Path(tempfile.mkdtemp())
    saved = (ut.DATA_DIR, ut.ARCHIVE_JSONL, ut.TABLE_CSV, ut.TABLE_JSONL)
    ut.DATA_DIR = d
    ut.ARCHIVE_JSONL = d / "archive.jsonl"
    ut.TABLE_CSV = d / "performance_table.csv"
    ut.TABLE_JSONL = d / "performance_table.jsonl"
    try:
        ut.record_run(_row(run_id="climb1", outcome="robot_fell", max_x_m=2.0, fall_diag_steps=200),
                      charts=False)
        bench = _row(run_id="ramp1", terrain_id="ramp_10deg", outcome="bench_terrain_complete",
                     max_x_m=9.0, fall_diag_steps=200)
        bench["run_category"] = ut.classify_run(bench)
        archive_rows, table_rows = ut.record_run(bench, charts=False)

        assert {r["run_id"] for r in archive_rows} == {"climb1", "ramp1"}, "both runs must be archived"
        assert {r["run_id"] for r in table_rows} == {"climb1"}, \
            "the ramp bench run must NOT appear on the stair-climb leaderboard"
    finally:
        ut.DATA_DIR, ut.ARCHIVE_JSONL, ut.TABLE_CSV, ut.TABLE_JSONL = saved


def test_migrate_seeds_archive_once():
    d = Path(tempfile.mkdtemp())
    legacy = d / "performance_table.jsonl"
    archive = d / "archive.jsonl"
    ut._write_jsonl(legacy, [_row(run_id="a"), _row(run_id="b")])

    ut._migrate_archive(archive, legacy)
    assert len(ut._load_jsonl(archive)) == 2, "migration must preserve every legacy run"

    # Second call is a no-op: the archive already exists and is the source of truth.
    ut._write_jsonl(legacy, [_row(run_id="c")])
    ut._migrate_archive(archive, legacy)
    assert len(ut._load_jsonl(archive)) == 2, "migration must not re-run once the archive exists"


# --------------------------------------------------------------------------
# Standalone runner (mirrors the other dual-mode tests in this suite)
# --------------------------------------------------------------------------

def _run_all() -> int:
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
