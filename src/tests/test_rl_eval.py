"""Host-safe tests for the blind-RL climb-eval backbone (fine_tuning/rl/eval_climb.py).

No Isaac / GPU / sim needed. These validate the measurement backbone for the THREE goals
(climb without falling, climb without colliding into risers, keep the patient clear) by
synthesizing ``debug/isaac_env.jsonl`` fall_diag rows and running the REAL
``analyze_climb.analyze_run`` (the single source of truth for the verdict) + eval_climb's
``score_run`` / ``select_best`` / ``enumerate_checkpoints``.

Run directly (python src/tests/test_rl_eval.py) or via pytest. MUST pass on Windows host.
"""
import json
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
# analyze_climb lives under sim/analysis and is imported by name (sibling-style) --
# mirror the path-shim eval_climb + the sweep tests use.
_ANALYSIS = os.path.join(_REPO, "sim", "analysis")
if _ANALYSIS not in sys.path:
    sys.path.insert(0, _ANALYSIS)

import analyze_climb  # noqa: E402
from fine_tuning.rl import eval_climb  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic fall_diag run-dir builder (matches the schema analyze_climb reads)
# ---------------------------------------------------------------------------
def _row(x, h, pitch, roll=1.0, yaw=2.0, tilt=6.0, gap=1.5, t=0.0, on_stairs=None):
    """One fall_diag `sim` payload dict (keys analyze_climb.load_fall_diag_rows expects)."""
    if on_stairs is None:
        on_stairs = x > 1.5
    return {
        "t": round(t, 3), "x": round(x, 4), "y": 0.05, "h": round(h, 4),
        "roll": roll, "pitch": pitch, "yaw": yaw, "tilt_deg": tilt,
        "gap_m": gap, "stairs_action_active": bool(on_stairs),
    }


def _write_run(base, name, rows):
    """Write a run dir with debug/isaac_env.jsonl of fall_diag rows. Returns its path."""
    rd = os.path.join(base, name)
    os.makedirs(os.path.join(rd, "debug"), exist_ok=True)
    with open(os.path.join(rd, "debug", "isaac_env.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for s in rows:
            fh.write(json.dumps({"event": {"action": "fall_diag"}, "sim": s}) + "\n")
    return rd


# --- trajectory generators (one per goal / failure mode) -------------------
def _clean_rows(n=40):
    """Advance x from -1 to ~3.2 (well past base+2 runs=2.61), steady h~0.30, small tilt,
    gap comfortably > 0.65 -> a CLEAN CLIMB."""
    rows = []
    for i in range(n):
        f = i / (n - 1)
        x = -1.0 + f * 4.2                       # -1.0 -> 3.2
        rows.append(_row(x, 0.30, pitch=(3.0 if i % 2 else -3.0), tilt=6.0, gap=0.9, t=f * 8.0))
    return rows


def _nose_down_rows(n=40):
    """On stairs but plowing nose-first: mean pitch <= -8 / a dive <= -22, low h -> COLLIDED."""
    rows = []
    for i in range(n):
        f = i / (n - 1)
        x = -1.0 + f * 3.0                       # reaches base (2.0) but wedges ~2.0
        on = x >= 1.9
        pitch = -12.0 if on else -2.0            # persistently nose-down on the steps
        if on and i == n - 3:
            pitch = -24.0                        # one deep nose-dive into a riser
        h = 0.19 if on else 0.30                 # low but not a full collapse
        rows.append(_row(x, h, pitch=pitch, tilt=14.0, gap=0.9, t=f * 8.0, on_stairs=on))
    return rows


def _tall_start_fall_rows(n=40):
    """A big body tilt (> 60 deg) -> FELL (flipped/toppled)."""
    rows = []
    for i in range(n):
        f = i / (n - 1)
        x = -1.0 + f * 3.5
        tilt = 6.0 + (80.0 if f > 0.7 else 0.0)  # topples past the 60 deg fall line late
        rows.append(_row(x, 0.28, pitch=-4.0, tilt=tilt, gap=0.9, t=f * 8.0))
    return rows


def _patient_too_close_rows(n=40):
    """Otherwise-clean climb but the patient gap drops < 0.65 while x>-3.5 -> patient risk."""
    rows = []
    for i in range(n):
        f = i / (n - 1)
        x = -1.0 + f * 4.2
        gap = 0.9 if f < 0.5 else 0.40           # crowds the patient on the second half
        rows.append(_row(x, 0.30, pitch=(3.0 if i % 2 else -3.0), tilt=6.0, gap=gap, t=f * 8.0))
    return rows


# ---------------------------------------------------------------------------
# goal-1/2/3 verdicts through the REAL analyze_run + score_run
# ---------------------------------------------------------------------------
def test_clean_climb_passes():
    with tempfile.TemporaryDirectory() as td:
        rd = _write_run(td, "run_sim_clean", _clean_rows())
        stats = analyze_climb.analyze_run(rd)["stats"]
        assert stats is not None
        assert stats["clean_climb"] is True, stats["verdict"]
        assert not stats["fell"] and not stats["collided"]
        s = eval_climb.score_run(rd)
        assert s["passed"] is True, s
        assert s["score"] > 1000.0  # clean-climb bonus dominates


def test_nose_down_collision_fails():
    with tempfile.TemporaryDirectory() as td:
        rd = _write_run(td, "run_sim_nose", _nose_down_rows())
        stats = analyze_climb.analyze_run(rd)["stats"]
        assert stats is not None
        assert stats["collided"] is True, stats["verdict"]
        assert stats["clean_climb"] is False
        s = eval_climb.score_run(rd)
        assert s["passed"] is False, s


def test_tall_start_fall_fails():
    with tempfile.TemporaryDirectory() as td:
        rd = _write_run(td, "run_sim_fall", _tall_start_fall_rows())
        stats = analyze_climb.analyze_run(rd)["stats"]
        assert stats is not None
        assert stats["fell"] is True, stats["verdict"]
        s = eval_climb.score_run(rd)
        assert s["passed"] is False, s


def test_patient_collision_risk_fails():
    with tempfile.TemporaryDirectory() as td:
        rd = _write_run(td, "run_sim_patient", _patient_too_close_rows())
        stats = analyze_climb.analyze_run(rd)["stats"]
        assert stats is not None
        assert stats["patient_collision_risk"] is True, stats
        s = eval_climb.score_run(rd)
        # Even if the climb itself is clean, crowding the patient must FAIL the gate.
        assert s["passed"] is False, s


# ---------------------------------------------------------------------------
# pure selection logic
# ---------------------------------------------------------------------------
def test_enumerate_checkpoints_orders_by_iter():
    with tempfile.TemporaryDirectory() as td:
        run = os.path.join(td, "logs", "rsl_rl", "o2stair", "run")
        os.makedirs(run)
        for name in ("model_100.pt", "model_1500.pt", "model_800.pt"):
            open(os.path.join(run, name), "w").close()
        got = eval_climb.enumerate_checkpoints(td, "o2stair", top_n=3)
        names = [p.name for p in got]
        assert names == ["model_1500.pt", "model_800.pt", "model_100.pt"], names
        # top_n slices the newest-by-iter.
        top1 = eval_climb.enumerate_checkpoints(td, "o2stair", top_n=1)
        assert [p.name for p in top1] == ["model_1500.pt"]
        # unknown experiment -> empty (no crash).
        assert eval_climb.enumerate_checkpoints(td, "nope", top_n=3) == []


def test_enumerate_checkpoints_across_runs():
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "logs", "rsl_rl", "o2stair")
        for run, models in (("2024_run_a", ("model_50.pt", "model_900.pt")),
                            ("2024_run_b", ("model_1200.pt",))):
            os.makedirs(os.path.join(base, run))
            for m in models:
                open(os.path.join(base, run, m), "w").close()
        got = [p.name for p in eval_climb.enumerate_checkpoints(td, "o2stair", top_n=2)]
        assert got == ["model_1200.pt", "model_900.pt"], got


def test_select_best_prefers_clean():
    # A clean higher climb vs a collided one -> the clean one wins.
    clean = eval_climb._score_from_stats(
        {"clean_climb": True, "fell": False, "collided": False,
         "patient_collision_risk": False, "steps_climbed": 4.0, "min_h_on": 0.30,
         "verdict": "CLEAN CLIMB"})
    clean["checkpoint"] = "model_1500.pt"
    collided = eval_climb._score_from_stats(
        {"clean_climb": False, "fell": False, "collided": True,
         "patient_collision_risk": False, "steps_climbed": 2.0, "min_h_on": 0.19,
         "verdict": "COLLIDED"})
    collided["checkpoint"] = "model_1200.pt"
    best = eval_climb.select_best([collided, clean])
    assert best is clean and best["passed"] is True

    # All-fail -> the least-bad (climbed farther before colliding) is returned, flagged False.
    fell_early = eval_climb._score_from_stats(
        {"clean_climb": False, "fell": True, "collided": False,
         "patient_collision_risk": False, "steps_climbed": 0.2, "min_h_on": 0.10,
         "verdict": "FELL"})
    fell_early["checkpoint"] = "model_A.pt"
    collided_far = eval_climb._score_from_stats(
        {"clean_climb": False, "fell": False, "collided": True,
         "patient_collision_risk": False, "steps_climbed": 3.0, "min_h_on": 0.20,
         "verdict": "COLLIDED"})
    collided_far["checkpoint"] = "model_B.pt"
    worst_best = eval_climb.select_best([fell_early, collided_far])
    assert worst_best is not None
    assert worst_best["passed"] is False
    assert worst_best["checkpoint"] == "model_B.pt", worst_best

    # empty -> None
    assert eval_climb.select_best([]) is None


def test_score_from_stats_none_is_worst():
    s = eval_climb._score_from_stats(None)
    assert s["passed"] is False
    assert s["score"] == float("-inf")


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for _fn in _tests:
        _fn()
        print("PASS", _fn.__name__)
    print(f"ALL {len(_tests)} TESTS PASSED")
