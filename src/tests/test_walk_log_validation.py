"""Host (no-Isaac) tests for the walk_log.csv validator.

Proves two things WITHOUT booting Isaac:
  1. The 9-check validator runs end-to-end on a realistic synthetic walk and the
     invariants a clean walk must satisfy (no pelvis jump, no double float, forward
     progress) come back PASS.
  2. The validator actually DETECTS faults -- inject a double-float tick and a
     pelvis teleport and the corresponding checks must FAIL. A validator that only
     ever passes is worthless (cf. project_climb_verdict_honesty).

Run: python tests/test_walk_log_validation.py  (or via pytest)
"""

import csv
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "analysis"))

import validate_walk_log as V  # noqa: E402

_HEADER = [
    "timestamp_ms", "body_part", "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z",
    "gait_phase", "right_foot_grounded", "left_foot_grounded", "active_stance_leg", "mode",
]


def _synth_walk(n=300, dt=1.0 / 120.0, speed=1.4):
    """A simple Z-up, +x-forward kinematic walk. One foot is always grounded."""
    rows = []
    cadence = 1.0  # gait cycles per second
    for i in range(n):
        t = i * dt
        ts = round(t * 1000.0, 1)
        phase = (cadence * t) % 1.0
        px = speed * t
        pz = 0.92 + 0.02 * math.sin(2.0 * math.pi * 2.0 * phase)  # ~2cm bob

        def leg(side_phase, lateral):
            # stance for first 60% of the leg's own phase, swing for the rest
            stance = side_phase < 0.6
            hip = (px, lateral, pz - 0.05)
            if stance:
                foot_z = 0.0
                knee_bend = 0.15  # slight
                ankle = (px + 0.02, lateral, 0.10)
                knee = (px + 0.01, lateral, 0.50)
                grounded = 1
            else:
                u = (side_phase - 0.6) / 0.4
                lift = 0.12 * math.sin(math.pi * u)
                foot_z = lift
                ankle = (px + 0.10 * u, lateral, 0.10 + lift)
                knee = (px + 0.08 * u, lateral, 0.52 + 0.5 * lift)  # flex up
                grounded = 0
            foot = (px + (0.10 if not stance else 0.0), lateral, foot_z)
            return hip, knee, ankle, foot, grounded

        lh, lk, la, lf, lg = leg(phase, 0.10)
        rh, rk, ra, rf, rg = leg((phase + 0.5) % 1.0, -0.10)
        # arms counter-swing: left shoulder forward when right leg (phase+0.5) swings fwd
        lsh = (px + 0.06 * math.sin(2 * math.pi * ((phase + 0.5) % 1.0)), 0.18, pz + 0.45)
        rsh = (px + 0.06 * math.sin(2 * math.pi * phase), -0.18, pz + 0.45)
        stance_leg = "left" if lg else "right"

        parts = {
            "pelvis": (px, 0.0, pz),
            "left_hip": lh, "left_knee": lk, "left_ankle": la, "left_foot": lf,
            "right_hip": rh, "right_knee": rk, "right_ankle": ra, "right_foot": rf,
            "left_shoulder": lsh, "right_shoulder": rsh,
            "left_elbow": (lsh[0], 0.20, pz + 0.20), "right_elbow": (rsh[0], -0.20, pz + 0.20),
            "left_hand": (lsh[0], 0.22, pz), "right_hand": (rsh[0], -0.22, pz),
            "spine_base": (px, 0.0, pz + 0.15), "spine_mid": (px, 0.0, pz + 0.30),
            "spine_top": (px, 0.0, pz + 0.42), "head": (px, 0.0, pz + 0.60),
        }
        for name, (x, y, z) in parts.items():
            rows.append([ts, name, round(x, 5), round(y, 5), round(z, 5), 0.0, 0.0, 0.0,
                         round(phase, 4), rg, lg, stance_leg, "walk"])
    return rows


def _write(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(_HEADER)
        w.writerows(rows)


def _by_name(results):
    return {name: (ok, detail) for name, ok, detail in results}


def test_validator_runs_and_clean_walk_passes_invariants(tmp_path):
    path = os.path.join(str(tmp_path), "walk_log.csv")
    _write(_synth_walk(), path)

    ticks = V._load(path)
    assert len(ticks) == 300  # one tick per timestamp, deduped

    results = V.validate(ticks)
    assert len(results) == 9
    # every check returns a clean tri-state -- never raises, never garbage
    for name, ok, detail in results:
        assert ok in (True, False, None), name
        assert isinstance(detail, str) and detail

    r = _by_name(results)
    # invariants a clean kinematic walk MUST satisfy
    assert r["2. no_pelvis_jump"][0] is True, r["2. no_pelvis_jump"][1]
    assert r["3. no_double_float"][0] is True, r["3. no_double_float"][1]
    assert r["6. forward_progress"][0] is True, r["6. forward_progress"][1]


def test_validator_detects_double_float_and_pelvis_jump(tmp_path):
    rows = _synth_walk()
    # Corrupt: force one tick to have BOTH feet airborne, and teleport the pelvis.
    bad_ts = rows[150 * 19][0]  # ts of an interior tick (19 parts/tick)
    for row in rows:
        if row[0] == bad_ts:
            row[9] = 0   # right_foot_grounded
            row[10] = 0  # left_foot_grounded
            if row[1] == "pelvis":
                row[4] = float(row[4]) + 0.5  # 50cm vertical teleport

    path = os.path.join(str(tmp_path), "walk_log_bad.csv")
    _write(rows, path)
    r = _by_name(V.validate(V._load(path)))

    assert r["3. no_double_float"][0] is False, r["3. no_double_float"][1]
    assert r["2. no_pelvis_jump"][0] is False, r["2. no_pelvis_jump"][1]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        class _TP:
            def __init__(self, p):
                self._p = p

            def __fspath__(self):
                return self._p

            def __str__(self):
                return self._p

        test_validator_runs_and_clean_walk_passes_invariants(_TP(d))
        test_validator_detects_double_float_and_pelvis_jump(_TP(d))
    print("walk_log validator self-tests passed")
