"""Regression tests for the 2026-07-11 stair/post-crest speed-guard fixes (incident 8.15,
CLAUDE.md ledger). Host-safe (no Isaac/hardware deps) -- exercises the pure functions added
to core/control/stair_policy.py for:

  F2. climb_gap_brake_scale       -- mid-climb patient-gap speed brake. None-gap contract
      corrected 2026-07-11 (second same-day follow-up): it now splits by person VISIBILITY
      -- not-detected -> no brake (the None gap near the crest just means the patient is
      out of view, incident 8.3 blind-carry; braking on it flipped the dog at x=6.19, run
      run_sim_20260711_153245_944), detected + None/invalid gap -> brake (8.8 unchanged).
  F3. lost_person_speed_taper_scale -- POST-CREST / top-landing-ONLY lost-person speed
      taper. Corrected 2026-07-11 (same-day follow-up): it originally shipped applied at
      every mid-climb speed path too and fought incident 8.3's designed blind-carry,
      permanently parking the robot at the stair base (run_sim_20260711_150906). See the
      "F3 scope contract" section below for the static call-site tests.
  F4. detect_landing_edge_dropoff -- post-crest top-landing forward drop-off probe.

F1 (post-crest hold enforcement) lives entirely in sim/isaac/isaac_env.py's
_step_go2_locomotion, which imports isaacsim/omniverse and cannot be imported on a plain
host (mirrors incident 8.4: "the real (hardware) import graph is exercised by nothing").
It is a 2-line clamp (`if _motion_hold_requested: vx = 0.0; hold = True`) applied right
before the ONLY reachable rl_policy.step() call for flat-ground PGTT walking, verified by
code reading + `python -m compileall`, not a runtime test here. The upstream state machine
it works around (HandoffController's stair-commit vx_floor / state=="walk" reacquire logic)
is already covered by test_pgtt_stair_handoff.py::test_post_climb_reacquire.
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))
sys.path.insert(0, REPO)

from core.control.stair_policy import (  # noqa: E402
    climb_gap_brake_scale,
    lost_person_speed_taper_scale,
    detect_landing_edge_dropoff,
    landing_edge_guard_suppress_crest_artifact,
    ClimbGapFilterState,
    filtered_climb_gap_m,
    LandingMarginState,
    _fully_on_top_landing,
    _crest_reached,
    stair_loss_floor_eligible,
)
from go2_locomotion.pgtt_stair_handoff import HandoffConfig  # noqa: E402


# --------------------------------------------------------------------------------------
# F2: climb_gap_brake_scale
# --------------------------------------------------------------------------------------

def test_gap_brake_full_speed_at_or_above_start():
    assert climb_gap_brake_scale(1.2, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 1.0
    assert climb_gap_brake_scale(2.0, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 1.0


def test_gap_brake_linear_taper_between_thresholds():
    # Midpoint of [0.85, 1.2] -> scale 0.5.
    mid = 0.85 + (1.2 - 0.85) / 2.0
    scale = climb_gap_brake_scale(mid, brake_start_m=1.2, brake_stop_m=0.85,
                                  person_detected=True)
    assert abs(scale - 0.5) < 1e-6, scale
    # Monotonic: closer gap -> lower (or equal) scale.
    s_far = climb_gap_brake_scale(1.1, brake_start_m=1.2, brake_stop_m=0.85,
                                  person_detected=True)
    s_near = climb_gap_brake_scale(0.9, brake_start_m=1.2, brake_stop_m=0.85,
                                   person_detected=True)
    assert s_far > s_near, (s_far, s_near)


def test_gap_brake_zero_at_or_below_stop():
    assert climb_gap_brake_scale(0.85, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0
    assert climb_gap_brake_scale(0.05, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0, \
        "the observed 0.05 m fused_gap_m from run_sim_20260711_140745_054 must fully brake"


def test_gap_brake_fails_toward_brake_on_invalid_gap_with_person_visible():
    """Incident 8.8 (unchanged for the DETECTED case): a person is VISIBLE but the gap is
    None / the 'no reading' sentinel (<=1e-3) -- a genuine sensor error while a real
    proximity risk exists must brake, not charge. Still the OPPOSITE default from the
    pre-existing collision-floor checks in this module, which deliberately SKIP (fail
    toward full speed) on that same sentinel."""
    assert climb_gap_brake_scale(None, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0
    assert climb_gap_brake_scale(0.0, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0
    assert climb_gap_brake_scale(-0.5, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=True) == 0.0


def test_gap_brake_no_brake_when_person_not_detected():
    """Second 8.15 scope correction (2026-07-11): with the person NOT detected the brake
    must stay at FULL scale regardless of the gap reading. Near the crest the smoothed gap
    goes None simply because the patient is out of view -- the NORMAL incident-8.3
    blind-carry situation, not a proximity risk -- and failing toward the brake on that
    None held the dog at commanded-zero on the incline for ~120 STAIR_LOSS_FLOOR frames
    until it toppled rear-high (run_sim_20260711_153245_944: climbed to x=6.19, flipped at
    roll 179 deg mid-crest). The hard collision floors on last_person_gap_m at each call
    site still backstop the person-was-close-then-dropped-out case."""
    # None gap + not detected == the run 153245 crest signature -> full scale, no brake.
    assert climb_gap_brake_scale(None, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=False) == 1.0
    # Sentinel / invalid readings with nobody visible -> likewise no brake.
    assert climb_gap_brake_scale(0.0, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=False) == 1.0
    # Even a numerically "close" stale gap does not brake when nobody is visible (absence
    # of a person is not a proximity risk; the hard _loss_block/_coll_block floors own that).
    assert climb_gap_brake_scale(0.9, brake_start_m=1.2, brake_stop_m=0.85,
                                 person_detected=False) == 1.0


# --------------------------------------------------------------------------------------
# F2 hardening (2026-07-11 review): filtered_climb_gap_m -- rolling-minimum noise filter
# --------------------------------------------------------------------------------------
# run_sim_20260711_195618_941 (a NEWLY-deployed fast/stable climber) failed the grade gate
# with a patient collision mid-climb. The trace below is the REAL sim.gap_m /
# sim.person_detected sequence read from that run's debug/isaac_env.jsonl fall_diag events
# (t = motion_elapsed_sim_sec). It shows the exact defect: a single noisy "far" reading
# (0.282 -> 0.885 in one control-loop tick) released climb_gap_brake_scale's memoryless
# brake to full scale for that frame while the true gap stayed under the 0.85 m
# brake_stop_m, producing full-speed forward pulses that tailgated the patient to 0.199 m.

_TRACE_RUN_20260711_195618_941 = [
    # (t_sec, gap_m, person_detected) -- verbatim from debug/isaac_env.jsonl fall_diag.
    (77.93, 0.805, True),
    (78.00, 0.673, True),
    (78.61, 0.581, True),
    (79.28, 0.432, True),
    (79.36, 0.491, True),
    (79.43, None, False),
    (79.88, 0.411, True),
    (80.03, 0.407, True),
    (80.41, 0.381, True),
    (80.48, 0.282, True),   # closing in
    (80.56, 0.282, True),
    (80.63, 0.885, True),   # <-- the 0.282 -> 0.885 one-frame jump
    (80.70, 0.629, True),
    (80.78, 0.620, True),
    (80.86, None, False),
    (82.36, 1.837, True),   # <-- oscillation begins: 1.837 -> 2.797 -> ... -> 3.047 -> 0.788
    (82.43, 2.797, True),
    (82.50, 2.734, True),
    (82.58, 1.862, True),
    (82.81, 3.047, True),
    (83.56, 0.886, True),
    (83.70, 0.788, True),
    (83.78, 0.199, True),   # true proximity the oscillation was hiding
    (83.93, 1.883, True),
]

_BRAKE_START_M = 1.2
_BRAKE_STOP_M = 0.85
_FILTER_WINDOW_SEC = 1.2


def _true_rolling_min(trace, idx, window_sec):
    """Independent (test-local, not production-code) rolling minimum over the trace's own
    (t, gap, detected) samples, for cross-checking filtered_climb_gap_m's output -- only
    counts samples where the person was detected, mirroring the None-gap contract."""
    t_now = trace[idx][0]
    vals = [g for (t, g, d) in trace[: idx + 1]
            if d and g is not None and (t_now - t) <= window_sec]
    return min(vals) if vals else None


def test_gap_brake_filter_never_releases_on_noisy_trace_replay():
    """Replays the ACTUAL noisy gap sequence from run_sim_20260711_195618_941 through
    filtered_climb_gap_m -> climb_gap_brake_scale and asserts the brake never meaningfully
    releases (stays <= 0.35, well below the 1.0 unbraked scale) at any sample where the true
    (rolling-minimum) gap was under 0.85 m -- in particular the 0.282 -> 0.885 jump at
    t=80.63 and every point inside the 1.837 -> 3.047 -> 0.788 oscillation."""
    state = ClimbGapFilterState()
    for idx, (t, gap, detected) in enumerate(_TRACE_RUN_20260711_195618_941):
        filtered = filtered_climb_gap_m(
            gap, person_detected=detected, state=state, now=t,
            window_sec=_FILTER_WINDOW_SEC,
        )
        scale = climb_gap_brake_scale(
            filtered, brake_start_m=_BRAKE_START_M, brake_stop_m=_BRAKE_STOP_M,
            person_detected=detected,
        )
        true_min = _true_rolling_min(_TRACE_RUN_20260711_195618_941, idx, _FILTER_WINDOW_SEC)
        if detected and true_min is not None and true_min < _BRAKE_STOP_M:
            assert scale <= 0.35, (
                f"t={t}: brake scale {scale} released too far while the true rolling-min "
                f"gap was {true_min} m (< brake_stop_m={_BRAKE_STOP_M}) -- filtered={filtered}"
            )
    # The specific jump frame from the failure trace: filtered value must reflect the recent
    # close readings (<=0.4ish), not the single noisy 0.885 sample itself.
    state2 = ClimbGapFilterState()
    filtered_at_jump = None
    for (t, gap, detected) in _TRACE_RUN_20260711_195618_941:
        filtered_at_jump = filtered_climb_gap_m(
            gap, person_detected=detected, state=state2, now=t, window_sec=_FILTER_WINDOW_SEC,
        )
        if t == 80.63:
            break
    assert filtered_at_jump is not None and filtered_at_jump <= 0.35, filtered_at_jump


def test_gap_brake_filter_releases_within_two_seconds_of_genuine_opening():
    """Once the TRUE gap genuinely opens (a sustained monotone rise, not a single noisy
    frame), the rolling-minimum filter must catch up and release the brake within roughly
    the filter window (~1.2 s) plus a small margin -- it must not permanently pin the brake
    on stale close history."""
    state = ClimbGapFilterState()
    t = 0.0
    # Seed a genuinely close approach.
    for gap in (0.5, 0.45, 0.4):
        filtered_climb_gap_m(gap, person_detected=True, state=state, now=t,
                             window_sec=_FILTER_WINDOW_SEC)
        t += 0.1
    close_scale = climb_gap_brake_scale(
        filtered_climb_gap_m(0.4, person_detected=True, state=state, now=t,
                             window_sec=_FILTER_WINDOW_SEC),
        brake_start_m=_BRAKE_START_M, brake_stop_m=_BRAKE_STOP_M, person_detected=True,
    )
    assert close_scale == 0.0, close_scale
    # Now the patient genuinely pulls away: sustained rise held well above brake_start_m.
    t_open_start = t
    scale_at_release = None
    while t < t_open_start + 2.0:
        t += 0.05
        filtered = filtered_climb_gap_m(1.5, person_detected=True, state=state, now=t,
                                        window_sec=_FILTER_WINDOW_SEC)
        scale_at_release = climb_gap_brake_scale(
            filtered, brake_start_m=_BRAKE_START_M, brake_stop_m=_BRAKE_STOP_M,
            person_detected=True,
        )
        if scale_at_release >= 1.0:
            break
    assert scale_at_release == 1.0, (
        f"brake did not release to full scale within 2s of a sustained gap opening to 1.5 m "
        f"(last scale={scale_at_release} at t={t - t_open_start:.2f}s into the opening)"
    )
    assert (t - t_open_start) <= 2.0 + 1e-6


def test_gap_brake_filter_resets_on_a_detection_gap_longer_than_the_window():
    """A fresh detection after a no-detection stretch LONGER than window_sec must not inherit
    a stale pre-loss minimum -- the window must have aged out naturally by then."""
    state = ClimbGapFilterState()
    # Close reading right before the person is lost.
    filtered_climb_gap_m(0.3, person_detected=True, state=state, now=0.0,
                         window_sec=_FILTER_WINDOW_SEC)
    # Long loss -- well beyond window_sec.
    filtered_climb_gap_m(None, person_detected=False, state=state, now=5.0,
                         window_sec=_FILTER_WINDOW_SEC)
    # Fresh detection, far gap.
    filtered = filtered_climb_gap_m(2.0, person_detected=True, state=state, now=5.05,
                                    window_sec=_FILTER_WINDOW_SEC)
    assert filtered == 2.0, (
        f"expected the fresh 2.0 m reading alone (stale 0.3 m sample must have aged out "
        f"across the {5.05 - 0.0:.1f}s gap), got {filtered}"
    )
    scale = climb_gap_brake_scale(filtered, brake_start_m=_BRAKE_START_M,
                                  brake_stop_m=_BRAKE_STOP_M, person_detected=True)
    assert scale == 1.0, scale


def test_gap_brake_filter_no_brake_when_not_detected_regardless_of_window_contents():
    """Preserves the 8.15 second-correction None-split EXACTLY: not-detected -> full scale,
    even with a close reading still inside the trailing window."""
    state = ClimbGapFilterState()
    filtered_climb_gap_m(0.3, person_detected=True, state=state, now=0.0,
                         window_sec=_FILTER_WINDOW_SEC)
    filtered = filtered_climb_gap_m(None, person_detected=False, state=state, now=0.2,
                                    window_sec=_FILTER_WINDOW_SEC)
    scale = climb_gap_brake_scale(filtered, brake_start_m=_BRAKE_START_M,
                                  brake_stop_m=_BRAKE_STOP_M, person_detected=False)
    assert scale == 1.0, scale


# --------------------------------------------------------------------------------------
# F3: lost_person_speed_taper_scale
# --------------------------------------------------------------------------------------

def test_lost_taper_full_speed_when_detected_or_briefly_lost():
    # None == "currently detected" -- the OPPOSITE None-semantics from the gap brake.
    assert lost_person_speed_taper_scale(None, taper_start_sec=3.0, taper_full_sec=6.0) == 1.0
    assert lost_person_speed_taper_scale(0.0, taper_start_sec=3.0, taper_full_sec=6.0) == 1.0
    assert lost_person_speed_taper_scale(3.0, taper_start_sec=3.0, taper_full_sec=6.0) == 1.0


def test_lost_taper_linear_ramp_to_zero_by_full_sec():
    mid = 3.0 + (6.0 - 3.0) / 2.0  # 4.5 s
    scale = lost_person_speed_taper_scale(mid, taper_start_sec=3.0, taper_full_sec=6.0)
    assert abs(scale - 0.5) < 1e-6, scale
    assert lost_person_speed_taper_scale(6.0, taper_start_sec=3.0, taper_full_sec=6.0) == 0.0
    assert lost_person_speed_taper_scale(32.0, taper_start_sec=3.0, taper_full_sec=6.0) == 0.0, \
        "the observed 32 s continuous loss from run_sim_20260711_140745_054 must be fully tapered"


def test_lost_taper_monotonic_decreasing():
    s_early = lost_person_speed_taper_scale(3.5, taper_start_sec=3.0, taper_full_sec=6.0)
    s_late = lost_person_speed_taper_scale(5.5, taper_start_sec=3.0, taper_full_sec=6.0)
    assert s_early > s_late, (s_early, s_late)


# --------------------------------------------------------------------------------------
# F3 scope contract (2026-07-11 regression fix, CLAUDE.md 8.15 correction)
# --------------------------------------------------------------------------------------
# lost_person_speed_taper_scale shipped applied at EVERY mid-climb stair speed path
# (persistence-latch forced climb, committed-climb, STAIR_LOSS_FLOOR, STAIR_APPROACH_COMMIT,
# and the near-stairs brief_loss floor in _apply_stair_command_policy) and fought incident
# 8.3's DESIGNED blind-carry: the patient rising out of the close-range camera FOV at the
# stair base is the NORMAL trigger for STAIR_LOSS_FLOOR, not a fault, so tapering the forward
# command toward zero there permanently parked the robot at the stair base
# (run_sim_20260711_150906: x=1.86, outcome robot_settled, climb never engaged). The taper is
# now scoped to the post-crest / top-landing phase ONLY, gated on the same one-way latch that
# arms the landing edge guard (_post_crest_landing_latched in core/main.py -- see the
# function's docstring in core/control/stair_policy.py). These are static source-scan
# contract tests (same approach as test_debug_info_ordering.py), not a runtime trace, since
# exercising main.py's frame loop needs the Isaac sim stack.

import re  # noqa: E402

_MAIN_PY = os.path.join(REPO, "core", "main.py")
_STAIR_POLICY_PY = os.path.join(REPO, "core", "control", "stair_policy.py")


def _call_site_lines(path, fn_name="lost_person_speed_taper_scale"):
    """Line numbers where fn_name is CALLED (not defined, imported, or merely mentioned in
    a comment/docstring) in the given source file."""
    call_re = re.compile(r"(?<!def )\b" + re.escape(fn_name) + r"\(")
    hits = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("def "):
                continue
            if call_re.search(line):
                hits.append(i)
    return hits


def test_taper_has_exactly_one_call_site_in_main_gated_on_post_crest_latch():
    """main.py may CALL the taper from exactly one place: the post-crest / top-landing
    forward-speed path, gated on _post_crest_landing_latched (the same one-way latch that
    arms the landing edge guard). Every prior call site (persistence-latch forced climb,
    committed-climb, STAIR_LOSS_FLOOR, STAIR_APPROACH_COMMIT) was mid-climb blind-carry
    (incident 8.3-class) and was removed -- a second, ungated call site would recreate the
    run_sim_20260711_150906 regression (CLAUDE.md 8.15 correction)."""
    hits = _call_site_lines(_MAIN_PY)
    assert len(hits) == 1, (
        f"expected exactly ONE lost_person_speed_taper_scale() call site in main.py, found "
        f"{len(hits)} at line(s) {hits} -- any mid-climb call site (persistence-latch forced "
        "climb, committed-climb, STAIR_LOSS_FLOOR, STAIR_APPROACH_COMMIT) recreates the "
        "run_sim_20260711_150906 regression; only the post-crest path may call it."
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    # The call must sit inside a block guarded by _post_crest_landing_latched -- scan
    # backward a generous window for the guard (comment lines are allowed in between).
    window = lines[max(0, call_line - 12):call_line - 1]
    assert any("_post_crest_landing_latched" in ln for ln in window), (
        f"lost_person_speed_taper_scale() call at main.py:{call_line} is not visibly gated "
        "on _post_crest_landing_latched within the preceding 12 lines -- it must only fire "
        "post-crest / top-landing (CLAUDE.md 8.15 correction)."
    )


def test_taper_not_called_in_apply_stair_command_policy():
    """_apply_stair_command_policy drives the near-stairs / brief_loss mid-climb floor --
    incident-8.3-class designed blind-carry, not a post-crest path. It must not call the
    taper (STAIR_LOSS_FLOOR does not apply it; only a post-crest-gated caller may)."""
    hits = _call_site_lines(_STAIR_POLICY_PY)
    assert hits == [], (
        f"lost_person_speed_taper_scale() called in stair_policy.py at line(s) {hits} -- "
        "expected zero call sites (see docstring: post-crest / top-landing only)."
    )


# --------------------------------------------------------------------------------------
# F4: detect_landing_edge_dropoff
# --------------------------------------------------------------------------------------

def _cfg():
    return HandoffConfig()


def synth_flat_depth(H=60, W=106, cam_h=0.40, vfov_deg=56.5, pitch_deg=0.5):
    """Mirrors test_pgtt_stair_handoff.synth_flat_depth (same camera model)."""
    cy = (H - 1) / 2.0
    vfov = math.radians(vfov_deg)
    pitch = math.radians(pitch_deg)
    D = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        theta_v = ((r - cy) / float(H)) * vfov
        ang = pitch + theta_v
        if ang > 0.02:
            D[r, :] = (cam_h / math.sin(ang)) * math.cos(theta_v)
    return D


def synth_landing_with_drop(H=60, W=106, cam_h=0.40, vfov_deg=56.5, pitch_deg=0.5,
                             edge_x_m=0.5, drop_m=0.4, max_range=3.0):
    """Ray-marches a flat floor that steps DOWN by drop_m at edge_x_m -- the descending-edge
    mirror of test_pgtt_stair_handoff.synth_staircase_depth's ascending riser."""
    cy = (H - 1) / 2.0
    vfov = math.radians(vfov_deg)
    pitch = math.radians(pitch_deg)

    def floor_z(px):
        return -drop_m if px >= edge_x_m else 0.0

    D = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        theta_v = ((r - cy) / float(H)) * vfov
        ang = pitch + theta_v
        if ang <= 1e-3:
            continue
        t = 0.0
        hit = None
        while t < max_range:
            px = t * math.cos(ang)
            pz = cam_h - t * math.sin(ang)
            if pz <= floor_z(px):
                hit = t
                break
            t += 0.004
        D[r, :] = (hit * math.cos(theta_v)) if hit is not None else 0.0
    return D


def test_edge_guard_flat_landing_is_clear():
    D_mm = synth_flat_depth() * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is False


def test_edge_guard_confirms_drop_within_reach():
    D_mm = synth_landing_with_drop(edge_x_m=0.5, drop_m=0.4) * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is True


def test_edge_guard_ignores_drop_beyond_reach():
    D_mm = synth_landing_with_drop(edge_x_m=1.5, drop_m=0.4) * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is False


def test_edge_guard_none_depth_reports_unknown():
    """Unknown (None), not a snap decision -- the CALLER fails toward stopping (incident 8.8),
    this function only reports that it could not evaluate."""
    assert detect_landing_edge_dropoff(None, _cfg(), reach_m=0.8, drop_m=0.3) is None


def test_edge_guard_no_valid_rows_fails_toward_edge():
    empty = np.zeros((60, 106), dtype=np.float32)
    assert detect_landing_edge_dropoff(empty, _cfg(), reach_m=0.8, drop_m=0.3) is True


# --------------------------------------------------------------------------------------
# F4 follow-up (2026-07-11 review of run_sim_20260711_223152_489): the shared depth
# back-projection (detect_landing_edge_dropoff / DepthStairDetector.detect) assumes a
# near-level camera (cfg.stair_cam_pitch_deg, a STATIC constant read from a HandoffConfig()
# built once at startup -- core/main.py:193-194) but the parkour depth camera is rigidly
# body-parented (sim/isaac/env/cameras.py:99-134) and truly inherits the robot's per-frame
# body pitch. Right after cresting, that run's GT body pitch sat at -8.6..-9.9 deg (well past
# the 5 deg level band) for 40+ consecutive frames while the dog held still, and
# detect_landing_edge_dropoff falsely read a descending edge on a genuinely flat, level
# landing -- forcing trans_x_cmd=0.0 forever (the deadlock). Reproduced below by feeding the
# SAME true-pitch deviation into synth_flat_depth (a perfectly flat, level floor with NO
# drop anywhere) while detect_landing_edge_dropoff still assumes the calibrated 0.5 deg.
# --------------------------------------------------------------------------------------

def test_edge_guard_false_positive_on_flat_floor_from_unsettled_crest_pitch():
    """Reproduces the run_sim_20260711_223152_489 deadlock: a perfectly flat, level floor
    (no drop anywhere) rendered from a camera whose TRUE pitch includes the robot's actual
    post-crest body attitude (-9.56 deg, the run's observed value) reads as a confirmed
    descending edge once back-projected with the static 0.5 deg calibration alone."""
    D_mm = synth_flat_depth(pitch_deg=0.5 + (-9.56)) * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is True


def test_edge_guard_true_positive_unaffected_at_level_pitch():
    """Sanity check that the reproduction above is about the PITCH deviation, not some other
    change: at the calibrated level pitch (the default), the same flat floor reads clear."""
    D_mm = synth_flat_depth(pitch_deg=0.5) * 1000.0
    assert detect_landing_edge_dropoff(D_mm, _cfg(), reach_m=0.8, drop_m=0.3) is False


# --------------------------------------------------------------------------------------
# landing_edge_guard_suppress_crest_artifact -- crest-aware suppression of the above false
# positive. Sim path uses the GT crest-relative distance (mirrors, but does not modify or
# call, _fully_on_top_landing / LandingMarginState); hardware fallback uses wall-clock time
# since the post-crest latch armed (incident 8.6: a duration, never a frame count). Both
# require the commanded direction to be AWAY from the crest (direction-of-travel
# discrimination). See the function's docstring for the full root-cause writeup.
# --------------------------------------------------------------------------------------

def test_crest_suppress_active_just_past_crest_moving_away_sim():
    """The run's actual position: 0.146 m past the GT crest entry, commanded forward
    (away from the crest, toward the patient) -- must suppress."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=0.146,
        since_crest_latch_sec=None,
        commanded_away_from_crest=True,
        suppress_reach_m=0.5,
        suppress_time_sec=2.0,
    ) is True


def test_crest_suppress_inactive_beyond_reach_sim():
    """Regression for the guard's ORIGINAL purpose (run_sim_20260711_140745_054: a genuine
    drop-off 2-3 m past the crest, well outside any crest-transition window) -- must NOT
    suppress; a confirmed finding this far from the crest still blocks."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=2.5,
        since_crest_latch_sec=None,
        commanded_away_from_crest=True,
        suppress_reach_m=0.5,
        suppress_time_sec=2.0,
    ) is False


def test_crest_suppress_boundary_is_half_open():
    """Exactly AT the suppression reach must no longer suppress (matches the guard's other
    half-open interval conventions, e.g. detect_landing_edge_dropoff's reach_m check)."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=0.5, since_crest_latch_sec=None,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=0.499, since_crest_latch_sec=None,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is True


def test_crest_suppress_never_when_commanded_toward_crest():
    """Direction-of-travel discrimination: even well within the crest window, a command
    that is NOT away from the crest (zero/backward) must never be suppressed."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=0.1, since_crest_latch_sec=0.1,
        commanded_away_from_crest=False, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False


def test_crest_suppress_hardware_fallback_uses_wall_clock_time():
    """No GT crest position (hardware): falls back to wall-clock seconds since the
    post-crest latch armed -- a duration, per incident 8.6, not a frame count."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=None, since_crest_latch_sec=1.5,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is True
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=None, since_crest_latch_sec=2.5,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False


def test_crest_suppress_fails_toward_blocking_with_no_signal_at_all():
    """Neither GT distance nor a latch timestamp is available: fail toward NOT suppressing
    (incident 8.8 -- the edge guard's own fail-toward-stop bias must not be defeated by a
    missing suppression signal)."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=None, since_crest_latch_sec=None,
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False


def test_crest_suppress_prefers_sim_gt_over_wall_clock_when_both_present():
    """If both signals happen to be available, the more precise sim GT distance drives the
    decision (a stale/irrelevant elapsed-time value must not override it)."""
    assert landing_edge_guard_suppress_crest_artifact(
        crest_relative_m=2.5, since_crest_latch_sec=0.1,  # GT says far past crest
        commanded_away_from_crest=True, suppress_reach_m=0.5, suppress_time_sec=2.0,
    ) is False


# --------------------------------------------------------------------------------------
# F3 third rescope (2026-07-11 review): _fully_on_top_landing -- "crest reached" != "safe
# to hold/taper". run_sim_20260711_195618_941: the dog stopped STRADDLING the crest lip
# (front feet on the landing, rear feet still on the last riser -- GT phase "staircase",
# x=6.15/end_x_m=6.27, pitch=-8.5 deg) with _post_crest_landing_latched already True, then
# rolled over after ~5s of commanded-zero hold. Separately, _post_crest_landing_latched
# itself was found to have latched at t=48.76s / x=1.27m -- 4+ metres and ~40s before the
# real staircase -- because _crest_reached's sim-GT fallback accepts phase=="flat_follow",
# which is ALSO the ordinary pre-stairs approach phase (_terrain_phase in
# world/sim_go2_stairs.py). _fully_on_top_landing gates the post-crest taper on a level
# pitch AND a travel/time margin computed fresh from frame_meta every call (not from the
# possibly-false latch), so neither failure mode can make it fire early.
# --------------------------------------------------------------------------------------

_LEVEL_DEG = 5.0
_MARGIN_M = 0.45
_MARGIN_TIME_SEC = 1.5


def _stair_demo_frame_meta(phase, x_m, pitch_deg):
    return {
        "stair_demo": {
            "phase": phase,
            "robot": {"x_m": x_m, "pitch_deg": pitch_deg},
        }
    }


def test_fully_on_landing_inactive_at_straddle_pitch():
    """The observed failure signature: front-on-landing/rear-on-riser straddle. GT phase is
    still "staircase" (x=6.15 < end_x_m=6.27) and pitch is still steep (-8.5 deg, over the
    5 deg level threshold) -- must NOT read as fully on the landing."""
    state = LandingMarginState()
    fm = _stair_demo_frame_meta("staircase", x_m=6.15, pitch_deg=-8.5)
    result = _fully_on_top_landing(
        fm, {}, state, now=100.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert result is False, result


def test_fully_on_landing_inactive_on_stairs_even_when_momentarily_level():
    """Isolates the POSITION criterion from the pitch criterion: even with a level pitch
    (<=5 deg), still-on-the-stairs GT phase ("staircase") must not immediately count as fully
    on the landing -- the hardware-portable time-margin fallback requires the level pitch to
    be SUSTAINED for margin_time_sec, not just true for one call."""
    state = LandingMarginState()
    fm = _stair_demo_frame_meta("staircase", x_m=6.20, pitch_deg=-2.0)  # level, but on stairs
    first = _fully_on_top_landing(
        fm, {}, state, now=100.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert first is False, "a single level-pitch sample must not immediately pass the margin"


def test_fully_on_landing_active_once_level_and_beyond_crest_margin():
    """Once GT phase genuinely reads top_landing (a reliable, purely x-derived signal -- see
    _terrain_phase) and the dog has walked margin_m past the FIRST confirmed sample with a
    level pitch, the gate must turn on."""
    state = LandingMarginState()
    # First top_landing sample -- captures the entry x. Margin not yet met.
    fm0 = _stair_demo_frame_meta("top_landing", x_m=6.30, pitch_deg=-3.0)
    assert _fully_on_top_landing(
        fm0, {}, state, now=100.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    ) is False
    # Still short of the 0.45 m margin.
    fm1 = _stair_demo_frame_meta("top_landing", x_m=6.55, pitch_deg=-1.0)
    assert _fully_on_top_landing(
        fm1, {}, state, now=100.5,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    ) is False
    # Past entry_x (6.30) + margin_m (0.45) = 6.75, and level.
    fm2 = _stair_demo_frame_meta("top_landing", x_m=6.80, pitch_deg=-0.5)
    assert _fully_on_top_landing(
        fm2, {}, state, now=101.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    ) is True


def test_fully_on_landing_pre_stairs_false_latch_not_immediately_active():
    """Regression for the newly-discovered _post_crest_landing_latched early-false-latch
    (run_sim_20260711_195618_941: latched at t=48.76s, x=1.27 m, phase=="flat_follow" -- well
    BEFORE the real staircase). A single flat-ground pre-stairs sample must not immediately
    read as fully on the landing (phase != "top_landing", so it falls to the time-margin
    fallback, which requires SUSTAINED level pitch -- not met on the very first sample)."""
    state = LandingMarginState()
    fm = _stair_demo_frame_meta("flat_follow", x_m=1.27, pitch_deg=-0.2)  # pre-stairs, level
    result = _fully_on_top_landing(
        fm, {}, state, now=48.76,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert result is False, result


def test_fully_on_landing_stair_base_regression_unchanged():
    """Regression for CLAUDE.md 8.15 correction 1 (run 2026-07-11_150906): mid-climb / stair
    BASE scenarios must stay False so the caller's `_post_crest_landing_latched and
    _fully_on_landing_now` AND-gate in main.py can only narrow when the taper fires, never
    widen it back into the stair-base blind-carry the correction protects."""
    state = LandingMarginState()
    fm = _stair_demo_frame_meta("stair_approach", x_m=1.86, pitch_deg=-1.0)
    result = _fully_on_top_landing(
        fm, {}, state, now=10.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert result is False, result


def test_fully_on_landing_no_pitch_reading_fails_toward_not_on_landing():
    """Incident 8.8: with no pitch reading available at all, fail toward "not fully on the
    landing" (never let the taper/hold-easing this gates turn on blind)."""
    state = LandingMarginState()
    fm = {"stair_demo": {"phase": "top_landing", "robot": {"x_m": 6.8}}}  # no pitch_deg
    result = _fully_on_top_landing(
        fm, {}, state, now=100.0,
        level_deg=_LEVEL_DEG, margin_m=_MARGIN_M, margin_time_sec=_MARGIN_TIME_SEC,
    )
    assert result is False, result


# --------------------------------------------------------------------------------------
# F5 (2026-07-11 second review of run_sim_20260711_195618_941): _crest_reached's sim-GT
# fallback false-latches on the ordinary PRE-stairs approach. "flat_follow" (and the
# standalone level-pitch check) are true for the entire approach before the dog has ever
# seen a riser -- this function's only caller already requires stairs_detected=True (a
# distant YOLO sighting is enough) to even be reached, so a brief person loss anywhere on
# the approach used to read as "crest reached". Fixed by gating those two arms on
# stairs_ever_confirmed (True once a genuine depth-confirmed stair reading has happened at
# least once this run); "top_landing" stays unconditional (it can ONLY be true past the
# real crest).
# --------------------------------------------------------------------------------------

def test_crest_reached_flat_follow_pre_stairs_without_evidence_is_false():
    """The observed false-latch signature verbatim: run_sim_20260711_195618_941 latched
    _post_crest_landing_latched at t=48.76s, x=1.27 m, phase=="flat_follow", pitch=-0.29 deg
    -- 4+ metres before start_x_m=2.0. Without prior stair evidence this must NOT read as
    the crest."""
    fm = _stair_demo_frame_meta("flat_follow", x_m=1.27, pitch_deg=-0.29)
    assert _crest_reached(fm, {}, stairs_ever_confirmed=False) is False
    # Default (no kwarg passed) must be the same safe False -- never silently permissive.
    assert _crest_reached(fm, {}) is False


def test_crest_reached_top_landing_always_true_regardless_of_evidence():
    """"top_landing" is a reliable, purely x-position-derived phase (_terrain_phase) -- it
    can ONLY be true past the real crest, so it stays unconditional (unchanged behaviour)."""
    fm = _stair_demo_frame_meta("top_landing", x_m=6.8, pitch_deg=-1.0)
    assert _crest_reached(fm, {}, stairs_ever_confirmed=False) is True
    assert _crest_reached(fm, {}, stairs_ever_confirmed=True) is True


def test_crest_reached_flat_follow_true_after_stair_evidence():
    """Once genuine stair evidence has been observed at least once this run (a real climb
    cannot happen without it), the SAME flat_follow phase legitimately means "back on flat
    ground after the stairs" and must count as the crest."""
    fm = _stair_demo_frame_meta("flat_follow", x_m=6.9, pitch_deg=-0.1)
    assert _crest_reached(fm, {}, stairs_ever_confirmed=True) is True


def test_crest_reached_level_pitch_fallback_also_gated_on_evidence():
    """The standalone level-pitch fallback (independent of phase) has the identical
    pre-stairs ambiguity as the flat_follow phase arm -- pitch is near-level for the whole
    approach too -- so it is gated the same way. Without evidence, a level pitch on a
    still-approaching phase must not count; with evidence, it must (e.g. a brief level pause
    mid-descent-from-the-crest)."""
    fm = _stair_demo_frame_meta("stair_approach", x_m=1.9, pitch_deg=0.5)
    assert _crest_reached(fm, {}, stairs_ever_confirmed=False) is False
    assert _crest_reached(fm, {}, stairs_ever_confirmed=True) is True


def test_crest_reached_hardware_sensor_path_unaffected():
    """The preferred hardware-sensor arm (sensor_imu_pitch / sensor_riser_dist_ahead) is
    OUT OF SCOPE for this fix and must be unaffected by stairs_ever_confirmed."""
    fm = {"sensor_imu_pitch": 0.05}  # ~2.9 deg, flat-enough
    assert _crest_reached(fm, {}, stairs_ever_confirmed=False) is True
    fm2 = {"sensor_riser_dist_ahead": 2.0}  # no riser within a tread ahead
    assert _crest_reached(fm2, {}, stairs_ever_confirmed=False) is True


# --------------------------------------------------------------------------------------
# F5: stair_loss_floor_eligible -- main.py's STAIR_LOSS_FLOOR dispatch branch must not fall
# through to a plain controller.stop() stance-lock while the climb persistence latch is
# still on and the dog is not yet fully clear of the stairs (run_sim_20260711_195618_941:
# straddled the crest lip, person lost, rolled off the 2.1 m edge, roll 4.3 -> 148 deg).
# --------------------------------------------------------------------------------------

def test_stair_loss_floor_eligible_stairs_now_alone_is_sufficient():
    """Unchanged pre-existing behaviour: a fresh genuine on-stairs detection is always
    eligible on its own, regardless of the latch/landing state."""
    assert stair_loss_floor_eligible(
        stairs_now=True, stair_climbing_latch=False,
        person_detected=False, fully_on_top_landing=False,
    ) is True


def test_stair_loss_floor_eligible_run6_terminal_state_straddle_not_stop():
    """The run-6 terminal state this fix targets: genuine detection gone stale
    (stairs_now=False) but the climb persistence latch still on, person lost, and the dog
    straddling the crest lip -- NOT fully on the landing. Must be eligible (the branch that
    keeps the gait alive with the forward floor/gap-brake), not fall through to stop()."""
    eligible = stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=True,
        person_detected=False, fully_on_top_landing=False,
    )
    assert eligible is True, (
        "run_sim_20260711_195618_941 terminal state (straddle, latch on, person lost, not "
        "fully on landing) must stay eligible for the forward-floor branch, not fall "
        "through to controller.stop() (the roll 4.3 -> 148 deg topple)"
    )


def test_stair_loss_floor_eligible_false_once_fully_on_landing():
    """Same latch/loss state, but now genuinely clear of the stairs (fully_on_top_landing
    True): a continued loss there is a "patient walked away on the flat landing" case, not
    incident-8.3 blind-carry -- the post-crest hold/taper path must take back over, so this
    branch must stop claiming eligibility."""
    eligible = stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=True,
        person_detected=False, fully_on_top_landing=True,
    )
    assert eligible is False, (
        "once fully_on_top_landing is True the latch-only arm must yield to the post-crest "
        "hold/taper path, not keep forcing the forward floor"
    )


def test_stair_loss_floor_eligible_person_visible_is_not_the_loss_case():
    """If the person IS currently detected, motion_allowed would ordinarily be driven by the
    live follow path already (this branch's own dispatch position is only reached when
    motion_allowed is False) -- but as a defence-in-depth boundary this predicate must not
    grant eligibility purely off the latch while a person is visible; that is not the
    incident-8.3 loss case this branch exists for."""
    eligible = stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=True,
        person_detected=True, fully_on_top_landing=False,
    )
    assert eligible is False, eligible


def test_stair_loss_floor_eligible_stair_base_regression_unchanged():
    """CLAUDE.md 8.15 correction 1 (run 2026-07-11_150906): before ANY genuine stair
    evidence has ever been observed, stair_climbing_latch is False (it only turns True once
    genuine evidence has been seen at least once -- see the persistence-latch block in
    core/main.py), so at the stair BASE this predicate is unreachable via the latch arm --
    identical to the pre-existing stairs_now-only behaviour. Must stay False, unchanged."""
    eligible = stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=False,
        person_detected=False, fully_on_top_landing=False,
    )
    assert eligible is False, eligible


def test_stair_loss_floor_eligible_both_false_falls_through():
    """Neither stairs_now nor a still-relevant latch: correctly NOT eligible (dispatch
    should continue evaluating the later STAIR_APPROACH_COMMIT / FLAT_LOSS_GLIDE / stop()
    branches, unchanged from today)."""
    assert stair_loss_floor_eligible(
        stairs_now=False, stair_climbing_latch=False,
        person_detected=True, fully_on_top_landing=False,
    ) is False


# --------------------------------------------------------------------------------------
# Static wiring contract: main.py's STAIR_LOSS_FLOOR dispatch elif must actually call
# stair_loss_floor_eligible(...) (not just stairs_now) so the predicate tested above is the
# real gate, not a dead extra function (same approach as
# test_taper_has_exactly_one_call_site_in_main_gated_on_post_crest_latch above).
# --------------------------------------------------------------------------------------

def test_main_dispatch_calls_stair_loss_floor_eligible():
    hits = _call_site_lines(_MAIN_PY, fn_name="stair_loss_floor_eligible")
    assert len(hits) == 1, (
        f"expected exactly ONE stair_loss_floor_eligible() call site in main.py, found "
        f"{len(hits)} at line(s) {hits}"
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    # The call must be reachable from the STAIR_LOSS_FLOOR elif -- scan forward a small
    # window for the four expected keyword arguments feeding the predicate.
    window = "".join(lines[call_line - 1: call_line + 6])
    for kw in ("stairs_now=", "stair_climbing_latch=", "person_detected=",
               "fully_on_top_landing="):
        assert kw in window, (
            f"stair_loss_floor_eligible() call at main.py:{call_line} is missing the "
            f"expected keyword argument {kw!r}"
        )


def test_main_dispatch_calls_landing_edge_guard_suppress_crest_artifact():
    """main.py must actually call landing_edge_guard_suppress_crest_artifact(...) when a
    descending edge is confirmed, so the false-positive fix above is wired into the real
    dispatch, not just a tested-but-dead extra function (same approach as
    test_main_dispatch_calls_stair_loss_floor_eligible above)."""
    hits = _call_site_lines(_MAIN_PY, fn_name="landing_edge_guard_suppress_crest_artifact")
    assert len(hits) == 1, (
        f"expected exactly ONE landing_edge_guard_suppress_crest_artifact() call site in "
        f"main.py, found {len(hits)} at line(s) {hits}"
    )
    call_line = hits[0]
    with open(_MAIN_PY, encoding="utf-8") as f:
        lines = f.readlines()
    # A wider window than the other wiring-contract tests in this file: this call site has an
    # inline comment (explaining the >= 0.0 vs > 0.0 choice for commanded_away_from_crest)
    # between two of its keyword arguments.
    window = "".join(lines[call_line - 1: call_line + 17])
    for kw in ("crest_relative_m=", "since_crest_latch_sec=",
               "commanded_away_from_crest=", "suppress_reach_m=", "suppress_time_sec="):
        assert kw in window, (
            f"landing_edge_guard_suppress_crest_artifact() call at main.py:{call_line} is "
            f"missing the expected keyword argument {kw!r}"
        )


if __name__ == "__main__":
    test_gap_brake_full_speed_at_or_above_start()
    test_gap_brake_linear_taper_between_thresholds()
    test_gap_brake_zero_at_or_below_stop()
    test_gap_brake_fails_toward_brake_on_invalid_gap_with_person_visible()
    test_gap_brake_no_brake_when_person_not_detected()
    test_gap_brake_filter_never_releases_on_noisy_trace_replay()
    test_gap_brake_filter_releases_within_two_seconds_of_genuine_opening()
    test_gap_brake_filter_resets_on_a_detection_gap_longer_than_the_window()
    test_gap_brake_filter_no_brake_when_not_detected_regardless_of_window_contents()
    test_lost_taper_full_speed_when_detected_or_briefly_lost()
    test_lost_taper_linear_ramp_to_zero_by_full_sec()
    test_lost_taper_monotonic_decreasing()
    test_taper_has_exactly_one_call_site_in_main_gated_on_post_crest_latch()
    test_taper_not_called_in_apply_stair_command_policy()
    test_edge_guard_flat_landing_is_clear()
    test_edge_guard_confirms_drop_within_reach()
    test_edge_guard_ignores_drop_beyond_reach()
    test_edge_guard_none_depth_reports_unknown()
    test_edge_guard_no_valid_rows_fails_toward_edge()
    test_edge_guard_false_positive_on_flat_floor_from_unsettled_crest_pitch()
    test_edge_guard_true_positive_unaffected_at_level_pitch()
    test_crest_suppress_active_just_past_crest_moving_away_sim()
    test_crest_suppress_inactive_beyond_reach_sim()
    test_crest_suppress_boundary_is_half_open()
    test_crest_suppress_never_when_commanded_toward_crest()
    test_crest_suppress_hardware_fallback_uses_wall_clock_time()
    test_crest_suppress_fails_toward_blocking_with_no_signal_at_all()
    test_crest_suppress_prefers_sim_gt_over_wall_clock_when_both_present()
    test_fully_on_landing_inactive_at_straddle_pitch()
    test_fully_on_landing_inactive_on_stairs_even_when_momentarily_level()
    test_fully_on_landing_active_once_level_and_beyond_crest_margin()
    test_fully_on_landing_pre_stairs_false_latch_not_immediately_active()
    test_fully_on_landing_stair_base_regression_unchanged()
    test_fully_on_landing_no_pitch_reading_fails_toward_not_on_landing()
    test_crest_reached_flat_follow_pre_stairs_without_evidence_is_false()
    test_crest_reached_top_landing_always_true_regardless_of_evidence()
    test_crest_reached_flat_follow_true_after_stair_evidence()
    test_crest_reached_level_pitch_fallback_also_gated_on_evidence()
    test_crest_reached_hardware_sensor_path_unaffected()
    test_stair_loss_floor_eligible_stairs_now_alone_is_sufficient()
    test_stair_loss_floor_eligible_run6_terminal_state_straddle_not_stop()
    test_stair_loss_floor_eligible_false_once_fully_on_landing()
    test_stair_loss_floor_eligible_person_visible_is_not_the_loss_case()
    test_stair_loss_floor_eligible_stair_base_regression_unchanged()
    test_stair_loss_floor_eligible_both_false_falls_through()
    test_main_dispatch_calls_stair_loss_floor_eligible()
    test_main_dispatch_calls_landing_edge_guard_suppress_crest_artifact()
    print("ALL STAIR SPEED GUARD TESTS PASS")
