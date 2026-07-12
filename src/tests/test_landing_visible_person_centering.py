"""Regression tests for the post-crest "visible person" landing-centering mode (host-safe).

Task (2026-07-12, run 28 review, run_sim_20260712_141230_357): run 27's gap (fixed by
landing_face_patient_align / test_landing_face_patient_align.py) was a LOST-person endgame.
Run 28 hit the mirror-image case -- the patient stayed person_detected=True the whole endgame
(rotation_error_deg steady at approx -24 deg for the final ~13 s), so the lost-case machine
never engaged and the dog sat in an ordinary standoff hold staring ~24 deg past the patient
(hold=True zeroes wz on the sim side regardless of what ordinary follow steering computed).
landing_visible_person_centering adds a second, RE-ARMABLE mode (NOT a one-way latch) that
closes the loop on the live bearing during any hold once fully clear of the stairs. See its
docstring (core/control/stair_policy.py) for the full hysteresis / two-backstop design.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.control.stair_policy import (
    LandingFaceAlignState,
    LandingVisibleCenterState,
    landing_visible_person_centering,
)

_DEFAULTS = dict(
    engage_deg=15.0,
    deadband_deg=8.0,
    max_rotation_deg=120.0,
    total_rotation_budget_deg=240.0,
    yaw_rate=0.35,
)


def _center(state, *, trigger, bearing_deg, now_wall, sim_t=None, **overrides):
    kwargs = dict(_DEFAULTS)
    kwargs.update(overrides)
    return landing_visible_person_centering(
        trigger=trigger,
        bearing_deg=bearing_deg,
        state=state,
        now_wall=now_wall,
        sim_t=sim_t,
        **kwargs,
    )


def test_inactive_until_trigger():
    # Never triggered -> never rotates, regardless of bearing.
    state = LandingVisibleCenterState()
    r = _center(state, trigger=False, bearing_deg=-90.0, now_wall=0.0)
    assert r.active is False
    assert r.yaw_rate_cmd == 0.0
    assert r.budget_exhausted is False


def test_hysteresis_engages_above_threshold_and_stops_inside_deadband():
    # Schmitt trigger: HIGH threshold = engage_deg (15), LOW threshold = deadband_deg (8).
    state = LandingVisibleCenterState()
    # Between deadband and engage while idle -> must NOT start (avoids chattering right at a
    # single shared boundary).
    r = _center(state, trigger=True, bearing_deg=-10.0, now_wall=0.0)
    assert r.active is False
    # Crosses the engage threshold -> starts, bang-bang at the fixed yaw_rate magnitude.
    r = _center(state, trigger=True, bearing_deg=-20.0, now_wall=0.1)
    assert r.active is True
    assert r.yaw_rate_cmd == 0.35  # -copysign(0.35, -20) -> +0.35 (turn left)
    # Drifts back into the band between deadband and engage -- an ALREADY-active engagement
    # keeps going (only the deadband, not the engage threshold, stops it once started).
    r = _center(state, trigger=True, bearing_deg=-10.0, now_wall=0.2)
    assert r.active is True
    assert r.yaw_rate_cmd == 0.35
    # Crosses into the deadband -> stops.
    r = _center(state, trigger=True, bearing_deg=-5.0, now_wall=0.3)
    assert r.active is False
    assert r.yaw_rate_cmd == 0.0


def test_positive_bearing_yields_negative_yaw_rate_cmd():
    # Patient on the RIGHT (+bearing) -> turn RIGHT -> negative yaw_rate_cmd (same sign
    # convention as landing_face_patient_align / main.py's forward-pursuit arc-yaw).
    state = LandingVisibleCenterState()
    r = _center(state, trigger=True, bearing_deg=30.0, now_wall=0.0)
    assert r.yaw_rate_cmd == -0.35


def test_rearm_after_reaching_deadband_when_bearing_grows_again():
    state = LandingVisibleCenterState()
    r1 = _center(state, trigger=True, bearing_deg=-40.0, now_wall=0.0)
    assert r1.active is True
    r2 = _center(state, trigger=True, bearing_deg=-4.0, now_wall=0.1)  # inside deadband -> stops
    assert r2.active is False
    # Between deadband and engage while idle -- must NOT restart.
    r3 = _center(state, trigger=True, bearing_deg=10.0, now_wall=0.2)
    assert r3.active is False
    # Crosses the engage threshold again (opposite side this time) -> re-arms.
    r4 = _center(state, trigger=True, bearing_deg=25.0, now_wall=0.3)
    assert r4.active is True
    assert r4.yaw_rate_cmd == -0.35


def test_yields_on_trigger_drop_mid_centering_lost_case_untouched():
    # Task hard constraint 3: person loss mid-centering -> commands 0 immediately, no
    # persistent memory of the interrupted engagement.
    state = LandingVisibleCenterState()
    r1 = _center(state, trigger=True, bearing_deg=-40.0, now_wall=0.0)
    assert r1.active is True
    r2 = _center(state, trigger=False, bearing_deg=-40.0, now_wall=0.1)
    assert r2.active is False
    assert r2.yaw_rate_cmd == 0.0
    assert r2.budget_exhausted is False
    # A separate lost-case state instance is never touched by this function -- it has its own
    # trigger (landing_lost_person_hold_active) and no coupling to LandingVisibleCenterState.
    lost_state = LandingFaceAlignState()
    assert lost_state.engaged is False


def test_rotation_proceeds_uninterrupted_no_edge_block_veto():
    # CORRECTED 2026-07-12 (run-28 review, run_sim_20260712_141230_357): this function used to
    # mirror landing_face_patient_align exactly -- an edge_block argument withheld rotation
    # ("active=False, yaw_rate_cmd=0.0") for the frame it was True, without resetting
    # state.centering. Removed for the same reason documented in landing_face_patient_align's
    # own EDGE-GUARD PRECEDENCE docstring paragraph: the edge latch is chronic at the dog's
    # terminal post-crest pose, so the veto made this mode unable to ever center in exactly
    # the endgame it exists for (run 28: patient visible at -24 deg for 13+ s, zero rotation).
    # Rotation now proceeds uninterrupted through an in-progress engagement -- nothing can
    # pause it mid-turn anymore. The replacement safety net (a measured PHYSICAL planar-drift
    # trip, go2_locomotion.yaw_align_drift.YawAlignDriftWatchdog) is unit-tested in
    # test_yaw_align_drift_watchdog.py, not here.
    state = LandingVisibleCenterState()
    r1 = _center(state, trigger=True, bearing_deg=-90.0, now_wall=0.0)
    assert r1.active is True
    r2 = _center(state, trigger=True, bearing_deg=-90.0, now_wall=0.1)
    assert r2.active is True
    assert r2.yaw_rate_cmd == 0.35
    # The SAME engagement keeps going, uninterrupted -- a bearing between deadband(8) and
    # engage(15) would never have STARTED a fresh engagement from idle, but this one continues
    # because it was already active (Schmitt-trigger "already active" rule, not an edge-block
    # resume).
    r3 = _center(state, trigger=True, bearing_deg=-10.0, now_wall=0.2)
    assert r3.active is True
    assert r3.yaw_rate_cmd == 0.35


def test_cumulative_budget_exhaustion_disables_future_rotation():
    state = LandingVisibleCenterState()
    now = 0.0
    r = None
    for _ in range(5000):
        r = _center(
            state, trigger=True, bearing_deg=-90.0, now_wall=now,
            max_rotation_deg=1000.0, total_rotation_budget_deg=5.0, yaw_rate=0.35,
        )
        now += 0.01
        if r.budget_exhausted:
            break
    assert r.budget_exhausted is True
    assert r.active is False
    assert r.yaw_rate_cmd == 0.0
    # Permanent: even a large bearing swing afterward never rotates again.
    r2 = _center(
        state, trigger=True, bearing_deg=90.0, now_wall=now + 1.0,
        max_rotation_deg=1000.0, total_rotation_budget_deg=5.0,
    )
    assert r2.active is False
    assert r2.budget_exhausted is True
    assert r2.yaw_rate_cmd == 0.0


def test_per_engagement_bound_ends_one_engagement_not_the_whole_budget():
    # The per-engagement bound (reuses --landing-face-patient-max-rotation-deg) and the
    # cumulative budget are two DISTINCT counters (task brief: "state clearly which you did" --
    # kept separate). Tripping the small per-engagement bound must end only that engagement,
    # leaving the (here, huge) cumulative budget unexhausted so a later re-arm still works.
    state = LandingVisibleCenterState()
    now = 0.0
    r = None
    for _ in range(5000):
        r = _center(
            state, trigger=True, bearing_deg=-90.0, now_wall=now,
            max_rotation_deg=5.0, total_rotation_budget_deg=1000.0, yaw_rate=0.35,
        )
        now += 0.01
        if not r.active:
            break
    assert r.active is False
    assert r.budget_exhausted is False
    assert state.total_rotated_rad > 0.0
    r2 = _center(
        state, trigger=True, bearing_deg=-90.0, now_wall=now + 1.0,
        max_rotation_deg=5.0, total_rotation_budget_deg=1000.0, yaw_rate=0.35,
    )
    assert r2.active is True
    assert r2.budget_exhausted is False


def test_disable_via_zero_yaw_rate_shared_with_lost_case():
    # Shared disable knob (--landing-face-patient-yaw-rate 0): disables both modes' rotation.
    state = LandingVisibleCenterState()
    r = _center(state, trigger=True, bearing_deg=-90.0, now_wall=0.0, yaw_rate=0.0)
    assert r.active is False
    assert r.yaw_rate_cmd == 0.0
    assert r.budget_exhausted is False


def test_disable_via_zero_engage_deg_is_visible_mode_only():
    # Visible-mode-ONLY disable (--landing-face-patient-track-engage-deg 0), tested on the RAW
    # argument -- CLAUDE.md zero-as-disabled-sentinel lesson: an engage_deg of exactly 0.0 must
    # not be read as "trigger on any nonzero bearing" (the opposite of disabled). A bearing far
    # outside even the deadband (-90) still produces no rotation, proving the raw arg was
    # checked rather than some derived "already aligned" condition.
    state = LandingVisibleCenterState()
    r = _center(state, trigger=True, bearing_deg=-90.0, now_wall=0.0, engage_deg=0.0)
    assert r.active is False
    assert r.yaw_rate_cmd == 0.0


def test_none_bearing_does_nothing():
    state = LandingVisibleCenterState()
    r = _center(state, trigger=True, bearing_deg=None, now_wall=0.0)
    assert r.active is False
    assert r.yaw_rate_cmd == 0.0
    assert r.budget_exhausted is False


def test_visible_center_never_emits_translation():
    # Sanity: the function has no vx/vy/hold concept at all -- only ever a yaw rate (task hard
    # constraint 1). Documents the contract the caller relies on: translation is governed
    # entirely by whatever hold reason already asserted it, independent of this result.
    state = LandingVisibleCenterState()
    r = _center(state, trigger=True, bearing_deg=-90.0, now_wall=0.0)
    assert not hasattr(r, "trans_x_cmd")
    assert not hasattr(r, "vx")
    assert not hasattr(r, "hold")
    assert not hasattr(r, "stop_decision")


def test_sim_time_aware_dt_not_wall_clock():
    # Incident 8.6: the per-engagement rotation accrual must integrate against sim_t when
    # available, not wall time (mirrors landing_face_patient_align's own sim-aware timing --
    # reused here via the same DetectionAgeState/detection_age_sec dual-clock primitive).
    state = LandingVisibleCenterState()
    r1 = _center(state, trigger=True, bearing_deg=-90.0, now_wall=0.0, sim_t=10.0)
    assert r1.active is True
    assert state.engagement_rotated_rad == 0.0  # first call: no prior anchor, dt=0
    # Big WALL jump, tiny SIM jump -> integrates only the tiny sim delta.
    r2 = _center(state, trigger=True, bearing_deg=-90.0, now_wall=90.0, sim_t=10.5)
    assert r2.active is True
    import math
    expected = 0.35 * 0.5
    assert math.isclose(state.engagement_rotated_rad, expected, rel_tol=1e-6)


if __name__ == "__main__":
    test_inactive_until_trigger()
    test_hysteresis_engages_above_threshold_and_stops_inside_deadband()
    test_positive_bearing_yields_negative_yaw_rate_cmd()
    test_rearm_after_reaching_deadband_when_bearing_grows_again()
    test_yields_on_trigger_drop_mid_centering_lost_case_untouched()
    test_rotation_proceeds_uninterrupted_no_edge_block_veto()
    test_cumulative_budget_exhaustion_disables_future_rotation()
    test_per_engagement_bound_ends_one_engagement_not_the_whole_budget()
    test_disable_via_zero_yaw_rate_shared_with_lost_case()
    test_disable_via_zero_engage_deg_is_visible_mode_only()
    test_none_bearing_does_nothing()
    test_visible_center_never_emits_translation()
    test_sim_time_aware_dt_not_wall_clock()
    print("OK")
