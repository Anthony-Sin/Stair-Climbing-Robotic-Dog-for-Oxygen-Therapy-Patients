"""Regression tests for the post-crest "face the patient" yaw alignment (host-safe).

Task (2026-07-12, run 27 review, run_sim_20260712_125440_963): after the dog crests the
staircase and the post-crest lost-person hold (landing_lost_person_hold_active) engages, it
used to freeze BOTH translation and rotation at whatever heading it happened to be facing --
run 27 parked ~33 deg off the patient's last-known bearing. landing_face_patient_align adds a
bounded, slow, YAW-ONLY rotation to face the patient before settling forever. See its
docstring (core/control/stair_policy.py) for the full one-way terminal state machine.
"""
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.control.stair_policy import LandingFaceAlignState, landing_face_patient_align

_DEFAULTS = dict(
    deadband_deg=8.0,
    timeout_sec=6.0,
    max_rotation_deg=120.0,
    yaw_rate=0.35,
)


def _align(state, *, trigger, bearing_deg, edge_block=False, now_wall, sim_t=None, **overrides):
    kwargs = dict(_DEFAULTS)
    kwargs.update(overrides)
    return landing_face_patient_align(
        trigger=trigger,
        bearing_deg=bearing_deg,
        edge_block=edge_block,
        state=state,
        now_wall=now_wall,
        sim_t=sim_t,
        **kwargs,
    )


def test_inactive_until_trigger():
    # Never triggered -> never engages, never rotates, regardless of bearing.
    state = LandingFaceAlignState()
    r = _align(state, trigger=False, bearing_deg=-32.854, now_wall=0.0)
    assert r.engaged is False
    assert r.active is False
    assert r.done is False
    assert r.yaw_rate_cmd == 0.0


def test_requires_trigger_not_just_bearing():
    # A live bearing alone (no trigger) must never engage -- the caller's trigger already
    # encodes "post_crest_landing_latched AND fully_on_top_landing AND not person_detected"
    # (landing_lost_person_hold_active); this function must not re-derive that gate.
    state = LandingFaceAlignState()
    for t in range(5):
        r = _align(state, trigger=False, bearing_deg=20.0, now_wall=float(t) * 0.1)
        assert r.engaged is False and r.yaw_rate_cmd == 0.0


def test_engages_and_rotates_toward_negative_bearing():
    # Run 27's frozen bearing: -32.854 deg (patient on the LEFT) -> turn LEFT/CCW -> POSITIVE
    # yaw_rate_cmd, mirroring core/main.py's forward-pursuit arc-yaw convention
    # (rotation_cmd = -copysign(arc, bearing); +left/CCW).
    state = LandingFaceAlignState()
    r = _align(state, trigger=True, bearing_deg=-32.854, now_wall=100.0)
    assert r.engaged is True
    assert r.active is True
    assert r.done is False
    assert r.yaw_rate_cmd == 0.35  # +yaw_rate: turn left toward the negative bearing


def test_rotates_toward_positive_bearing_is_negative_command():
    # Patient on the RIGHT (+bearing) -> turn RIGHT -> negative yaw_rate_cmd.
    state = LandingFaceAlignState()
    r = _align(state, trigger=True, bearing_deg=25.0, now_wall=0.0)
    assert r.yaw_rate_cmd == -0.35


def test_deadband_stops_rotation_and_finishes():
    # Bearing already inside the deadband -> done immediately, no rotation ever commanded.
    state = LandingFaceAlignState()
    r = _align(state, trigger=True, bearing_deg=5.0, now_wall=0.0, deadband_deg=8.0)
    assert r.engaged is True
    assert r.active is False
    assert r.done is True
    assert r.yaw_rate_cmd == 0.0


def test_deadband_reached_mid_sequence_stops():
    # Starts outside the deadband, a later frame reports the bearing has closed inside it
    # (the caller's live/last-seen bearing can change frame to frame) -> stops there.
    state = LandingFaceAlignState()
    r1 = _align(state, trigger=True, bearing_deg=-30.0, now_wall=0.0)
    assert r1.active is True and r1.done is False
    r2 = _align(state, trigger=True, bearing_deg=-4.0, now_wall=0.5)
    assert r2.active is False
    assert r2.done is True
    assert r2.yaw_rate_cmd == 0.0


def test_timeout_stops_rotation_wall_clock():
    # No sim_t available (real-hardware-shaped) -> wall-clock elapsed against timeout_sec.
    state = LandingFaceAlignState()
    r1 = _align(state, trigger=True, bearing_deg=-90.0, now_wall=1000.0, timeout_sec=6.0)
    assert r1.active is True
    r2 = _align(state, trigger=True, bearing_deg=-90.0, now_wall=1000.0 + 6.5, timeout_sec=6.0)
    assert r2.active is False
    assert r2.done is True
    assert r2.yaw_rate_cmd == 0.0


def test_timeout_is_sim_time_aware_not_wall_clock():
    # Sim-aware clock (mirrors detection_age_sec / DetectionAgeState, incident 8.6): a huge
    # WALL gap must not itself finish the sequence if sim_t barely advanced -- and a sim_t
    # gap that crosses timeout_sec DOES finish it even with a small wall delta. This mirrors
    # the "88s wall / 15s sim" ratio documented for detection_age_sec's own incident.
    state = LandingFaceAlignState()
    r1 = _align(state, trigger=True, bearing_deg=-90.0, now_wall=0.0, sim_t=10.0, timeout_sec=6.0)
    assert r1.active is True
    # Big WALL jump, tiny SIM jump -> still active (sim says only ~1s elapsed).
    r2 = _align(state, trigger=True, bearing_deg=-90.0, now_wall=90.0, sim_t=11.0, timeout_sec=6.0)
    assert r2.active is True
    assert r2.done is False
    # Now sim_t crosses the 6s sim-time budget -> done, even though the wall delta since r2
    # is small.
    r3 = _align(state, trigger=True, bearing_deg=-90.0, now_wall=90.5, sim_t=16.5, timeout_sec=6.0)
    assert r3.done is True
    assert r3.yaw_rate_cmd == 0.0


def test_rotation_bound_stops_before_timeout():
    # Small max_rotation_deg -> the cumulative-rotation bound trips before the time budget.
    # The bound is checked once per frame against the PRIOR frame's accumulated total (a
    # discrete-time check, matching every other per-frame latch in this module), so a single
    # frame's worth of rotation (yaw_rate * dt) can land just past it before the NEXT frame
    # observes done=True -- bound the test's own step size accordingly instead of asserting
    # zero overshoot.
    state = LandingFaceAlignState()
    step_sec = 0.01
    now = 0.0
    r = None
    for _ in range(5000):
        r = _align(
            state, trigger=True, bearing_deg=-90.0, now_wall=now,
            timeout_sec=60.0, max_rotation_deg=5.0, yaw_rate=0.35,
        )
        if r.done:
            break
        now += step_sec
    assert r.done is True
    assert r.yaw_rate_cmd == 0.0
    # Bounded within one frame step's worth of overshoot (yaw_rate * step_sec).
    max_overshoot_deg = math.degrees(0.35 * step_sec)
    assert math.degrees(state.rotated_rad) <= 5.0 + max_overshoot_deg + 1e-6


def test_edge_block_vetoes_rotation_but_does_not_finish():
    # Hard constraint 3: the landing edge guard is authoritative -- hold wins over rotation.
    # Rotation is withheld THIS frame, but the sequence is not abandoned (it can resume once
    # edge_block clears), and it remains bounded by the timeout regardless.
    state = LandingFaceAlignState()
    r1 = _align(state, trigger=True, bearing_deg=-90.0, now_wall=0.0)
    assert r1.active is True
    r2 = _align(state, trigger=True, bearing_deg=-90.0, now_wall=0.1, edge_block=True)
    assert r2.engaged is True
    assert r2.active is False
    assert r2.done is False
    assert r2.yaw_rate_cmd == 0.0
    # Clears -> resumes rotating (bearing on the LEFT / negative -> +yaw_rate, per the
    # established sign convention -- see test_engages_and_rotates_toward_negative_bearing).
    r3 = _align(state, trigger=True, bearing_deg=-90.0, now_wall=0.2, edge_block=False)
    assert r3.active is True
    assert r3.yaw_rate_cmd == 0.35


def test_none_bearing_never_recorded_does_nothing():
    # Hard constraint 4: no bearing has EVER been recorded -> do nothing, ever.
    state = LandingFaceAlignState()
    r = _align(state, trigger=True, bearing_deg=None, now_wall=0.0)
    assert r.engaged is True
    assert r.active is False
    assert r.done is True
    assert r.yaw_rate_cmd == 0.0
    # Terminal: a later frame with a real bearing still does not rotate.
    r2 = _align(state, trigger=True, bearing_deg=-40.0, now_wall=1.0)
    assert r2.yaw_rate_cmd == 0.0
    assert r2.done is True


def test_explicit_disable_via_zero_yaw_rate_still_engages_hold():
    # Explicit disable path (0 == off, tested on the RAW yaw_rate argument): still latches
    # the terminal translation-hold (engaged=True) -- the safety fix -- but never rotates.
    state = LandingFaceAlignState()
    r = _align(state, trigger=True, bearing_deg=-90.0, now_wall=0.0, yaw_rate=0.0)
    assert r.engaged is True
    assert r.active is False
    assert r.done is True
    assert r.yaw_rate_cmd == 0.0


def test_disabled_yaw_rate_is_distinct_from_a_legitimate_zero_bearing():
    # CLAUDE.md zero-as-disabled-sentinel lesson: disabling (yaw_rate=0) must be tested on
    # the raw arg, not on some computed value that is ALSO legitimately 0.0 for an unrelated
    # reason (e.g. a perfectly-aligned bearing). Confirm the two paths behave identically in
    # OUTCOME (both finish with yaw_rate_cmd==0.0) but are reached through different branches
    # by checking the disabled case finishes even with a bearing FAR outside the deadband
    # (a legitimate-zero-bearing finish would only trip when bearing is small).
    disabled_state = LandingFaceAlignState()
    r_disabled = _align(
        disabled_state, trigger=True, bearing_deg=-90.0, now_wall=0.0, yaw_rate=0.0,
        deadband_deg=1.0,
    )
    assert r_disabled.done is True and r_disabled.yaw_rate_cmd == 0.0

    aligned_state = LandingFaceAlignState()
    r_aligned = _align(
        aligned_state, trigger=True, bearing_deg=0.5, now_wall=0.0, yaw_rate=0.35,
        deadband_deg=1.0,
    )
    assert r_aligned.done is True and r_aligned.yaw_rate_cmd == 0.0


def test_terminal_latch_never_reactivates_after_finishing():
    # Once done, stays done forever -- even if trigger flickers False then True again, or
    # the bearing swings back outside the deadband.
    state = LandingFaceAlignState()
    r1 = _align(state, trigger=True, bearing_deg=5.0, now_wall=0.0)  # inside deadband -> done
    assert r1.done is True
    r2 = _align(state, trigger=False, bearing_deg=-90.0, now_wall=1.0)
    assert r2.engaged is True  # one-way: engaged never un-latches
    assert r2.done is True
    assert r2.yaw_rate_cmd == 0.0
    r3 = _align(state, trigger=True, bearing_deg=-90.0, now_wall=2.0)
    assert r3.done is True
    assert r3.yaw_rate_cmd == 0.0


def test_engaged_latches_even_when_trigger_drops_mid_rotation():
    # Hard constraint 1: re-acquiring the person (trigger -> False) mid-turn must not release
    # engagement -- engaged stays True and rotation continues toward the last-good bearing
    # value the caller keeps passing (the caller is expected to keep passing a live bearing
    # while person_detected; this function only asserts it never DISENGAGES).
    state = LandingFaceAlignState()
    r1 = _align(state, trigger=True, bearing_deg=-40.0, now_wall=0.0)
    assert r1.engaged is True and r1.active is True
    r2 = _align(state, trigger=False, bearing_deg=-10.0, now_wall=0.1)
    assert r2.engaged is True
    assert r2.active is True  # still outside the 8 deg deadband -> keeps rotating
    assert r2.yaw_rate_cmd == 0.35


def test_yaw_rate_cmd_never_drives_translation():
    # Sanity: the function has no vx/vy concept at all -- it only ever returns a yaw rate.
    # (Documents the contract the caller relies on: trans_x_cmd is zeroed by the CALLER,
    # independent of anything this function returns.)
    state = LandingFaceAlignState()
    r = _align(state, trigger=True, bearing_deg=-90.0, now_wall=0.0)
    assert not hasattr(r, "trans_x_cmd")
    assert not hasattr(r, "vx")


if __name__ == "__main__":
    test_inactive_until_trigger()
    test_requires_trigger_not_just_bearing()
    test_engages_and_rotates_toward_negative_bearing()
    test_rotates_toward_positive_bearing_is_negative_command()
    test_deadband_stops_rotation_and_finishes()
    test_deadband_reached_mid_sequence_stops()
    test_timeout_stops_rotation_wall_clock()
    test_timeout_is_sim_time_aware_not_wall_clock()
    test_rotation_bound_stops_before_timeout()
    test_edge_block_vetoes_rotation_but_does_not_finish()
    test_none_bearing_never_recorded_does_nothing()
    test_explicit_disable_via_zero_yaw_rate_still_engages_hold()
    test_disabled_yaw_rate_is_distinct_from_a_legitimate_zero_bearing()
    test_terminal_latch_never_reactivates_after_finishing()
    test_engaged_latches_even_when_trigger_drops_mid_rotation()
    test_yaw_rate_cmd_never_drives_translation()
    print("OK")
