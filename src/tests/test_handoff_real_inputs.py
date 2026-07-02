"""Regression tests for the real-robot handoff-input fixes (host-safe, no Isaac).

The sim feeds the handoff FSM ground-truth inputs (base_z, body velocity, an oriented
yaw) that the real port cannot supply, and defaulting them to 0.0/None silently inverted
safety guards into hazards. These tests pin the fixes: an absent input DISABLES its guard
explicitly (never false-fires), and the stair-commit holds the heading captured AT commit
time rather than steering to absolute yaw 0 (== the robot's power-on pose on hardware).

Sim behaviour is unchanged (sim always passes real floats) -- covered by
test_pgtt_stair_handoff.py, which must stay green alongside this file.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling test helpers

from go2_locomotion.pgtt_stair_handoff import (  # noqa: E402
    StallDetector, HandoffController, HandoffConfig,
)
from test_pgtt_stair_handoff import synth_staircase_depth, _FakePgtt  # noqa: E402


def test_base_z_none_disables_climb_watchdog():
    """With no vertical odometry (base_z=None) the progress watchdog must be DISABLED,
    not force-abort every climb after climb_stall_timeout_sec (which drops onto PGTT
    mid-staircase -> flip). Contrast test_climb_progress_watchdog, which uses real base_z.
    """
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_arm_after_secs=0.0, climb_backend="blind_rl",
                        climb_max_sec=999.0, climb_stall_timeout_sec=1.0, climb_progress_min_m=0.05,
                        top_clear_debounce_sec=0.1)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=D, stairs_action_active=True, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22,
                riser_dist_ahead=0.50, stairs_ahead_gt=True, base_x=0.0, person_gap_m=5.0)
    ho.update(now=1.0, dt=0.05, base_z=None, **base)
    assert ho.state == "climb", "should engage via approach room even without base_z"
    t = 1.0
    r = None
    for _ in range(60):                 # 3.0 s -- well past the 1.0 s watchdog window
        t += 0.05
        r = ho.update(now=t, dt=0.05, base_z=None, **base)
    assert r["state"] == "climb", f"base_z=None must DISABLE the watchdog, not hand back: {r}"


def test_body_fwd_none_never_stalls():
    """No velocity odometry (body_fwd=None) must never read as a stall (coercing to 0.0
    would fire a false climb hand-off on a robot that is walking fine)."""
    sd = StallDetector(HandoffConfig(stall_consec_sec=0.6, stall_cmd_min_mps=0.05,
                                     stall_min_progress_m=0.04))
    dt = 0.02
    fired = any(sd.update(dt, cmd_vx=0.16, body_fwd=None) for _ in range(60))  # 1.2 s commanded
    assert not fired, "absent velocity must NOT register as a stall"


def test_no_false_climb_engage_without_velocity_or_riser():
    """Real-robot absence (no velocity AND no riser distance) must not phantom-engage the
    climb -- neither the stall path (velocity) nor the approach path (riser) is available."""
    cfg = HandoffConfig(stall_consec_sec=0.3, climb_attempt=True, climb_backend="blind_rl",
                        require_controller_stairs=False, stair_commit_arm_after_secs=0.0,
                        stair_commit_enabled=True)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    r = None
    for i in range(30):
        r = ho.update(now=1.0 + i * 0.05, dt=0.05, go2=object(), depth_hw=D,
                      cmd_vx=0.16, stairs_action_active=True, base_z=None,
                      body_speed=None, body_fwd=None,
                      roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0,
                      height_above_step=None, person_detected=False,
                      yaw=0.0, y_lateral=0.0, riser_dist_ahead=None)
    assert r["state"] == "walk" and not r["stalled"], f"absent inputs must not false-engage: {r}"


def test_commit_holds_commit_heading_not_absolute_zero():
    """The stair-commit heading-lock holds the heading captured AT commit, not absolute 0."""
    cfg = HandoffConfig(stair_commit_enabled=True, stair_commit_yaw_kp=2.0,
                        stair_commit_lat_kp=1.2, stair_commit_wz_max=0.6,
                        stair_commit_vx_floor=0.22, stair_commit_max_sec=25.0,
                        stair_commit_arm_after_secs=0.0, require_controller_stairs=False)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    common = dict(go2=object(), depth_hw=synth_staircase_depth(), stairs_action_active=False,
                  base_z=0.30, roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0,
                  height_above_step=0.30, y_lateral=0.0, riser_dist_ahead=0.55,
                  body_speed=0.0, body_fwd=0.0, cmd_vx=0.0)
    # Commit begins at yaw=0.5 rad -> that heading is captured.
    r0 = ho.update(now=1.0, dt=0.05, yaw=0.5, person_detected=False, **common)
    assert r0["committing"], r0
    # Still at the commit heading -> ~zero correction (NOT steering toward absolute 0, which
    # with the old absolute-yaw lock would have been -kp*0.5 clipped to -wz_max).
    r1 = ho.update(now=1.05, dt=0.05, yaw=0.5, person_detected=False, **common)
    assert abs(r1["wz_override"]) < 1e-6, f"should hold the commit heading, not steer to 0: {r1}"
    # Drifted +0.3 rad off the commit heading -> corrects back (negative wz).
    r2 = ho.update(now=1.10, dt=0.05, yaw=0.8, person_detected=False, **common)
    assert r2["wz_override"] < 0.0, f"should correct back toward the commit heading: {r2}"
    # Person re-detected -> commit releases (captured heading cleared).
    r3 = ho.update(now=1.15, dt=0.05, yaw=0.8, person_detected=True, **common)
    assert not r3["committing"], r3


if __name__ == "__main__":
    test_base_z_none_disables_climb_watchdog()
    test_body_fwd_none_never_stalls()
    test_no_false_climb_engage_without_velocity_or_riser()
    test_commit_holds_commit_heading_not_absolute_zero()
    print("ALL REAL-INPUT HANDOFF TESTS PASS")
