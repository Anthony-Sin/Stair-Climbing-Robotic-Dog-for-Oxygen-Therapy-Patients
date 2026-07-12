"""Offline (no Isaac) tests for the dual-policy PGTT<->stair-climb handoff.

Validates the two perception/decision pieces of the Task-2 handoff host-side:
  * DepthStairDetector counts >=2 risers from a synthetic head-on staircase depth
    image (perpendicular distance_to_image_plane, as Isaac reports it) and reports a
    plausible leading-edge distance, while flat ground shows 0 stairs.
  * StallDetector fires only when the walker is commanded forward yet not moving
    (the divergence gate), and never on a deliberate hold or while walking normally.

The closed-loop climber's kinematics have their own self-test
(closed_loop_stair_climber.py __main__); this covers the new glue.

Run: python tests/test_pgtt_stair_handoff.py
"""

import math
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))
sys.path.insert(0, REPO)  # go2_locomotion package lives at the repo root

from go2_locomotion.pgtt_stair_handoff import (  # noqa: E402
    StallDetector, DepthStairDetector, HandoffController, HandoffConfig, GO2_LEG_CLEARANCE_M,
    stair_engage_person_ghost_veto, stair_entry_lead_ok,
)


def synth_staircase_depth(H=60, W=106, cam_h=0.40, vfov_deg=56.5, pitch_deg=0.5,
                          x_base=0.45, rise=0.15, run=0.305, n_steps=14, max_range=3.0):
    """Forward ray-cast a head-on staircase into a (H,W) perpendicular-depth image (m).

    Mirrors the parkour depth cam (106x60, ~56.5 deg vFOV, 0.40 m high, ~0.5 deg down).
    Returns distance_to_image_plane (perpendicular) so the detector's inverse projection
    is exercised exactly as it is against real Isaac depth.
    """
    cy = (H - 1) / 2.0
    vfov = math.radians(vfov_deg)
    pitch = math.radians(pitch_deg)

    def solid_top(px):
        if px < x_base:
            return 0.0
        k = min(int((px - x_base) // run) + 1, n_steps)
        return k * rise

    D = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        theta_v = ((r - cy) / float(H)) * vfov
        ang = pitch + theta_v  # downward angle from horizontal
        if ang <= 1e-3:
            D[r, :] = 0.0       # ray points up -> "sky", invalid
            continue
        t = 0.0
        hit = None
        while t < max_range:
            px = t * math.cos(ang)
            pz = cam_h - t * math.sin(ang)
            if pz <= solid_top(px):
                hit = t
                break
            t += 0.004
        D[r, :] = (hit * math.cos(theta_v)) if hit is not None else 0.0
    return D


def synth_flat_depth(H=60, W=106, cam_h=0.40, vfov_deg=56.5, pitch_deg=0.5):
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


def test_detector():
    det = DepthStairDetector(HandoffConfig())
    out = det.detect(synth_staircase_depth())
    assert out["stair_detected"], f"should detect a staircase: {out}"
    assert out["stair_count"] >= 2, f"expected >=2 risers, got {out}"
    assert out["leading_edge_distance"] is not None and out["leading_edge_distance"] < 1.5
    out_flat = det.detect(synth_flat_depth())
    assert out_flat["stair_count"] < 2, f"flat ground must not show >=2 stairs: {out_flat}"
    print(f"detector OK  staircase={out}  flat={out_flat}  (leg_clearance={GO2_LEG_CLEARANCE_M} m)")


def test_stall():
    cfg = HandoffConfig(stall_consec_sec=0.6, stall_speed_mps=0.06,
                        stall_cmd_min_mps=0.05, stall_divergence_mps=0.12)
    sd = StallDetector(cfg)
    dt = 0.02
    fired_at = None
    for i in range(60):
        if sd.update(dt, cmd_vx=0.16, body_fwd=0.01) and fired_at is None:
            fired_at = (i + 1) * dt
    assert fired_at is not None and 0.55 <= fired_at <= 0.7, f"stall fired at {fired_at}"
    sd.reset()
    assert not any(sd.update(dt, cmd_vx=0.0, body_fwd=0.0) for _ in range(60)), \
        "a commanded hold must not register as a stall"
    sd.reset()
    assert not any(sd.update(dt, cmd_vx=0.6, body_fwd=0.55) for _ in range(60)), \
        "normal walking must not register as a stall"
    print(f"stall OK  (fired at {fired_at:.2f}s; hold/walk correctly ignored)")


class _FakePgtt:
    """Minimal PgttLocomotionPolicy stand-in for the FSM test."""
    def __init__(self):
        self.applied = []
    def current_act_positions(self, go2):
        return np.zeros(12, dtype=np.float32)
    def apply_external_act_targets(self, go2, t):
        self.applied.append(np.asarray(t, dtype=np.float32))


def test_fsm():
    cfg = HandoffConfig(stall_consec_sec=0.3, climb_riser_height_m=0.15,
                        climb_max_sec=5.0, re_eval_cooldown_sec=1.5,
                        require_controller_stairs=True, climb_attempt=True,
                        climb_backend="ik")
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    go2 = object()
    dt, now, base_z = 0.05, 0.0, 0.30

    # Phase 1: walking normally toward the stairs (moving) -> stays WALK (no stall).
    for _ in range(10):
        now += dt
        r = ho.update(now=now, dt=dt, go2=go2, depth_hw=D, cmd_vx=0.5,
                      stairs_action_active=True, base_z=base_z, body_speed=0.5,
                      roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0,
                      height_above_step=0.30, foot_contacts=None)
        assert not r["climb"] and r["state"] == "walk", f"should not climb while moving: {r}"

    # Phase 2: stalled at the stairs (commanded forward, body stuck) -> engage CLIMB.
    engaged = None
    for _ in range(20):
        now += dt
        r = ho.update(now=now, dt=dt, go2=go2, depth_hw=D, cmd_vx=0.16,
                      stairs_action_active=True, base_z=base_z, body_speed=0.0,
                      roll=0.0, pitch=0.1, roll_rate=0.0, pitch_rate=0.0,
                      height_above_step=0.30, foot_contacts=None)
        if r["state"] == "climb":
            engaged = r
            break
    assert engaged is not None, "should engage CLIMB when stalled in front of >=2 stairs"
    assert engaged["climb"] and engaged["targets_act"] is not None
    assert len(np.asarray(engaged["targets_act"]).reshape(-1)) == 12

    # Phase 3: body rose one riser -> hand back to WALK.
    handed_back = False
    for _ in range(6):
        now += dt
        r = ho.update(now=now, dt=dt, go2=go2, depth_hw=D, cmd_vx=0.16,
                      stairs_action_active=True, base_z=base_z + 0.16, body_speed=0.0,
                      roll=0.0, pitch=0.1, roll_rate=0.0, pitch_rate=0.0,
                      height_above_step=0.30, foot_contacts=None)
        if r["state"] == "walk":
            handed_back = True
            break
    assert handed_back, "should hand back to WALK after climbing one riser"
    print("fsm OK  (walk->climb on stall@>=2 stairs, climb->walk after one riser)")


def test_stair_commit():
    cfg = HandoffConfig(stair_commit_enabled=True, stair_commit_yaw_kp=2.0,
                        stair_commit_lat_kp=1.2, stair_commit_wz_max=0.6,
                        stair_commit_vx_floor=0.22, stair_commit_max_sec=25.0,
                        stair_commit_arm_after_secs=0.0,  # disable arm delay for unit test
                        require_controller_stairs=False)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    go2 = object()
    # riser_dist_ahead=0.55 simulates GT terrain confirming a real step riser is ahead
    # (our terrain-gate requires this OR state=="climb" to prevent false commits on a
    # person's body depth at the same range).
    common = dict(go2=go2, depth_hw=D, stairs_action_active=False, base_z=0.30,
                  roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0,
                  height_above_step=0.30, yaw=0.3, y_lateral=1.0, riser_dist_ahead=0.55)

    # Person VISIBLE near the stairs -> no commit (the normal follow loop steers).
    r = ho.update(now=1.0, dt=0.05, cmd_vx=0.3, body_speed=0.3, body_fwd=0.3,
                  person_detected=True, **common)
    assert not r["committing"] and r["wz_override"] is None, r

    # Person LOST with stairs ahead AND GT terrain confirms riser (riser_dist_ahead=0.55),
    # robot drifted off-axis (yaw=+0.3, y=+1.0) -> commit: steer back toward yaw=0 / y=0.
    r = ho.update(now=2.0, dt=0.05, cmd_vx=0.0, body_speed=0.0, body_fwd=0.0,
                  person_detected=False, **common)
    assert r["committing"], r
    assert r["wz_override"] is not None and r["wz_override"] < 0.0, r
    assert abs(r["vx_floor"] - 0.22) < 1e-6, r

    # Person RE-detected -> commit releases, normal follow resumes.
    r = ho.update(now=2.5, dt=0.05, cmd_vx=0.3, body_speed=0.3, body_fwd=0.3,
                  person_detected=True, **common)
    assert not r["committing"] and r["wz_override"] is None, r
    print("stair_commit OK  (lost+stairs -> heading-hold up + forward floor; re-detect -> release)")


def test_approach_engage():
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_max_sec=25.0, stair_commit_arm_after_secs=0.0)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=D, stairs_action_active=True, base_z=0.30, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22)
    # Riser JAMMED (0.30 < min_room 0.40) -> do NOT engage (no room -> would flip).
    r = ho.update(now=1.0, dt=0.05, riser_dist_ahead=0.30, **base)
    assert r["state"] == "walk", f"jammed must not approach-engage: {r}"
    # Riser at a STANDOFF (0.50 in [0.40, 0.65]) + committing -> engage with room.
    # Default backend is "parkour" -> use_parkour True, no IK targets (isaac_env runs it).
    r = ho.update(now=1.1, dt=0.05, riser_dist_ahead=0.50, **base)
    assert r["state"] == "climb" and r["climb"], f"standoff should engage with room: {r}"
    assert r["use_parkour"] and r["targets_act"] is None, f"parkour backend hot-swap: {r}"
    print("approach_engage OK  (jammed skipped; standoff engages with room; parkour hot-swap)")


def test_ghost_engage_veto_pure_function():
    """stair_engage_person_ghost_veto in isolation: sized from run_sim_20260712_013638_835's
    ghost engage (leading_edge_m=0.648, GT patient gap stable at 1.08-1.11 m across the whole
    +/-2s bracket around the engage -> diff ~0.448 m) vs its real engage (leading_edge_m=0.451,
    GT patient gap ~1.7 m -> diff ~1.25 m)."""
    # Ghost: within the 0.5 m window -> veto.
    assert stair_engage_person_ghost_veto(
        leading_edge_distance_m=0.648, person_gap_m=1.096, window_m=0.5) is True
    # Real engage: far outside even a generous window -> never vetoed.
    assert stair_engage_person_ghost_veto(
        leading_edge_distance_m=0.451, person_gap_m=1.7, window_m=0.5) is False
    # Either input missing (real hardware: no GT person_gap_m; or no leading-edge reading
    # this frame) -> cannot compare -> no veto (fails toward NOT vetoing a real engage on
    # missing data, rather than guessing).
    assert stair_engage_person_ghost_veto(
        leading_edge_distance_m=0.648, person_gap_m=None, window_m=0.5) is False
    assert stair_engage_person_ghost_veto(
        leading_edge_distance_m=None, person_gap_m=1.096, window_m=0.5) is False


def test_ghost_engage_veto_blocks_person_as_risers_stall_engage():
    """Integration: run_sim_20260712_013638_835's exact ghost engage signature
    (engage_reason="wedge_stall", riser_dist_ahead_m=None, leading_edge_m=0.648,
    level_heights_m starting with a non-zero first riser -- a leg, not ground) at a GT
    patient distance of 1.096 m must NOT engage once ghost_engage_gap_window_m is active,
    even though every other stall_engage condition (has_stairs, stalled, near_enough) is
    satisfied -- exactly as it was in the un-fixed run."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_arm_after_secs=0.0, ghost_engage_gap_window_m=0.5)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    # Force the detector output to the run's exact ghost reading -- avoids depending on the
    # synthetic depth-image geometry to land at a specific leading_edge_distance.
    ho.detector.detect = lambda depth_hw: {
        "stair_detected": True, "stair_count": 11, "leading_edge_distance": 0.648,
        "level_heights_m": [0.0, 0.232, 0.32, 0.371],
    }
    base = dict(go2=object(), depth_hw=object(), stairs_action_active=True, base_z=2.30,
                body_speed=0.02, roll=0.0, pitch=-0.06, roll_rate=0.0, pitch_rate=0.0,
                height_above_step=0.30, person_detected=True, yaw=-0.23, y_lateral=0.13,
                body_fwd=0.02, cmd_vx=0.20)
    r = None
    for i in range(20):  # commanded-but-not-moving long enough to satisfy stall_consec_sec
        r = ho.update(now=1.0 + i * 0.1, dt=0.1, riser_dist_ahead=None,
                      base_x=6.709, person_gap_m=1.096, **base)
    assert r["state"] == "walk" and not r["climb"], \
        f"person-as-risers ghost must be vetoed, not engaged: {r}"


def test_ghost_engage_veto_does_not_block_real_engage_with_far_patient():
    """Same standoff-engage scenario as test_approach_engage, but now WITH a GT
    person_gap_m far from the riser (5.0 m, mirroring the run's real engage where the
    patient was ~1.7 m ahead while leading_edge_m=0.451) -- the veto must not interfere."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_max_sec=25.0, stair_commit_arm_after_secs=0.0,
                        ghost_engage_gap_window_m=0.5)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=D, stairs_action_active=True, base_z=0.30, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22)
    ho.update(now=1.0, dt=0.05, riser_dist_ahead=0.30, person_gap_m=5.0, **base)
    r = ho.update(now=1.1, dt=0.05, riser_dist_ahead=0.50, person_gap_m=5.0, **base)
    assert r["state"] == "climb" and r["climb"], f"far patient must not veto a real engage: {r}"


def test_stair_entry_lead_ok_pure_function():
    """stair_entry_lead_ok in isolation: sized from run_sim_20260712_023126_786 (run 14),
    whose "wedge_stall" engage fired at a GT planar gap of just 0.765 m (well under the
    2.4 m default) and closed to a 0.452 m collision eight climb-steps later."""
    # Below the required lead -> not OK (gate holds the dog at the base).
    assert stair_entry_lead_ok(patient_lead_m=0.765, min_lead_m=2.4) is False
    assert stair_entry_lead_ok(patient_lead_m=0.0, min_lead_m=2.4) is False
    # At/above the required lead -> OK.
    assert stair_entry_lead_ok(patient_lead_m=2.4, min_lead_m=2.4) is True
    assert stair_entry_lead_ok(patient_lead_m=3.0, min_lead_m=2.4) is True
    # No GT (real hardware, or no patient sidecar) -> cannot compare -> gate is a no-op
    # (mirrors stair_engage_person_ghost_veto's None contract).
    assert stair_entry_lead_ok(patient_lead_m=None, min_lead_m=2.4) is True


def test_stair_entry_gate_holds_a_close_approach_engage_and_releases_once_ahead():
    """Same standoff-engage scenario as test_approach_engage (riser at a standoff + room +
    committing -> would otherwise engage), but with the patient's GT lead below
    stair_entry_min_lead_m: ENGAGE must be held (state stays "walk"). Once the SAME frame's
    lead crosses the threshold, the (still-valid) engage condition fires immediately."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_max_sec=25.0, stair_commit_arm_after_secs=0.0,
                        stair_entry_min_lead_m=2.4)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=D, stairs_action_active=True, base_z=0.30, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22)
    # Standoff + committing (as test_approach_engage), but the patient's lead is only 0.765 m
    # (run 14's exact engage-time number) -- must be HELD, not engaged.
    r = ho.update(now=1.0, dt=0.05, riser_dist_ahead=0.50, patient_lead_m=0.765, **base)
    assert r["state"] == "walk", f"insufficient lead must hold at the base, not engage: {r}"
    r = ho.update(now=1.05, dt=0.05, riser_dist_ahead=0.50, patient_lead_m=1.5, **base)
    assert r["state"] == "walk", f"still below the 2.4 m threshold -> still held: {r}"
    # Lead now clears the threshold -> the still-valid engage condition fires.
    r = ho.update(now=1.10, dt=0.05, riser_dist_ahead=0.50, patient_lead_m=2.4, **base)
    assert r["state"] == "climb" and r["climb"], f"lead >= threshold must release the gate: {r}"


def test_stair_entry_gate_no_gt_lead_does_not_block_real_engage():
    """patient_lead_m=None (no GT -- real hardware, or waypoint test with no patient) must
    not interfere with an otherwise-valid engage, mirroring the ghost veto's None contract."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_max_sec=25.0, stair_commit_arm_after_secs=0.0,
                        stair_entry_min_lead_m=2.4)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=D, stairs_action_active=True, base_z=0.30, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22)
    ho.update(now=1.0, dt=0.05, riser_dist_ahead=0.30, patient_lead_m=None, **base)
    r = ho.update(now=1.1, dt=0.05, riser_dist_ahead=0.50, patient_lead_m=None, **base)
    assert r["state"] == "climb" and r["climb"], f"no GT lead must not block a real engage: {r}"


def test_stair_entry_gate_never_clamps_an_ongoing_climb():
    """Incident 8.15: the gate only guards the walk->climb TRANSITION. Once engaged, a
    patient_lead_m that later drops below the threshold (she fell behind mid-climb) must
    never abort or hold the ongoing climb."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_arm_after_secs=0.0, climb_backend="blind_rl",
                        climb_max_sec=999.0, climb_stall_timeout_sec=999.0,
                        stair_entry_min_lead_m=2.4)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    Dstairs = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=Dstairs, stairs_action_active=True, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22,
                riser_dist_ahead=0.50, stairs_ahead_gt=True, base_x=0.0)
    ho.update(now=1.0, dt=0.05, base_z=0.30, patient_lead_m=3.0, **base)
    assert ho.state == "climb", "should engage with a sufficient lead"
    # Lead now reads BELOW the entry threshold mid-climb -- must not touch the ongoing climb.
    r = ho.update(now=1.05, dt=0.05, base_z=0.32, patient_lead_m=0.5, **base)
    assert r["state"] == "climb" and r["climb"], \
        f"an ONGOING climb must not be held/aborted by the S1 entry gate: {r}"


def test_climb_elapsed_sec_resets_at_engage_and_dt_accumulates():
    """climb_elapsed_sec (task, 2026-07-12, run 32 review) is exposed in update()'s returned
    dict as the SIM-TIME (dt-accumulated, incident 8.6) source the blind-mount step-down
    (go2_locomotion.locomotion_arbiter.blind_mount_climb_vx_floor) reads. It reuses the
    EXISTING vertical-progress-watchdog accumulator (self._climb_elapsed): reset to 0.0 the
    frame ENGAGE fires, then += dt every frame while state=="climb" -- including the ENGAGE
    frame itself, since the "walk" block's state flip is not an elif away from the "climb"
    block below it in the same update() call."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_arm_after_secs=0.0, climb_backend="blind_rl",
                        climb_max_sec=999.0, climb_stall_timeout_sec=999.0,
                        stair_entry_min_lead_m=0.0)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    Dstairs = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=Dstairs, stairs_action_active=True, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22,
                riser_dist_ahead=0.50, stairs_ahead_gt=True, base_x=0.0)
    # Not climbing yet: the key is still present (default 0.0), never a KeyError.
    r0 = ho.update(now=0.9, dt=0.05, base_z=0.30, patient_lead_m=5.0,
                   **{**base, "riser_dist_ahead": 2.0})  # too far ahead to engage this frame
    assert r0["state"] == "walk"
    assert abs(r0["climb_elapsed_sec"] - 0.0) < 1e-9, r0["climb_elapsed_sec"]
    # ENGAGE this frame -- climb_elapsed_sec is dt-accumulated from 0.0 on the SAME frame
    # (the climb block runs immediately after the walk block sets state="climb").
    r1 = ho.update(now=1.0, dt=0.05, base_z=0.30, patient_lead_m=5.0, **base)
    assert r1["state"] == "climb" and r1["climb"]
    assert abs(r1["climb_elapsed_sec"] - 0.05) < 1e-9, r1["climb_elapsed_sec"]
    # Next frame: accumulates by another dt.
    r2 = ho.update(now=1.05, dt=0.05, base_z=0.32, patient_lead_m=5.0, **base)
    assert abs(r2["climb_elapsed_sec"] - 0.10) < 1e-9, r2["climb_elapsed_sec"]
    r3 = ho.update(now=1.10, dt=0.07, base_z=0.34, patient_lead_m=5.0, **base)
    assert abs(r3["climb_elapsed_sec"] - 0.17) < 1e-9, r3["climb_elapsed_sec"]


def test_stair_entry_gate_interval_is_non_empty_against_isaac_env_hard_wait():
    """Static source-scan (same approach as test_taper_has_exactly_one_call_site_in_main_
    gated_on_post_crest_latch -- isaac_env.py imports isaacsim and cannot be imported on a
    plain host). Reads PATIENT_HARD_WAIT_LEAD_M's literal value out of isaac_env.py's source
    and asserts [HandoffConfig().stair_entry_min_lead_m, PATIENT_HARD_WAIT_LEAD_M) is
    non-empty -- the non-deadlock interval both docstrings (HandoffConfig.
    stair_entry_min_lead_m, isaac_env.py's PATIENT_HARD_WAIT_LEAD_M comment) derive."""
    isaac_env_py = os.path.join(REPO, "sim", "isaac", "isaac_env.py")
    with open(isaac_env_py, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"^PATIENT_HARD_WAIT_LEAD_M\s*=\s*([0-9.]+)", src, re.MULTILINE)
    assert m is not None, "PATIENT_HARD_WAIT_LEAD_M not found in isaac_env.py"
    hard_wait_m = float(m.group(1))
    gate_m = float(HandoffConfig().stair_entry_min_lead_m)
    assert gate_m < hard_wait_m, (
        f"stair_entry_min_lead_m ({gate_m}) must be strictly below isaac_env.py's "
        f"PATIENT_HARD_WAIT_LEAD_M ({hard_wait_m}), or the dog's entry gate and the "
        "patient's hard-wait could both hold simultaneously (deadlock)."
    )


def test_top_egress():
    """Top-of-stairs egress: crest -> walk off the last step -> hand back to PGTT, person-gated."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_max_sec=25.0, stair_commit_arm_after_secs=0.0,
                        climb_backend="blind_rl",
                        top_egress_enabled=True, top_clear_debounce_sec=0.1,
                        top_egress_distance_m=0.20, top_egress_vx=0.22,
                        top_egress_standoff_m=0.60, top_egress_max_sec=4.0, climb_max_sec=60.0)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    Dstairs, Dflat = synth_staircase_depth(), synth_flat_depth()
    base = dict(go2=object(), stairs_action_active=True, base_z=0.30, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22)

    # Engage the blind_rl climb at a standoff (committing + room), stairs still ahead.
    ho.update(now=1.0, dt=0.05, depth_hw=Dstairs, riser_dist_ahead=0.50,
              stairs_ahead_gt=True, base_x=0.0, person_gap_m=5.0, **base)
    r = ho.update(now=1.05, dt=0.05, depth_hw=Dstairs, riser_dist_ahead=0.50,
                  stairs_ahead_gt=True, base_x=0.0, person_gap_m=5.0, **base)
    assert r["state"] == "climb" and r["use_parkour"], r
    assert not r["top_egress"], f"stairs still ahead -> not egress yet: {r}"

    # Reach the top: flat depth AND ground-truth clear. Debounce, then enter egress with a
    # forward floor (patient far -> push the rear feet off the crest), still under the climb policy.
    t, entered = 1.1, None
    for _ in range(10):
        t += 0.05
        r = ho.update(now=t, dt=0.05, depth_hw=Dflat, riser_dist_ahead=None,
                      stairs_ahead_gt=False, base_x=0.0, person_gap_m=5.0, **base)
        if r["top_egress"]:
            entered = r
            break
    assert entered is not None and entered["state"] == "climb", "should egress at the crest, still climbing"
    assert abs(entered["climb_vx_floor"] - 0.22) < 1e-6, f"patient far -> egress pushes forward: {entered}"

    # Patient now CLOSE on the landing -> floor drops to 0 (hold in place, never collide).
    r = ho.update(now=t + 0.05, dt=0.05, depth_hw=Dflat, riser_dist_ahead=None,
                  stairs_ahead_gt=False, base_x=0.0, person_gap_m=0.4, **base)
    assert r["top_egress"] and r["climb_vx_floor"] == 0.0, f"close patient -> hold: {r}"

    # Walk the rear feet off the crest (base_x advances past top_egress_distance_m) -> hand back.
    handed, bx = False, 0.0
    for _ in range(12):
        t += 0.05
        bx += 0.05
        r = ho.update(now=t, dt=0.05, depth_hw=Dflat, riser_dist_ahead=None,
                      stairs_ahead_gt=False, base_x=bx, person_gap_m=5.0, **base)
        if r["state"] == "walk":
            handed = True
            break
    assert handed, "should hand back to PGTT after the egress travel completes"
    print("top_egress OK  (crest -> egress push; close patient holds at 0; hands back after egress)")


def test_egress_stops_at_goal():
    """Egress must stop AT a forward goal (waypoint) and not overrun it (no person)."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_arm_after_secs=0.0,
                        climb_backend="blind_rl", climb_max_sec=999.0, climb_stall_timeout_sec=999.0,
                        top_egress_enabled=True, top_clear_debounce_sec=0.1,
                        top_egress_distance_m=0.50, top_egress_goal_stop_m=0.12, climb_progress_min_m=0.05)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    Dstairs, Dflat = synth_staircase_depth(), synth_flat_depth()
    base = dict(go2=object(), stairs_action_active=True, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22,
                person_gap_m=9.0)  # person far / off-lane (waypoint test)
    z = 0.30
    ho.update(now=1.0, dt=0.05, depth_hw=Dstairs, riser_dist_ahead=0.50, stairs_ahead_gt=True,
              base_x=6.0, base_z=z, **base)
    # Crest at x=6.0; waypoint at 6.3. Egress pushes (floor>0) while far from the goal, then ENDS
    # AT the goal (within goal_stop 0.12 m) -- before the 0.5 m egress distance -- without overrun.
    t, pushed_far, handed = 1.0, False, False
    bx = 6.0
    for _ in range(20):
        t += 0.05; bx += 0.02; z += 0.001
        goal = 6.3 - bx
        r = ho.update(now=t, dt=0.05, depth_hw=Dflat, riser_dist_ahead=None, stairs_ahead_gt=False,
                      base_x=bx, base_z=z, forward_goal_dist_m=goal, **base)
        if r.get("top_egress") and goal > 0.12 and (r["climb_vx_floor"] or 0.0) > 0.0:
            pushed_far = True            # floor IS emitted while still far from the goal
        if r["state"] == "walk":
            handed = True
            assert goal <= 0.12 + 1e-6, f"must hand back AT the goal, not before (goal={goal:.3f})"
            assert bx <= 6.30 + 1e-6, f"must not overrun the waypoint (x={bx:.3f} > 6.30)"
            break
    assert pushed_far, "egress must push forward (floor>0) while far from the goal"
    assert handed, "egress should end (hand back) at the waypoint"
    print("egress_stops_at_goal OK  (pushes while far; ends AT the waypoint; no overrun)")


def test_no_false_crest_between_risers():
    """A transient flat depth profile mid-climb (GT still rising) must NOT declare the top."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_arm_after_secs=0.0,
                        climb_backend="blind_rl", top_clear_debounce_sec=0.1,
                        top_egress_distance_m=0.20, climb_max_sec=60.0)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    Dstairs, Dflat = synth_staircase_depth(), synth_flat_depth()
    base = dict(go2=object(), stairs_action_active=True, base_z=0.30, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22)
    ho.update(now=1.0, dt=0.05, depth_hw=Dstairs, riser_dist_ahead=0.50,
              stairs_ahead_gt=True, base_x=0.0, person_gap_m=5.0, **base)
    # Flat DEPTH (det clear) but the ground truth still says a riser rises ahead -> NOT cleared.
    t = 1.05
    for _ in range(10):
        t += 0.05
        r = ho.update(now=t, dt=0.05, depth_hw=Dflat, riser_dist_ahead=0.40,
                      stairs_ahead_gt=True, base_x=0.0, person_gap_m=5.0, **base)
        assert not r["top_egress"], f"GT riser still ahead -> must not crest: {r}"
    assert r["state"] == "climb", "still climbing between risers"
    print("no_false_crest OK  (flat depth but GT rising -> stays in climb, no egress)")


def test_climb_progress_watchdog():
    """A still-RISING climb is not cut off; a wedged (no-height-gain) climb MID-STAIRCASE now
    HOLDS the climber instead of handing back to the flat-ground PGTT walker.

    SAFETY (incident 8.8): handing the legs to PGTT on a 26 deg incline topples the dog --
    run_sim_20260703_193958 handed back on a mid-stair stall at z=1.607 m (still upright, 14 deg
    tilt, ~step 9 of 14) and flipped 95 s later. So on the incline (`stairs_clear` False) the stall
    watchdog must still FIRE (retry heartbeat increments) but NEVER surrender the incline; the only
    climb->walk handback is at the crest (egress), covered by the egress/crest tests."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_arm_after_secs=0.0,
                        climb_backend="blind_rl", climb_max_sec=999.0,
                        climb_stall_timeout_sec=1.0, climb_progress_min_m=0.05,
                        top_clear_debounce_sec=0.1)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    Dstairs = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=Dstairs, stairs_action_active=True, body_speed=0.2,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22,
                riser_dist_ahead=0.50, stairs_ahead_gt=True, base_x=0.0, person_gap_m=5.0)
    # Engage, then climb while the body keeps RISING -> never hands back on the watchdog.
    z = 0.30
    ho.update(now=1.0, dt=0.05, base_z=z, **base)
    t = 1.0
    for _ in range(40):                       # 2.0 s with steady height gain (> stall window)
        t += 0.05; z += 0.02                  # +0.4 m/s vertical -> always "progressing"
        r = ho.update(now=t, dt=0.05, base_z=z, **base)
    assert r["state"] == "climb", f"a still-rising climb must NOT hand back: {r}"
    # Now WEDGE it MID-STAIRCASE (stairs_ahead_gt=True, depth still sees risers -> stairs_clear
    # False): the body height freezes, so the watchdog fires -- but the climb must HOLD, never hand
    # the incline to the flat walker, and record a retry heartbeat instead.
    for _ in range(60):                       # 3.0 s wedged -> several stall windows
        t += 0.05
        r = ho.update(now=t, dt=0.05, base_z=z, **base)   # z frozen
        assert r["state"] == "climb", f"a wedge on the incline must HOLD, never hand back: {r}"
    assert r["telemetry"]["handoff_climb_stall_retries"] >= 1, \
        f"the stall watchdog must still FIRE (retry heartbeat) on a wedge: {r['telemetry']}"
    print("climb_progress_watchdog OK  (rising climb continues; wedged climb HOLDS the incline + retries)")


def test_post_climb_reacquire():
    """After top_egress_done, forward floor is suppressed while yaw is large (robot spins
    in place to re-acquire the patient), then re-enabled once yaw re-aligns."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_arm_after_secs=0.0,
                        climb_backend="blind_rl", climb_max_sec=999.0, climb_stall_timeout_sec=999.0,
                        top_egress_enabled=True, top_clear_debounce_sec=0.1,
                        top_egress_distance_m=0.50, top_egress_goal_stop_m=0.12,
                        climb_progress_min_m=0.05,
                        post_climb_yaw_threshold_deg=20.0)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    Dstairs, Dflat = synth_staircase_depth(), synth_flat_depth()

    # 1. Engage the climb.
    base_climb = dict(go2=object(), stairs_action_active=True, body_speed=0.2,
                      roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                      person_detected=False, y_lateral=0.0, body_fwd=0.2, cmd_vx=0.22,
                      person_gap_m=9.0)
    ho.update(now=1.0, dt=0.05, depth_hw=Dstairs, riser_dist_ahead=0.50, stairs_ahead_gt=True,
              base_x=0.0, base_z=0.30, yaw=0.0, **base_climb)
    assert ho.state == "climb", "should be in climb after engage"

    # 2. Complete the egress (flat depth + GT clear for debounce + travel >= 0.50).
    t, bx, z = 1.0, 0.0, 0.30
    handed = False
    for _ in range(60):
        t += 0.05; bx += 0.02; z += 0.001
        r = ho.update(now=t, dt=0.05, depth_hw=Dflat, riser_dist_ahead=None, stairs_ahead_gt=False,
                      base_x=bx, base_z=z, yaw=0.0, **base_climb)
        if r["state"] == "walk":
            handed = True
            break
    assert handed, "egress should complete and hand back to walk"
    assert ho._post_climb_reacquire, "_post_climb_reacquire must be True right after egress handback"

    # 3. Large yaw (simulating robot rotated 80 deg during climb) -> vx_floor must be None.
    r_large = ho.update(now=t + 0.05, dt=0.05, depth_hw=Dflat, riser_dist_ahead=None,
                        stairs_ahead_gt=False, base_x=bx, base_z=z,
                        yaw=math.radians(80.0), **base_climb)
    assert r_large["state"] == "walk"
    assert r_large.get("vx_floor") is None, \
        f"forward floor must be suppressed while yaw=80 deg off-axis; got {r_large.get('vx_floor')}"
    assert r_large.get("wz_override") is not None, "yaw correction wz must still be emitted"

    # 4. Small yaw (re-aligned) -> vx_floor resumes and flag clears.
    r_small = ho.update(now=t + 0.10, dt=0.05, depth_hw=Dflat, riser_dist_ahead=None,
                        stairs_ahead_gt=False, base_x=bx, base_z=z,
                        yaw=math.radians(5.0), **base_climb)
    assert r_small.get("vx_floor") is not None and r_small["vx_floor"] > 0.0, \
        f"forward floor must resume once yaw is small; got {r_small.get('vx_floor')}"
    assert not ho._post_climb_reacquire, "_post_climb_reacquire must clear once re-aligned"

    # 5. Alternative: person re-detected while still off-axis -> flag clears immediately.
    ho._post_climb_reacquire = True  # reset for this sub-test
    base_person = dict(base_climb)
    base_person["person_detected"] = True
    r_person = ho.update(now=t + 0.15, dt=0.05, depth_hw=Dflat, riser_dist_ahead=None,
                         stairs_ahead_gt=False, base_x=bx, base_z=z,
                         yaw=math.radians(60.0), **base_person)
    assert not ho._post_climb_reacquire, "_post_climb_reacquire must clear when person re-detected"
    print("post_climb_reacquire OK  (fwd floor suppressed while off-axis; resumes on re-align or person detect)")


def test_caller_hold_blocks_engage_and_walk_floor():
    """Incident 8.15 extension (run_sim_20260711_155123_326): while the perception
    controller commands a stance-hold (caller_hold=True, main.py hold_request forwarded
    from isaac_env's F1 _motion_hold_requested capture), the FSM must not ENGAGE a new
    climb nor emit any walk-state forward floor. That run: the depth detector read the
    STANDING PATIENT on the top landing as an 8-step staircase (8.3's person-as-risers),
    the commit vx_floor armed the stall detector against the commanded stop, a
    'wedge_stall' climb engaged AT the patient (both engages during a continuous
    fsm=STOP/hold=True/vx=0 stretch) and drove the dog up the patient's legs -- flip at
    x=8.63, roll 180 deg. Every legitimate engage happens with the caller allowing motion
    (hold=False), so this veto costs nothing on the designed paths."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_max_sec=25.0, stair_commit_arm_after_secs=0.0,
                        stall_consec_sec=0.3)
    D = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=D, stairs_action_active=True, base_z=0.30,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0)

    # 1. HELD + wedged (cmd 0, body 0, patient-as-stairs depth, "riser" confirmed ahead --
    #    the run's exact landing signature): no commit floor, no stall arming, NO engage.
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    t = 0.0
    for _ in range(40):  # 2.0 s >> stall_consec_sec
        t += 0.05
        r = ho.update(now=t, dt=0.05, cmd_vx=0.0, body_speed=0.0, body_fwd=0.0,
                      riser_dist_ahead=0.50, caller_hold=True, **base)
        assert r["state"] == "walk", f"caller_hold must veto engage: {r}"
        assert r.get("vx_floor") is None, \
            f"no walk-state forward floor against a commanded stance-hold: {r.get('vx_floor')}"
        assert not r.get("tread_creep_active"), "no tread-creep push under caller_hold"

    # 2. SAME inputs with caller_hold released -> the commit floor arms, and the approach
    #    engage fires (riser at standoff + committing) -- the designed path still works.
    ho2 = HandoffController(cfg, _FakePgtt(), logger=None)
    engaged = False
    t = 0.0
    for _ in range(40):
        t += 0.05
        r = ho2.update(now=t, dt=0.05, cmd_vx=0.0, body_speed=0.0, body_fwd=0.0,
                       riser_dist_ahead=0.50, caller_hold=False, **base)
        if r["state"] == "climb":
            engaged = True
            break
    assert engaged, "with caller_hold False the same scenario must still engage (designed path)"

    # 3. caller_hold arriving MID-CLIMB must NOT abort/clamp the ongoing climb (8.9 /
    #    blind-carry: never strand the climber on the incline because the controller
    #    asked to stop -- the mid-climb exits stay tilt/progress/egress-owned).
    r = ho2.update(now=t + 0.05, dt=0.05, cmd_vx=0.0, body_speed=0.0, body_fwd=0.0,
                   riser_dist_ahead=0.50, caller_hold=True, **base)
    assert r["state"] == "climb" and r["climb"], \
        f"an ONGOING climb must not be aborted by caller_hold: {r}"
    print("caller_hold OK  (hold vetoes engage + walk floors; release engages; mid-climb unaffected)")


def test_run18_wedge_stall_releases_once_patient_lead_crosses_threshold():
    """Fix C documentation-as-test (2026-07-12 review, run 18 wedge,
    run_sim_20260712_103237_267). Runs 15-18 all settled short of the stairs; run 18 got
    the dog to within 0.30 m of riser 1 (JAMMED -- below climb_min_room_m 0.40, so
    approach_room can never engage; every one of run 18's logged vetoes was
    reason="wedge_stall") and wedged there. The Isaac-side stair-commit vx_floor
    (independent of main.py's own STAIR_LOSS_FLOOR forward command -- see the Fix A/B
    near-field/too-close review on the same incident) already drove the walker into the
    riser and kept it stalled for the whole tail of the run (fall_diag telemetry:
    handoff_state=walk, stair_commit=True, stalled=True, stall_cmd_sec~20 s continuously
    from sim_t=36.2 onward, hold_request=False throughout -- caller_hold was never the
    live blocker in this run). Every OTHER wedge_stall precondition was therefore already
    satisfied (has_stairs, near_enough, not caller_hold, not ghost-vetoed once the
    patient is this far up the stairs). The ONLY gate still closed was
    stair_entry_min_lead_m (2.4 m): the run's LAST logged frame (vision_main_trace.jsonl
    sim_t=41.895) has gt_patient.x=4.1326 and robot x_m=1.741 -> patient_lead_m=2.392 --
    0.008 m short of the 2.4 m threshold -- and the run simply ran out of frames right
    there (isaac_env.jsonl's last handoff_engage_vetoed_lead, sim_t~35.35, already shows
    patient_lead_m=2.306 climbing toward it; zero handoff_engage events anywhere in the
    file). CONCLUSION: wedge_stall does NOT need a widened engage window (no
    HandoffConfig / handoff_controller.py change) -- it fires the instant patient_lead_m
    crosses the existing 2.4 m gate, exactly as designed. This test reproduces that exact
    boundary against the real HandoffController state machine so a future change to
    stair_entry_min_lead_m or the wedge_stall preconditions cannot silently invalidate
    this "no sim-side change needed" conclusion without failing a test."""
    cfg = HandoffConfig(climb_attempt=True, climb_engage_standoff_m=0.65, climb_min_room_m=0.40,
                        stair_commit_enabled=True, require_controller_stairs=False,
                        stair_commit_max_sec=25.0, stair_commit_arm_after_secs=0.0,
                        stair_entry_min_lead_m=2.4, stall_consec_sec=0.3,
                        ghost_engage_gap_window_m=0.5)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    base = dict(go2=object(), depth_hw=D, stairs_action_active=True, base_z=0.30,
                roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0, height_above_step=0.30,
                person_detected=False, yaw=0.0, y_lateral=0.0, body_fwd=0.0, body_speed=0.0,
                caller_hold=False, base_x=1.74)
    # Wedged JAMMED at the riser (0.30 < climb_min_room_m 0.40 -- approach_room can never
    # fire): commanded forward (the walk-state commit floor / main.py's own loss floor,
    # either source -- the stall detector only sees the effective commanded speed) but
    # not moving -> stall accumulates. Patient lead held just below the run-18 final-frame
    # value (2.39) for long enough to satisfy stall_consec_sec -- must stay HELD.
    t = 0.0
    r = None
    for _ in range(20):  # 1.0 s >> stall_consec_sec (0.3 s)
        t += 0.05
        r = ho.update(now=t, dt=0.05, cmd_vx=0.16, riser_dist_ahead=0.30,
                      person_gap_m=2.39, patient_lead_m=2.39, **base)
    assert r["state"] == "walk", f"lead still below threshold -> must stay held: {r}"
    assert r["stalled"], f"the wedge itself must register as a stall: {r}"
    # The SAME wedge, one more frame later, with the patient's lead now past the gate
    # (2.41 > 2.4) -- the already-satisfied wedge_stall condition fires immediately, no
    # other code path touched.
    r = ho.update(now=t + 0.05, dt=0.05, cmd_vx=0.16, riser_dist_ahead=0.30,
                  person_gap_m=2.41, patient_lead_m=2.41, **base)
    assert r["state"] == "climb" and r["climb"], \
        f"lead crossing 2.4 m must release the SAME wedge into wedge_stall engage: {r}"
    print("run18 wedge_stall OK  (jammed + stalled wedge held below the lead gate, "
          "releases the instant lead crosses 2.4 m -- no sim-side change needed)")


if __name__ == "__main__":
    test_detector()
    test_stall()
    test_fsm()
    test_stair_commit()
    test_approach_engage()
    test_ghost_engage_veto_pure_function()
    test_ghost_engage_veto_blocks_person_as_risers_stall_engage()
    test_ghost_engage_veto_does_not_block_real_engage_with_far_patient()
    test_stair_entry_lead_ok_pure_function()
    test_stair_entry_gate_holds_a_close_approach_engage_and_releases_once_ahead()
    test_stair_entry_gate_no_gt_lead_does_not_block_real_engage()
    test_stair_entry_gate_never_clamps_an_ongoing_climb()
    test_stair_entry_gate_interval_is_non_empty_against_isaac_env_hard_wait()
    test_top_egress()
    test_egress_stops_at_goal()
    test_no_false_crest_between_risers()
    test_climb_progress_watchdog()
    test_post_climb_reacquire()
    test_caller_hold_blocks_engage_and_walk_floor()
    test_run18_wedge_stall_releases_once_patient_lead_crosses_threshold()
    print("ALL HANDOFF TESTS PASS")
