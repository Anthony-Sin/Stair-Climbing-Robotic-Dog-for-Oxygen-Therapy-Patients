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
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))

from locomotion.pgtt_stair_handoff import (  # noqa: E402
    StallDetector, DepthStairDetector, HandoffController, HandoffConfig, GO2_LEG_CLEARANCE_M,
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
                        require_controller_stairs=False)
    ho = HandoffController(cfg, _FakePgtt(), logger=None)
    D = synth_staircase_depth()
    go2 = object()
    common = dict(go2=go2, depth_hw=D, stairs_action_active=False, base_z=0.30,
                  roll=0.0, pitch=0.0, roll_rate=0.0, pitch_rate=0.0,
                  height_above_step=0.30, yaw=0.3, y_lateral=1.0)

    # Person VISIBLE near the stairs -> no commit (the normal follow loop steers).
    r = ho.update(now=1.0, dt=0.05, cmd_vx=0.3, body_speed=0.3, body_fwd=0.3,
                  person_detected=True, **common)
    assert not r["committing"] and r["wz_override"] is None, r

    # Person LOST with stairs ahead, robot drifted off-axis (yaw=+0.3, y=+1.0) ->
    # commit: steer back (wz negative) toward yaw=0 / y=0, and apply the forward floor.
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
                        stair_commit_max_sec=25.0)
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


if __name__ == "__main__":
    test_detector()
    test_stall()
    test_fsm()
    test_stair_commit()
    test_approach_engage()
    print("ALL HANDOFF TESTS PASS")
