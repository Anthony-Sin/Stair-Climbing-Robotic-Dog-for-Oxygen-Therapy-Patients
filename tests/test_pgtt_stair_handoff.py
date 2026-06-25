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
sys.path.insert(0, REPO)  # go2_locomotion package lives at the repo root

from go2_locomotion.pgtt_stair_handoff import (  # noqa: E402
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
                        stair_commit_arm_after_secs=0.0,  # disable arm delay for unit test
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
    """A still-RISING climb is not cut off; a wedged (no-height-gain) climb hands back."""
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
    # Now WEDGE it: body height frozen -> watchdog fires within climb_stall_timeout_sec.
    handed = False
    for _ in range(40):
        t += 0.05
        r = ho.update(now=t, dt=0.05, base_z=z, **base)   # z frozen
        if r["state"] == "walk":
            handed = True
            break
    assert handed, "a wedged climb (no height gain) must hand back via the progress watchdog"
    print("climb_progress_watchdog OK  (rising climb continues; wedged climb hands back)")


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


if __name__ == "__main__":
    test_detector()
    test_stall()
    test_fsm()
    test_stair_commit()
    test_approach_engage()
    test_top_egress()
    test_egress_stops_at_goal()
    test_no_false_crest_between_risers()
    test_climb_progress_watchdog()
    test_post_climb_reacquire()
    print("ALL HANDOFF TESTS PASS")
