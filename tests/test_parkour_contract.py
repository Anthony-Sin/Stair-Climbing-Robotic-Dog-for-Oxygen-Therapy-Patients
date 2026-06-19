"""Offline (no Isaac) contract test for the Extreme-Parkour Go2 runner -- the
sole locomotion policy.

Weight-free checks (always run): depth preprocess shape/range, the fixed config
contract (control_hz=50, kp=40/kd=1, realism off by default), and the per-leg
swing/stance command summary.

Weighted checks (skip cleanly if the gitignored weights are absent): loads the
real shipped weights through ParkourLocomotionPolicy against a stub articulation
and asserts the end-to-end wiring: proprio(53) -> depth encode -> estimator ->
history encoder -> actor(114) -> 12 actions, joint remap (type-major Isaac order
-> policy order), torque clamp, the delta_yaw heading-command path, and that the
sim-to-real realism suite (obs noise + latency + actuator imperfections) keeps
torques finite and within the per-leg limits.

Run: python tests/test_parkour_contract.py
"""

import logging
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))
sys.path.insert(0, os.path.join(REPO, "sim", "bot"))
sys.path.insert(0, os.path.join(REPO, "core"))

ASSETS = os.path.join(REPO, "sim", "isaac", "assets", "policies", "parkour")
BASE = os.path.join(ASSETS, "base_jit.pt")
VISION = os.path.join(ASSETS, "vision_weight.pt")

# Isaac Nucleus Go2 reports DOFs joint-type-major (all hips, then thighs, then
# calves) -- exercise the name remap with that order, not the policy order.
ISAAC_DOF_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]


class StubGo2:
    def __init__(self, dof_names):
        self.dof_names = list(dof_names)
        self._q = np.zeros(12, dtype=np.float32)
        self._qd = np.zeros(12, dtype=np.float32)
        self.efforts = None

    def get_world_pose(self):
        return (np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def get_angular_velocity(self):
        return np.zeros(3, dtype=np.float32)

    def get_joint_positions(self):
        return self._q

    def get_joint_velocities(self):
        return self._qd

    def set_joint_efforts(self, e):
        self.efforts = np.asarray(e, dtype=np.float32)


def _fake_policy():
    """A ParkourLocomotionPolicy with __init__ bypassed (no model needed) so the
    pure-Python leg_command_summary() swing/stance logic can be tested offline."""
    from parkour_locomotion_policy import ParkourLocomotionPolicy, PARKOUR_DEFAULT_POS
    pol = object.__new__(ParkourLocomotionPolicy)
    pol._last_target_policy = PARKOUR_DEFAULT_POS.copy()
    pol.prev_action = np.zeros(12, dtype=np.float32)
    return pol


def _test_weight_free():
    from parkour_locomotion_policy import (
        ParkourLocomotionPolicy, ParkourPolicyConfig, PARKOUR_DEFAULT_POS, PARKOUR_JOINT_ORDER,
        max_body_tilt_rad,
    )

    # 1) depth preprocess: [60,106] metres -> [1,58,87] in [-0.5, 0.5]
    raw = np.linspace(0.0, 4.0, 60 * 106, dtype=np.float32).reshape(60, 106)
    dep = ParkourLocomotionPolicy.preprocess_depth(raw, 0.0, 2.0)
    assert tuple(dep.shape) == (1, 58, 87), f"depth shape {tuple(dep.shape)}"
    assert -0.501 <= float(dep.min()) and float(dep.max()) <= 0.501, \
        f"depth range [{float(dep.min()):.3f}, {float(dep.max()):.3f}]"
    print(f"OK depth preprocess -> {tuple(dep.shape)} range "
          f"[{float(dep.min()):.3f}, {float(dep.max()):.3f}]")

    # 2) fixed config contract: 50 Hz control, kp40/kd1, realism off by default.
    cfg = ParkourPolicyConfig(base_model_path="(none)", vision_model_path="(none)")
    assert cfg.control_hz == 50.0, f"control_hz {cfg.control_hz}"
    assert cfg.kp == 40.0 and cfg.kd == 1.0, f"gains {cfg.kp}/{cfg.kd}"
    assert cfg.obs_noise_enabled is False and int(cfg.obs_latency_steps) == 0
    assert cfg.joint_limit_clamp is False
    assert cfg.backlash_rad == 0.0 and cfg.torque_derate == 1.0 and cfg.torque_rate_limit_nm == 0.0
    assert cfg.stair_action_norm_max == 8.0
    assert abs(max_body_tilt_rad(-0.41, 0.22) - 0.41) < 1e-9
    print("OK config contract: control_hz=50, kp=40/kd=1, realism off by default")

    # 3) leg_command_summary swing/stance from the live target (no model needed).
    s = _fake_policy().leg_command_summary()
    assert s["swing_legs"] == [], f"default should be all stance, got {s['swing_legs']}"
    assert set(s["leg_commands"]) == {"FL", "FR", "RL", "RR"}
    assert all(c["state"] == "stance" for c in s["leg_commands"].values())

    pol = _fake_policy()
    tp = PARKOUR_DEFAULT_POS.copy()
    fr_calf = [i for i, (l, j) in enumerate(PARKOUR_JOINT_ORDER) if l == "fr" and j == "calf"][0]
    tp[fr_calf] = -2.1  # bend the knee past the -1.5 default -> leg retracts -> swing
    pol._last_target_policy = tp
    s = pol.leg_command_summary()
    assert s["swing_legs"] == ["FR"], f"expected FR swing, got {s['swing_legs']}"
    assert s["leg_commands"]["FR"]["state"] == "swing"
    assert s["leg_commands"]["FR"]["foot_lift_m"] > 0.02
    assert s["leg_commands"]["FL"]["state"] == "stance"
    print("OK leg_command_summary swing/stance from live target")


def _test_person_mask():
    """Terrain-preserving person mask keeps the step the person stands on visible to the
    policy (the stair-base fall fix), while still removing the near body that causes the
    close-range surge. The legacy 'far' fill blanks the box to clear (reproduces the fall).
    """
    from parkour_depth_mask import mask_person_in_parkour_depth

    H, W = 60, 106
    # Staircase-ish ground: far at the top of the frame, nearer at the bottom (0.4..1.2 m).
    depth = np.repeat(np.linspace(1.2, 0.4, H, dtype=np.float32)[:, None], W, axis=1)
    # Person standing centered and near: a vertical slab at ~0.45 m occluding the steps.
    body = depth.copy()
    body[18:51, 42:64] = 0.45
    pb = [0.40, 0.30, 0.60, 0.85]

    out_t, box_t, st_t = mask_person_in_parkour_depth(body, pb, fill_mode="terrain")
    out_f, box_f, st_f = mask_person_in_parkour_depth(body, pb, fill_mode="far")
    assert box_t is not None and box_f is not None, "mask should map a valid box"
    cx1, cy1, cx2, cy2 = box_t
    roi_t = out_t[cy1:cy2, cx1:cx2]
    roi_f = out_f[cy1:cy2, cx1:cx2]
    # Terrain fill: NO far/sky values leak in (the policy still sees the riser), the body
    # slab is removed (raised toward the terrain reference), and real terrain is preserved.
    assert roi_t.max() < 50.0, f"terrain fill leaked a far value: max={roi_t.max()}"
    assert st_t["terrain_ref_m"] is not None and 0.3 < st_t["terrain_ref_m"] < 1.3, \
        f"terrain ref out of range: {st_t['terrain_ref_m']}"
    assert st_t["preserved_terrain_px"] > 0, "terrain fill preserved no real terrain"
    assert roi_t.min() > 0.45, "near body (0.45 m) not removed by terrain fill"
    # Far fill: the legacy 'clear' blanking that blinds the policy to the step.
    assert roi_f.min() >= 1e5 - 1.0, f"far fill should be 1e5, got {roi_f.min()}"
    assert st_f["fill_mode"] == "far"
    print("OK person mask: terrain keeps the step (ref=%.2fm, %d px kept), far blanks to clear"
          % (st_t["terrain_ref_m"], st_t["preserved_terrain_px"]))

    # Genuinely occluded (no valid terrain visible anywhere) -> far fallback, so the near
    # body never leaks through as terrain (the surge is still suppressed).
    sky = np.full((H, W), 1.0e5, dtype=np.float32)
    out_o, box_o, st_o = mask_person_in_parkour_depth(sky, [0.30, 0.30, 0.70, 0.70], fill_mode="terrain")
    assert st_o["fill_mode"] == "far_fallback", f"expected far_fallback, got {st_o['fill_mode']}"
    assert out_o[box_o[1]:box_o[3], box_o[0]:box_o[2]].min() >= 1e5 - 1.0
    print("OK person mask: no-terrain occlusion falls back to far-fill (no surge leak)")

    # Bad/empty bbox -> input returned unchanged, no box, no stats.
    o2, b2, s2 = mask_person_in_parkour_depth(body, None)
    assert b2 is None and s2 is None, "bad bbox should return (input, None, None)"
    print("OK person mask: bad bbox returns input unchanged")


def _test_heading_slew():
    """The parkour heading (delta_yaw) command is slew-limited so a bbox jump cannot snap
    the bearing and jolt the gait at a terrain transition (the smoothing primitive)."""
    from pid_controller import SlewRateLimiter

    lim = SlewRateLimiter(3.0)            # 3 rad/s
    lim.reset(0.0)
    y = lim.update(1.0)                   # step input from 0 -> 1
    assert 0.0 < y < 1.0, f"slew should rate-limit a step, not jump to target, got {y}"

    lim0 = SlewRateLimiter(0.0)           # 0 disables slew -> pass-through
    lim0.reset(0.0)
    assert lim0.update(1.0) == 1.0, "slew rate 0 should pass the target through"
    print("OK heading slew limiter rate-limits a step (%.3f<1.0) and passes through when disabled" % y)


def _run_pipeline(cfg, label, *, delta_yaw=None):
    """Step the loaded policy and assert finite torques within the per-leg limits."""
    from parkour_locomotion_policy import ParkourLocomotionPolicy
    policy = ParkourLocomotionPolicy(cfg, ISAAC_DOF_NAMES, logger=logging.getLogger("parkour_test"))
    go2 = StubGo2(ISAAC_DOF_NAMES)
    policy.reset()
    raw = np.linspace(0.0, 4.0, 60 * 106, dtype=np.float32).reshape(60, 106)
    dt = policy.interval_sec
    for i in range(12):
        policy.submit_depth(raw + 0.01 * i)
        tel = policy.step(go2, (0.6, 0.0, 0.0), dt,
                          foot_contacts=np.array([30, 30, 5, 5.0]), delta_yaw=delta_yaw)
        assert tel["ran_policy"], "policy did not run at control interval"
        assert go2.efforts is not None and go2.efforts.shape == (12,)
        assert np.all(np.isfinite(go2.efforts)), "non-finite torque"
        lim = policy.torque_limits_isaac
        assert np.all(np.abs(go2.efforts) <= lim + 1e-3), "torque exceeds limit"
        assert np.all(np.isfinite(policy.prev_action)) and policy.prev_action.shape == (12,)
    diag = policy.diagnostics()
    assert diag["depth_seen"], "depth never consumed"
    summ = policy.leg_command_summary()
    assert set(summ["leg_commands"].keys()) == {"FL", "FR", "RL", "RR"}
    print(f"OK pipeline [{label}]: {diag}")


def _test_soft_hold():
    """Verify that when hold=True is passed, self.hold_strength ramps up and blends the action_np to 0.0,
    and when hold=False is passed, it ramps down to 0.0.
    """
    from parkour_locomotion_policy import ParkourLocomotionPolicy, ParkourPolicyConfig
    cfg = ParkourPolicyConfig(
        base_model_path=BASE, vision_model_path=VISION,
        hold_ramp_sec=0.25,  # 0.25 seconds to ramp
        hold_speed_threshold=0.0  # disable gating for pure ramping check
    )
    policy = ParkourLocomotionPolicy(cfg, ISAAC_DOF_NAMES, logger=logging.getLogger("parkour_test"))
    go2 = StubGo2(ISAAC_DOF_NAMES)
    policy.reset()
    raw = np.zeros((60, 106), dtype=np.float32)
    dt = policy.interval_sec  # nominal control step time (0.02s)

    # 1. Initially hold_strength should be 0.0
    assert policy.hold_strength == 0.0
    
    # 2. Run a step with hold=False. hold_strength should remain 0.0
    policy.submit_depth(raw)
    policy.step(go2, (0.0, 0.0, 0.0), dt, hold=False)
    assert policy.hold_strength == 0.0
    diag = policy.diagnostics()
    assert diag["hold_active"] is False
    assert diag["hold_strength"] == 0.0

    # 3. Run step(s) with hold=True. hold_strength should increase.
    # With hold_ramp_sec = 0.25 and control_hz = 50 (interval = 0.02),
    # step_dt = 0.02. Ramping up: 0.02 / 0.25 = 0.08 per step.
    # After 1 step of hold=True: hold_strength should be 0.08
    policy.submit_depth(raw)
    policy.step(go2, (0.0, 0.0, 0.0), dt, hold=True)
    assert abs(policy.hold_strength - 0.08) < 1e-4, f"expected hold_strength ~0.08, got {policy.hold_strength}"
    diag = policy.diagnostics()
    assert diag["hold_active"] is True
    assert diag["hold_strength"] == 0.08

    # After 13 steps (13 * 0.08 = 1.04), hold_strength should saturate to 1.0.
    # Let's run 15 steps of hold=True.
    for _ in range(15):
        policy.submit_depth(raw)
        policy.step(go2, (0.0, 0.0, 0.0), dt, hold=True)
    
    assert policy.hold_strength == 1.0
    diag = policy.diagnostics()
    assert diag["hold_active"] is True
    assert diag["hold_strength"] == 1.0
    
    # When hold_strength is 1.0, action_np should be exactly 0.0, and prev_action should be 0.0.
    assert np.all(policy.prev_action == 0.0)

    # 4. Now run step(s) with hold=False. hold_strength should ramp down.
    # After 1 step: 1.0 - 0.08 = 0.92
    policy.submit_depth(raw)
    policy.step(go2, (0.0, 0.0, 0.0), dt, hold=False)
    assert abs(policy.hold_strength - 0.92) < 1e-4, f"expected hold_strength ~0.92, got {policy.hold_strength}"

    # After 15 steps of hold=False, it should saturate to 0.0.
    for _ in range(15):
        policy.submit_depth(raw)
        policy.step(go2, (0.0, 0.0, 0.0), dt, hold=False)
    assert policy.hold_strength == 0.0
    diag = policy.diagnostics()
    assert diag["hold_active"] is False
    assert diag["hold_strength"] == 0.0
    
    # 5. reset() should also clear hold_strength
    policy.hold_strength = 0.5
    policy.reset()
    assert policy.hold_strength == 0.0

    print("OK soft-hold ramping and blending")


def main():
    _test_weight_free()
    _test_person_mask()
    _test_heading_slew()
    _test_soft_hold()

    if not (os.path.exists(BASE) and os.path.exists(VISION)):
        print(f"SKIP: parkour weights not found under {ASSETS} (weight-free checks passed)")
        return 0

    from parkour_locomotion_policy import ParkourPolicyConfig

    # Clean ("perfect env") pipeline, plus the delta_yaw heading-command path.
    _run_pipeline(
        ParkourPolicyConfig(base_model_path=BASE, vision_model_path=VISION,
                            heading_mode="command"),
        "clean + delta_yaw", delta_yaw=0.3,
    )

    # Hybrid heading mode injects the bearing at the policy level exactly like command
    # (the "self-steer on the stairs" handoff lives upstream in _step_go2_locomotion,
    # which passes delta_yaw=None there). Confirm hybrid is a valid mode and wires through.
    _run_pipeline(
        ParkourPolicyConfig(base_model_path=BASE, vision_model_path=VISION,
                            heading_mode="hybrid"),
        "hybrid + delta_yaw", delta_yaw=0.3,
    )

    # Real-simulated-env realism on: obs noise + latency + actuator imperfections
    # must keep torques finite and within the per-leg limits.
    _run_pipeline(
        ParkourPolicyConfig(
            base_model_path=BASE, vision_model_path=VISION,
            obs_noise_enabled=True, obs_latency_steps=1,
            joint_limit_clamp=True, backlash_rad=0.01,
            torque_derate=0.9, torque_rate_limit_nm=20.0,
        ),
        "realism suite on",
    )

    print("PARKOUR CONTRACT TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
