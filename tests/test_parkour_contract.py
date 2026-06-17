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


def main():
    _test_weight_free()

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
