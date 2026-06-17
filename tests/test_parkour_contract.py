"""Offline (no Isaac) contract test for the Extreme-Parkour Go2 runner.

Loads the real shipped weights through ParkourLocomotionPolicy against a stub
articulation and asserts the end-to-end wiring: depth preprocess shape/range,
proprio(53) -> depth encode -> estimator -> history encoder -> actor(114) -> 12
actions, joint remap (type-major Isaac order -> policy order), torque clamp.

Run: python tests/test_parkour_contract.py
Skips cleanly if the gitignored weights are absent.
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


def main():
    if not (os.path.exists(BASE) and os.path.exists(VISION)):
        print(f"SKIP: parkour weights not found under {ASSETS}")
        return 0

    from parkour_locomotion_policy import (
        ParkourLocomotionPolicy, ParkourPolicyConfig, PARKOUR_TORQUE_LIMITS,
    )

    # 1) depth preprocess: [60,106] metres -> [1,58,87] in [-0.5, 0.5]
    raw = np.linspace(0.0, 4.0, 60 * 106, dtype=np.float32).reshape(60, 106)
    dep = ParkourLocomotionPolicy.preprocess_depth(raw, 0.0, 2.0)
    assert tuple(dep.shape) == (1, 58, 87), f"depth shape {tuple(dep.shape)}"
    assert -0.501 <= float(dep.min()) and float(dep.max()) <= 0.501, \
        f"depth range [{float(dep.min()):.3f}, {float(dep.max()):.3f}]"
    print(f"OK depth preprocess -> {tuple(dep.shape)} range "
          f"[{float(dep.min()):.3f}, {float(dep.max()):.3f}]")

    cfg = ParkourPolicyConfig(base_model_path=BASE, vision_model_path=VISION)
    policy = ParkourLocomotionPolicy(cfg, ISAAC_DOF_NAMES, logger=logging.getLogger("parkour_test"))
    go2 = StubGo2(ISAAC_DOF_NAMES)
    policy.reset()

    # 2) step through several control steps (dt = control interval) with depth.
    dt = policy.interval_sec
    for i in range(12):
        policy.submit_depth(raw + 0.01 * i)
        tel = policy.step(go2, (0.6, 0.0, 0.0), dt, foot_contacts=np.array([30, 30, 5, 5.0]))
        assert tel["ran_policy"], "policy did not run at control interval"
        assert go2.efforts is not None and go2.efforts.shape == (12,)
        assert np.all(np.isfinite(go2.efforts)), "non-finite torque"
        # torque within per-joint limits (isaac order)
        lim = policy.torque_limits_isaac
        assert np.all(np.abs(go2.efforts) <= lim + 1e-3), "torque exceeds limit"
        assert np.all(np.isfinite(policy.prev_action)), "non-finite action"
        assert policy.prev_action.shape == (12,)

    diag = policy.diagnostics()
    assert diag["depth_seen"], "depth never consumed"
    print(f"OK runner: {diag}")
    summ = policy.leg_command_summary()
    assert set(summ["leg_commands"].keys()) == {"FL", "FR", "RL", "RR"}
    print("OK leg_command_summary keys + torque clamp + 114-obs actor wiring")
    print("PARKOUR CONTRACT TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
