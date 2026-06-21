"""Parity tests for the RL locomotion deployment contract.

These pin the constants a future real LowCmd controller MUST reproduce to run the
sim's policy (observation layout, action scale, default pose, joint order, the
policy<->articulation joint remap, and the PD gains in RADIAN units). The contract
is exported at sim startup to reports/rl_deployment_contract.json; this test asserts
the same contract object is internally consistent and that the joint remap
round-trips, so a real deployment checked against the manifest cannot silently
drift from the sim policy.

No Isaac/torch/model file needed: RLLocomotionPolicy.__init__ is bypassed via
object.__new__ (same approach as test_lidar_fusion.py). Run directly
(python tests/test_rl_contract.py) or via pytest.
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("core", os.path.join("sim", "isaac")):
    _p = os.path.join(_REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from locomotion import rl_locomotion_policy as rlp


# Nucleus Go2 reports its DOFs joint-type-major (all hips, then thighs, then
# calfs) -- deliberately NOT the policy's leg-major FR/FL/RR/RL order, so the
# remap is actually exercised.
_ISAAC_DOFS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]


def _fake_policy(dof_names=None, kp=20.0, kd=0.5):
    """Build a policy with the real joint-map/contract logic but no model load."""
    pol = object.__new__(rlp.RLLocomotionPolicy)  # bypass __init__ (no model needed)
    pol.dof_names = list(dof_names if dof_names is not None else _ISAAC_DOFS)
    pol.policy_to_isaac = pol._build_joint_map(pol.dof_names)
    pol.n = len(rlp.POLICY_JOINT_ORDER)
    pol.default_pos_policy = np.array(
        [rlp.POLICY_DEFAULT_BY_JOINT[j] for (_l, j) in rlp.POLICY_JOINT_ORDER], np.float32
    )
    pol.action_scale_policy = np.array(
        [rlp.POLICY_ACTION_SCALE_BY_JOINT[j] for (_l, j) in rlp.POLICY_JOINT_ORDER], np.float32
    )
    pol.config = rlp.RLLocomotionPolicyConfig(policy_path="(none)", kp=kp, kd=kd)
    pol._policy_kind = "torchscript"
    pol.policy_path = rlp.Path("(nonexistent)")
    return pol


def test_joint_map_correct_and_roundtrips():
    pol = _fake_policy()
    # Each policy slot maps to an isaac DOF whose name carries that leg + joint.
    for slot, (leg, joint) in enumerate(rlp.POLICY_JOINT_ORDER):
        name = pol.dof_names[pol.policy_to_isaac[slot]].lower()
        assert leg in name and joint in name, (leg, joint, name)
    # policy -> isaac -> policy is the identity (no value lands in the wrong joint).
    v = np.arange(12, dtype=np.float32)
    back = pol._isaac_to_policy_vector(pol._policy_to_isaac_vector(v))
    assert np.allclose(back, v)


def test_contract_observation_layout_sums_to_45():
    c = _fake_policy().deployment_contract()
    assert c["observation"]["size"] == 45
    assert sum(seg["dim"] for seg in c["observation"]["layout"]) == 45
    # Segment order is the trained obs order.
    names = [seg["name"] for seg in c["observation"]["layout"]]
    assert names == [
        "base_ang_vel_body", "projected_gravity", "commands",
        "dof_pos_minus_default", "dof_vel", "prev_action",
    ]


def test_contract_action_and_default_pose():
    c = _fake_policy().deployment_contract()
    assert c["action"]["type"] == "joint_position_residual"
    assert len(c["action"]["scale_vector_policy_order"]) == 12
    assert c["default_pose_rad"]["by_joint"] == {"hip": 0.0, "thigh": 0.8, "calf": -1.5}
    assert c["action"]["scale_by_joint_rad"] == {"hip": 0.125, "thigh": 0.25, "calf": 0.25}
    # Joint order is leg-major FR,FL,RR,RL.
    assert c["joint_order"][0] == "fr_hip"
    assert c["joint_order"][-1] == "rl_calf"
    assert len(c["joint_order"]) == 12


def test_contract_gains_are_radian_units():
    # The documented footgun: USD DriveAPI defaults to DEGREES; the policy needs
    # radian-unit gains. The contract must state this explicitly.
    c = _fake_policy(kp=20.0, kd=0.5).deployment_contract()
    assert c["actuation"]["gain_units"] == "radian"
    assert c["actuation"]["kp"] == 20.0 and c["actuation"]["kd"] == 0.5
    assert c["actuation"]["control_mode"] == "torque"


def test_contract_joint_map_names_match():
    c = _fake_policy().deployment_contract()
    for (leg, joint), name in zip(rlp.POLICY_JOINT_ORDER, c["joint_map"]["mapped_isaac_names"]):
        assert leg in name.lower() and joint in name.lower()
    # Missing checkpoint file -> sha256 is None, not a crash.
    assert c["policy"]["sha256"] is None


class _FakeArtic:
    """Minimal articulation stub for exercising the explicit-PD torque path."""

    def __init__(self, q, qd, limits):
        self._q = np.asarray(q, np.float32)
        self._qd = np.asarray(qd, np.float32)
        self._limits = np.asarray(limits, np.float32)
        self.applied = None

    def get_joint_positions(self):
        return self._q

    def get_joint_velocities(self):
        return self._qd

    def get_dof_limits(self):
        return self._limits

    def set_joint_efforts(self, efforts):
        self.applied = np.asarray(efforts, np.float32)


def _torque_policy(**cfg_kw):
    pol = object.__new__(rlp.RLLocomotionPolicy)
    pol.dof_names = ["a", "b", "c"]
    pol.logger = None
    pol._limits_read = False
    pol._joint_pos_lower = None
    pol._joint_pos_upper = None
    pol._last_torque = np.zeros(3, np.float32)
    # kp=10, kd=0 so tau == 10 * effective position error; no clip / no slew.
    pol.config = rlp.RLLocomotionPolicyConfig(
        policy_path="(none)", kp=10.0, kd=0.0, torque_limit=1.0e6, **cfg_kw
    )
    return pol


def test_torque_joint_limit_clamp_reads_from_articulation():
    pol = _torque_policy(joint_limit_clamp=True)
    pol.last_targets_isaac = np.array([5.0, 0.0, -5.0], np.float32)  # 1st/3rd out of range
    art = _FakeArtic(q=[0, 0, 0], qd=[0, 0, 0], limits=[[-1, 1], [-1, 1], [-1, 1]])
    pol._apply_torque_control(art)
    # Target clamped to [-1, 1] -> tau = 10 * clamped_target.
    assert np.allclose(art.applied, [10.0, 0.0, -10.0])
    assert pol._joint_pos_lower is not None  # limits were actually read


def test_torque_backlash_deadbands_small_errors():
    pol = _torque_policy(backlash_rad=0.5)
    pol.last_targets_isaac = np.array([0.3, 1.0, 0.0], np.float32)
    art = _FakeArtic(q=[0, 0, 0], qd=[0, 0, 0], limits=[[-9, 9], [-9, 9], [-9, 9]])
    pol._apply_torque_control(art)
    # err 0.3 is within +/-0.5 band -> 0; err 1.0 -> 0.5; err 0 -> 0. tau = 10*err.
    assert np.allclose(art.applied, [0.0, 5.0, 0.0])


def test_torque_derate_scales_output():
    pol = _torque_policy(torque_derate=0.5)
    pol.last_targets_isaac = np.array([1.0, 0.0, 0.0], np.float32)
    art = _FakeArtic(q=[0, 0, 0], qd=[0, 0, 0], limits=[[-9, 9], [-9, 9], [-9, 9]])
    pol._apply_torque_control(art)
    assert np.allclose(art.applied, [5.0, 0.0, 0.0])  # 0.5 * (10 * 1.0)


def test_torque_default_is_ideal_pd():
    # All actuator knobs off => plain tau = kp*(target-q) - kd*qd, unclamped.
    pol = _torque_policy()
    pol.last_targets_isaac = np.array([2.0, 0.0, 0.0], np.float32)
    art = _FakeArtic(q=[0, 0, 0], qd=[0, 0, 0], limits=[[-1, 1], [-1, 1], [-1, 1]])
    pol._apply_torque_control(art)
    assert np.allclose(art.applied, [20.0, 0.0, 0.0])  # NOT clamped (clamp off)


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for _fn in _tests:
        _fn()
        print("PASS", _fn.__name__)
    print(f"ALL {len(_tests)} TESTS PASSED")
