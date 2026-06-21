"""Host tests for the LowState->articulation adapter (no ROS 2 / no robot).

Pins the FR-first dof ordering and the IMU/joint reads the policies depend on.

Run: python tests/test_real_lowstate_articulation.py  (or via pytest)
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from real.control.lowstate_articulation import LowStateArticulation, GO2_DOF_NAMES, N_MOTORS
from go2_locomotion.go2_locomotion_utils import quat_to_matrix


class _IMU:
    def __init__(self, quat, gyro):
        self.quaternion = list(quat)
        self.gyroscope = list(gyro)


class _Motor:
    def __init__(self, q, dq):
        self.q = q
        self.dq = dq


class _LowState:
    def __init__(self, quat, gyro, qs, dqs):
        self.imu_state = _IMU(quat, gyro)
        self.motor_state = [_Motor(qs[i], dqs[i]) for i in range(N_MOTORS)]


def _make():
    quat = [1.0, 0.0, 0.0, 0.0]
    gyro = [0.1, -0.2, 0.3]
    qs = [float(i) * 0.1 for i in range(N_MOTORS)]
    dqs = [float(i) * 0.01 for i in range(N_MOTORS)]
    return _LowState(quat, gyro, qs, dqs), quat, gyro, qs, dqs


def test_dof_names_fr_first():
    art = LowStateArticulation(_make()[0])
    assert tuple(art.dof_names) == GO2_DOF_NAMES
    assert art.dof_names[0] == "FR_hip_joint"
    assert art.dof_names[3] == "FL_hip_joint"  # FR then FL (SDK order)
    assert len(art.dof_names) == 12


def test_joint_reads_in_order():
    ls, _, _, qs, dqs = _make()
    art = LowStateArticulation(ls)
    assert np.allclose(art.get_joint_positions(), qs)
    assert np.allclose(art.get_joint_velocities(), dqs)


def test_world_pose_zero_position_identity_quat():
    ls, quat, _, _, _ = _make()
    art = LowStateArticulation(ls)
    pos, q = art.get_world_pose()
    assert np.allclose(pos, 0.0)
    assert np.allclose(q, quat)


def test_angular_velocity_is_rot_times_gyro():
    quat = [0.9239, 0.0, 0.0, 0.3827]  # ~45 deg yaw
    gyro = [0.1, -0.2, 0.3]
    ls = _LowState(quat, gyro, [0.0] * 12, [0.0] * 12)
    art = LowStateArticulation(ls)
    expected = quat_to_matrix(np.asarray(quat, dtype=np.float32)) @ np.asarray(gyro, dtype=np.float32)
    assert np.allclose(art.get_angular_velocity(), expected, atol=1e-5)


def test_gyro_field_fallback_name():
    # Some drivers expose `gyro` instead of `gyroscope`; the adapter accepts both.
    class _IMU2:
        quaternion = [1.0, 0.0, 0.0, 0.0]
        gyro = [0.0, 0.0, 0.5]

    class _LS2:
        imu_state = _IMU2()
        motor_state = [_Motor(0.0, 0.0) for _ in range(12)]

    art = LowStateArticulation(_LS2())
    assert np.allclose(art.get_angular_velocity(), [0.0, 0.0, 0.5], atol=1e-6)


def test_linear_velocity_optional():
    ls = _make()[0]
    assert LowStateArticulation(ls).get_linear_velocity() is None
    art = LowStateArticulation(ls, linear_velocity=[0.4, 0.0, 0.0])
    assert np.allclose(art.get_linear_velocity(), [0.4, 0.0, 0.0])


def test_setters_are_noops():
    art = LowStateArticulation(_make()[0])
    art.set_joint_position_targets(np.zeros(12))
    art.set_joint_efforts(np.zeros(12))
    art.apply_action(object())  # must not raise


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
