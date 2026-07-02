"""Adapt a Unitree ``LowState`` into the duck-typed "articulation" the policies expect.

The PGTT and blind_rl policies were written against an Isaac articulation object
(methods ``get_world_pose``, ``get_angular_velocity``, ``get_joint_positions/velocities``
and the various ``set_*`` writers). This adapter presents the same surface over a
``unitree_go/msg/LowState`` so the EXACT same policy code runs on the real robot --
no policy changes, just a different state source. (It is the hardware twin of the
sim's articulation, mirroring the original ``PhysicalGo2Articulation``.)

Two things make the joint-order trap disappear:
  * ``dof_names`` is advertised in the real **FR-first** SDK order
    (FR,FL,RR,RL x hip,thigh,calf). The policies build their obs/act remaps from
    these NAMES, so feeding them FR-first names makes the remaps resolve to the SDK
    order automatically and the LowCmd write-back is identity.
  * ``get_world_pose`` returns position = zeros. The policies only use base XY to
    center PGTT's heightscan; zero XY means the heightscan grid is sampled in the
    BODY frame, which is exactly what the real LiDAR heightscan provider produces.

The ``set_*`` writers are no-ops: on hardware the joint targets are read back from
``policy.last_targets_isaac`` and sent as ``/lowcmd`` by the control node, not pushed
into a physics engine.

LowState field names follow the verified ``unitree_go/msg`` IDL: ``imu_state`` has
``quaternion`` [w,x,y,z], ``gyroscope`` [3], ``accelerometer`` [3]; ``motor_state[i]``
has ``q``/``dq``. We read them defensively (``gyroscope`` or legacy ``gyro``) so the
adapter survives small driver naming differences -- a real sim2real footgun.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from go2_locomotion.go2_locomotion_utils import quat_to_matrix

# Real Unitree SDK motor order: legs FR, FL, RR, RL; each hip, thigh, calf. These
# names parse through the policies' classify_dof() to (leg, joint), so advertising
# them in this order makes every policy remap resolve to the SDK index order.
GO2_DOF_NAMES: Tuple[str, ...] = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
N_MOTORS = 12


def _imu_field(imu: Any, *names: str) -> Optional[np.ndarray]:
    for n in names:
        v = getattr(imu, n, None)
        if v is not None:
            return np.asarray(v, dtype=np.float32)
    return None


class LowStateArticulation:
    """Duck-typed articulation backed by one ``LowState`` snapshot.

    Construct (or ``update(low_state)``) once per control tick, then hand it to
    ``policy.step(articulation, ...)``.
    """

    def __init__(self, low_state: Any, *, linear_velocity: Optional[Sequence[float]] = None) -> None:
        self.low_state = low_state
        # Body-frame base velocity (from SportModeState if available, else None). The
        # handoff stall detector uses it; None is handled gracefully there.
        self._lin_vel = None if linear_velocity is None else np.asarray(linear_velocity, dtype=np.float32)

    def update(self, low_state: Any, *, linear_velocity: Optional[Sequence[float]] = None) -> "LowStateArticulation":
        self.low_state = low_state
        self._lin_vel = None if linear_velocity is None else np.asarray(linear_velocity, dtype=np.float32)
        return self

    @property
    def dof_names(self) -> List[str]:
        return list(GO2_DOF_NAMES)

    # ---------------------------------------------------------------- base state
    def get_world_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """(position, quaternion). Position is zeros (see module docstring); quat is
        the IMU orientation [w, x, y, z]."""
        quat = _imu_field(self.low_state.imu_state, "quaternion")
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.zeros(3, dtype=np.float32), quat.astype(np.float32)

    def get_angular_velocity(self) -> np.ndarray:
        """World-frame angular velocity = R(quat) @ body_gyro (policies expect world)."""
        quat = _imu_field(self.low_state.imu_state, "quaternion")
        gyro = _imu_field(self.low_state.imu_state, "gyroscope", "gyro")
        if quat is None or gyro is None:
            return np.zeros(3, dtype=np.float32)
        rot = quat_to_matrix(quat)
        return (rot @ gyro).astype(np.float32)

    def get_linear_velocity(self) -> Optional[np.ndarray]:
        """Body/world planar velocity if a SportModeState estimate was supplied, else None."""
        return self._lin_vel

    # --------------------------------------------------------------- joint state
    def get_joint_positions(self) -> np.ndarray:
        ms = self.low_state.motor_state
        return np.array([ms[i].q for i in range(N_MOTORS)], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        ms = self.low_state.motor_state
        return np.array([ms[i].dq for i in range(N_MOTORS)], dtype=np.float32)

    # --------------------------------------------------- writers (no-ops on real)
    # The policies call one of these to "apply" their targets in sim. On hardware
    # the targets are read from policy.last_targets_isaac and sent as /lowcmd, so
    # these do nothing -- but they must exist for the policy code to run unchanged.
    def set_joint_position_targets(self, *_a, **_k) -> None: ...
    def set_joint_positions(self, *_a, **_k) -> None: ...
    def set_joint_positions_to_apply(self, *_a, **_k) -> None: ...
    def set_joint_efforts(self, *_a, **_k) -> None: ...
    def set_joint_efforts_to_apply(self, *_a, **_k) -> None: ...
    def apply_action(self, *_a, **_k) -> None: ...
