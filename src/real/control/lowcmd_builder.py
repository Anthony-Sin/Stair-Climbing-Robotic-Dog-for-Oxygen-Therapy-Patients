"""Build the per-motor fields of a Unitree ``LowCmd`` (and the mandatory CRC).

Pure, host-testable: this module produces the 20-slot field arrays
(mode/q/dq/kp/kd/tau) and the CRC32 algorithm; the rclpy node maps them onto the
actual ``unitree_go/msg/LowCmd`` and publishes. Keeping the numeric/safety logic
here (out of the ROS node) means the joint-order, mode, gain, and damping behavior
is unit-tested with no ROS 2 present.

THREE Go2 LowCmd invariants this enforces (a violation = silently dropped or unsafe):
  * per active motor ``mode = 0x01`` (PMSM servo). 0x00 leaves the joint passive.
  * applied torque is the MOTOR's PD: ``kp*(q-q_meas) + kd*(dq-dq_meas) + tau``.
    We send ``q=target, dq=0, tau=0`` and let the motor close the loop -- this is how
    BOTH the PGTT (kp40/kd0.5) position drive and the blind_rl (kp20/kd0.5) policy
    are reproduced on hardware (the blind_rl software torque clip does NOT transfer;
    the motor enforces its own limit -- see the port plan's actuation note).
  * a valid ``crc32_core`` over the serialized command, or the MCU ignores it.

The Go2 has 12 joints but ``LowCmd.motor_cmd`` has 20 slots; slots 12..19 are zeroed
(mode 0x00). ``crc32_core`` is the canonical Unitree algorithm (unitree_legged_sdk);
its serialization-layout correctness against the real msg is validated on the robot
by the preflight CRC roundtrip (a wrong layout silently drops every command).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

N_CMD_SLOTS = 20      # unitree_go/msg/LowCmd.motor_cmd length
N_ACTIVE = 12         # Go2 uses motors 0..11 (FR,FL,RR,RL x hip,thigh,calf)
MODE_SERVO = 0x01     # PMSM servo (enabled): motor runs its PD on q/dq/kp/kd/tau
MODE_PASSIVE = 0x00   # joint passive / disabled
_U32 = 0xFFFFFFFF


@dataclass
class LowCmdFields:
    """The 20-slot motor command arrays, ready to copy onto a LowCmd msg."""

    mode: List[int] = field(default_factory=lambda: [MODE_PASSIVE] * N_CMD_SLOTS)
    q: List[float] = field(default_factory=lambda: [0.0] * N_CMD_SLOTS)
    dq: List[float] = field(default_factory=lambda: [0.0] * N_CMD_SLOTS)
    kp: List[float] = field(default_factory=lambda: [0.0] * N_CMD_SLOTS)
    kd: List[float] = field(default_factory=lambda: [0.0] * N_CMD_SLOTS)
    tau: List[float] = field(default_factory=lambda: [0.0] * N_CMD_SLOTS)


def build_low_cmd_fields(
    targets_isaac: Sequence[float],
    kp: float,
    kd: float,
) -> LowCmdFields:
    """12 joint position targets (SDK/FR-first order) -> 20-slot motor command.

    Active motors get ``mode=0x01, q=target, dq=0, kp, kd, tau=0`` (motor-PD); the 8
    unused slots stay passive/zero. ``targets_isaac`` is already in the SDK index
    order because the LowState adapter advertised FR-first dof_names -> identity.
    """
    t = np.asarray(targets_isaac, dtype=np.float64).reshape(-1)
    if t.shape[0] != N_ACTIVE:
        raise ValueError(f"expected {N_ACTIVE} joint targets, got {t.shape[0]}")
    out = LowCmdFields()
    for i in range(N_ACTIVE):
        out.mode[i] = MODE_SERVO
        out.q[i] = float(t[i])
        out.dq[i] = 0.0
        out.kp[i] = float(kp)
        out.kd[i] = float(kd)
        out.tau[i] = 0.0
    return out


def build_damping_fields(kd: float = 5.0) -> LowCmdFields:
    """A safe damping command: kp=0, pure velocity damping on the active motors.

    With kp=0 the position target is irrelevant, so this brakes the legs smoothly
    toward rest regardless of pose -- the watchdog's fail-safe and the pre-/post-
    control state. Better than mode=0x00 (fully passive) for a standing dog.
    """
    out = LowCmdFields()
    for i in range(N_ACTIVE):
        out.mode[i] = MODE_SERVO
        out.q[i] = 0.0
        out.dq[i] = 0.0
        out.kp[i] = 0.0
        out.kd[i] = float(kd)
        out.tau[i] = 0.0
    return out


def crc32_core(words: Sequence[int]) -> int:
    """Unitree's CRC32 (crc32_core from unitree_legged_sdk) over a uint32 word array.

    Bit-reflected-input variant unique to Unitree -- NOT a stock zlib CRC32; the MCU
    rejects a stock CRC. Ported faithfully with explicit 32-bit masking for Python's
    unbounded ints. The caller serializes the LowCmd struct into the uint32 words.
    """
    crc = 0xFFFFFFFF
    poly = 0x04C11DB7
    for word in words:
        data = int(word) & _U32
        xbit = 0x80000000
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & _U32) ^ poly
            else:
                crc = (crc << 1) & _U32
            if data & xbit:
                crc ^= poly
            xbit >>= 1
    return crc & _U32
