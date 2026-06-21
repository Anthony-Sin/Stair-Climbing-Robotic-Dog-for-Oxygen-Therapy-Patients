"""Host tests for the LowCmd field builder + CRC (no ROS 2 / no robot).

Validates the Go2 LowCmd invariants: mode=0x01 on the 12 active motors / 0x00 on the
8 spares, motor-PD fields (q=target, dq=tau=0, kp/kd as given), the damping fallback,
and that the CRC is deterministic + input-sensitive. The CRC's byte-layout correctness
is validated on hardware by the preflight roundtrip; here we guard the algorithm.

Run: python tests/test_real_lowcmd_builder.py  (or via pytest)
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from real.control.lowcmd_builder import (
    build_low_cmd_fields,
    build_damping_fields,
    crc32_core,
    N_CMD_SLOTS,
    N_ACTIVE,
    MODE_SERVO,
    MODE_PASSIVE,
)


def test_active_motors_servo_spares_passive():
    targets = [0.1 * i for i in range(N_ACTIVE)]
    f = build_low_cmd_fields(targets, kp=40.0, kd=0.5)
    assert len(f.mode) == N_CMD_SLOTS
    assert all(f.mode[i] == MODE_SERVO for i in range(N_ACTIVE))
    assert all(f.mode[i] == MODE_PASSIVE for i in range(N_ACTIVE, N_CMD_SLOTS))
    assert f.q[:N_ACTIVE] == targets
    assert all(f.kp[i] == 40.0 and f.kd[i] == 0.5 for i in range(N_ACTIVE))
    assert all(f.dq[i] == 0.0 and f.tau[i] == 0.0 for i in range(N_ACTIVE))
    # spares fully zeroed
    assert all(f.q[i] == 0.0 and f.kp[i] == 0.0 and f.kd[i] == 0.0 for i in range(N_ACTIVE, N_CMD_SLOTS))


def test_wrong_target_count_raises():
    try:
        build_low_cmd_fields([0.0] * 11, kp=40.0, kd=0.5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for 11 targets")


def test_damping_is_kd_only():
    f = build_damping_fields(kd=5.0)
    assert all(f.mode[i] == MODE_SERVO for i in range(N_ACTIVE))
    assert all(f.kp[i] == 0.0 for i in range(N_ACTIVE))       # kp=0 => pose-independent braking
    assert all(f.kd[i] == 5.0 for i in range(N_ACTIVE))


def test_crc_deterministic_and_sensitive():
    a = crc32_core([0x12345678, 0x00000001, 0xDEADBEEF])
    b = crc32_core([0x12345678, 0x00000001, 0xDEADBEEF])
    assert a == b                       # deterministic
    assert 0 <= a <= 0xFFFFFFFF         # 32-bit
    assert crc32_core([0x12345678]) != crc32_core([0x12345679])   # sensitive to data
    assert crc32_core([0]) != crc32_core([0, 0])                  # sensitive to length


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
