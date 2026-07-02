"""Host-side tests for the real follow-command wire format (no ROS 2 needed).

This pins the contract the ROS 2 publisher (RealRobotController) and the low-level
subscriber (low_level_control_node) share, so they can never silently disagree on
the Float32MultiArray layout.

Run: python tests/test_real_follow_command.py  (or via pytest)
"""
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from real.control.follow_command import FollowCommand, FOLLOW_CMD_LEN


def test_pack_length():
    assert len(FollowCommand().pack()) == FOLLOW_CMD_LEN


def test_stamp_roundtrips_and_publish_override_wins():
    cmd = FollowCommand(vx=0.2, stamp=123.5)
    assert cmd.pack()[-1] == 123.5                       # field rides in the last slot
    assert cmd.pack(stamp=999.0)[-1] == 999.0            # publish-time override wins
    assert FollowCommand.unpack(cmd.pack(stamp=42.0)).stamp == 42.0


def test_legacy_12field_vector_unpacks_unstamped():
    # A pre-stamp 12-length wire vector still decodes (stamp -> 0.0), so a mixed-version
    # bring-up degrades to the old behaviour instead of raising on the 50 Hz hot path.
    legacy = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, math.nan,
              math.nan, math.nan, math.nan, math.nan]
    out = FollowCommand.unpack(legacy)
    assert out.vx == 0.1 and out.person_detected is True
    assert out.stamp == 0.0


def test_full_roundtrip():
    cmd = FollowCommand(
        vx=0.42,
        wz=-0.13,
        yaw_err=0.05,
        stairs_detected=True,
        stairs_action_active=False,
        hold=True,
        person_detected=True,
        gap_m=1.37,
        person_bbox=(0.1, 0.2, 0.6, 0.9),
    )
    out = FollowCommand.unpack(cmd.pack())
    assert out == cmd


def test_none_optionals_roundtrip_via_nan():
    cmd = FollowCommand(vx=0.1, gap_m=None, person_bbox=None)
    packed = cmd.pack()
    # gap + 4 bbox slots are NaN on the wire ...
    assert math.isnan(packed[7])
    assert all(math.isnan(v) for v in packed[8:12])
    # ... and decode back to None (not NaN floats).
    out = FollowCommand.unpack(packed)
    assert out.gap_m is None
    assert out.person_bbox is None


def test_bool_coercion_from_floats():
    out = FollowCommand.unpack([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, math.nan,
                                math.nan, math.nan, math.nan, math.nan])
    assert out.stairs_detected is True
    assert out.stairs_action_active is False
    assert out.hold is True
    assert out.person_detected is False


def test_from_move_args_matches_core_main_call():
    # The exact kwargs core/main.py:1336 passes to controller.move(...).
    cmd = FollowCommand.from_move_args(
        0.6, 0.0,
        yaw_err=0.2,
        stairs_detected=True,
        person_bbox=[0.0, 0.0, 0.5, 0.5],
        stairs_action_active=True,
        hold=False,
        person_detected=True,
        gap_m=0.8,
    )
    assert cmd.vx == 0.6 and cmd.wz == 0.0 and cmd.yaw_err == 0.2
    assert cmd.stairs_action_active is True and cmd.gap_m == 0.8
    assert cmd.person_bbox == (0.0, 0.0, 0.5, 0.5)


def test_unpack_rejects_short_vector():
    try:
        FollowCommand.unpack([0.0, 1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError on short wire vector")


if __name__ == "__main__":
    test_pack_length()
    test_stamp_roundtrips_and_publish_override_wins()
    test_legacy_12field_vector_unpacks_unstamped()
    test_full_roundtrip()
    test_none_optionals_roundtrip_via_nan()
    test_bool_coercion_from_floats()
    test_from_move_args_matches_core_main_call()
    test_unpack_rejects_short_vector()
    print("OK")
