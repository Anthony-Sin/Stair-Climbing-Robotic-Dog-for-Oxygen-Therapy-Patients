"""The high-level follow command core/ produces, and its on-the-wire layout.

``core/main.py`` drives the robot through ``controller.move(vx, 0, wz, ...)`` -- a
small bundle of a forward/yaw velocity plus a few perception flags. In the native
ROS 2 port that bundle is published by ``RealRobotController`` (process A: core +
vision) and consumed by ``low_level_control_node`` (process B: the 50 Hz policy
loop). This module is the SINGLE SOURCE OF TRUTH for that bundle and how it packs
onto the wire, so the publisher and subscriber can never silently disagree.

Wire format: a fixed-length ``std_msgs/Float32MultiArray`` (the typed-msg-free
fallback the plan sanctions for v1 -- no custom ``go2_msgs`` colcon build needed to
bring the robot up). Booleans ride as 0.0/1.0; absent optionals (gap, bbox) ride as
NaN. The final slot carries the publisher's clock stamp (seconds) so the 50 Hz
consumer can age-gate a stale command instead of executing the last one forever (the
vision process dying mid-climb must NOT leave the dog walking blind). Because it is
plain floats, the pack/unpack is pure Python and unit-tested on a bare host
(tests/test_real_follow_command.py) with no ROS 2 present.

Hardening note: once the stack is up, this can be swapped for a typed
``go2_msgs/msg/FollowCommand`` with zero logic change -- only the (de)serialization
boundary in ``real_robot_controller`` / ``low_level_control_node`` moves; the field
set and meaning stay exactly as defined here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Topics the command + its companion depth frame travel on.
FOLLOW_CMD_TOPIC = "/go2/cmd_custom"
DEPTH_TOPIC = "/go2/camera/depth"

# Fixed Float32MultiArray layout. Index -> field. Keep in lockstep with pack()/unpack().
#   0 vx                  forward velocity command (m/s)
#   1 wz                  yaw-rate command (rad/s)   [parkour self-steers via yaw_err; kept for completeness]
#   2 yaw_err             bearing/heading error to the person (rad)
#   3 stairs_detected     stair policy is preparing (0/1)
#   4 stairs_action_active near-field stair engagement gate (0/1) -> drives the WALK->CLIMB handoff
#   5 hold                stance-hold request (0/1)
#   6 person_detected     person in frame this tick (0/1)
#   7 gap_m               tracked person distance (m), NaN if unknown
#   8..11 bbox            normalized person bbox [x1,y1,x2,y2] in [0,1], all-NaN if no bbox
#   12 stamp             publisher clock time (s) at publish; 0.0/NaN/absent => unstamped
FOLLOW_CMD_LEN = 13   # canonical packed length: 12 core fields + trailing stamp
_CORE_LEN = 12        # minimum decodable length (stamp is optional: fwd/back compat)
_IDX_GAP = 7
_IDX_BBOX = 8
_IDX_STAMP = 12


@dataclass(frozen=True)
class FollowCommand:
    """The high-level command + perception flags for one control tick."""

    vx: float = 0.0
    wz: float = 0.0
    yaw_err: float = 0.0
    stairs_detected: bool = False
    stairs_action_active: bool = False
    hold: bool = False
    person_detected: bool = False
    gap_m: Optional[float] = None
    person_bbox: Optional[Tuple[float, float, float, float]] = None
    # Publisher clock time (s) stamped at publish. 0.0 means "unstamped" (default-
    # constructed command, or a legacy 12-field wire vector); the consumer's staleness
    # gate treats an unstamped command conservatively. NOT part of the control logic --
    # a pure transport/liveness field.
    stamp: float = 0.0

    @classmethod
    def from_move_args(
        cls,
        trans_x: float,
        rotation: float,
        *,
        yaw_err: float = 0.0,
        stairs_detected: bool = False,
        stairs_action_active: bool = False,
        hold: bool = False,
        person_detected: bool = False,
        gap_m: Optional[float] = None,
        person_bbox: Optional[Sequence[float]] = None,
    ) -> "FollowCommand":
        """Build from the exact kwargs ``core.main`` passes to ``controller.move``."""
        bbox: Optional[Tuple[float, float, float, float]] = None
        if person_bbox is not None:
            b = tuple(float(v) for v in person_bbox)
            if len(b) == 4:
                bbox = b  # type: ignore[assignment]
        return cls(
            vx=float(trans_x),
            wz=float(rotation),
            yaw_err=float(yaw_err),
            stairs_detected=bool(stairs_detected),
            stairs_action_active=bool(stairs_action_active),
            hold=bool(hold),
            person_detected=bool(person_detected),
            gap_m=None if gap_m is None else float(gap_m),
            person_bbox=bbox,
        )

    def pack(self, stamp: Optional[float] = None) -> List[float]:
        """Serialize to the fixed ``FOLLOW_CMD_LEN`` float vector.

        ``stamp`` overrides the field so the ROS publisher can stamp with its live clock
        at the moment of publish (``cmd.pack(stamp=node.get_clock().now()...)``) without
        rebuilding the frozen command; omit it and the instance's own ``stamp`` is used.
        """
        data = [0.0] * FOLLOW_CMD_LEN
        data[0] = float(self.vx)
        data[1] = float(self.wz)
        data[2] = float(self.yaw_err)
        data[3] = 1.0 if self.stairs_detected else 0.0
        data[4] = 1.0 if self.stairs_action_active else 0.0
        data[5] = 1.0 if self.hold else 0.0
        data[6] = 1.0 if self.person_detected else 0.0
        data[_IDX_GAP] = math.nan if self.gap_m is None else float(self.gap_m)
        if self.person_bbox is None:
            data[_IDX_BBOX:_IDX_BBOX + 4] = [math.nan] * 4
        else:
            data[_IDX_BBOX:_IDX_BBOX + 4] = [float(v) for v in self.person_bbox]
        data[_IDX_STAMP] = float(self.stamp if stamp is None else stamp)
        return data

    @classmethod
    def unpack(cls, data: Sequence[float]) -> "FollowCommand":
        """Deserialize from a ``FOLLOW_CMD_LEN`` float vector (the publisher's pack()).

        The trailing ``stamp`` is optional: a ``_CORE_LEN`` (12) legacy vector still
        decodes (stamp -> 0.0, i.e. "unstamped"), so a mixed-version bring-up degrades
        to the pre-stamp behaviour rather than raising on the hot path.
        """
        if len(data) < _CORE_LEN:
            raise ValueError(
                f"FollowCommand wire vector too short: {len(data)} < {_CORE_LEN}"
            )
        gap = float(data[_IDX_GAP])
        bbox_vals = [float(v) for v in data[_IDX_BBOX:_IDX_BBOX + 4]]
        bbox = None if any(math.isnan(v) for v in bbox_vals) else tuple(bbox_vals)
        stamp = float(data[_IDX_STAMP]) if len(data) > _IDX_STAMP else 0.0
        if math.isnan(stamp):
            stamp = 0.0
        return cls(
            vx=float(data[0]),
            wz=float(data[1]),
            yaw_err=float(data[2]),
            stairs_detected=data[3] != 0.0,
            stairs_action_active=data[4] != 0.0,
            hold=data[5] != 0.0,
            person_detected=data[6] != 0.0,
            gap_m=None if math.isnan(gap) else gap,
            person_bbox=bbox,  # type: ignore[arg-type]
            stamp=stamp,
        )
