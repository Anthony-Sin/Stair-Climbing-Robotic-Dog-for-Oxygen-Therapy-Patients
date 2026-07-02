"""ROS 2 publisher that implements the ``core.main`` robot-controller contract.

This is the seam between the hardware-agnostic controller in ``core/`` and the
native-ROS 2 Go2 stack. ``core/main.py`` builds it (via ``_build_robot_controller``
when ``--ros2`` is set) and drives it with the SAME calls it makes in sim:
``move(vx, 0, wz, ...)``, ``stop()``, ``initialize()``, ``shutdown()``, ``is_ready()``.

It does NOT touch joints and does NOT touch ``unitree_sdk2``. It only publishes:
  * the high-level follow command bundle on ``/go2/cmd_custom`` (Float32MultiArray,
    layout owned by ``follow_command``), and
  * the companion depth frame (resized to the 106x60 policy size) on
    ``/go2/camera/depth`` (sensor_msgs/Image, 16UC1).

The 50 Hz policy loop + ``/lowcmd`` writing live in ``low_level_control_node``
(process B); standing up and releasing sport mode live in ``sport_startup_node``.
Keeping this class a pure publisher means the heavy GPU vision process (process A)
never blocks the real-time control process.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

from real.control.follow_command import (
    FollowCommand,
    FOLLOW_CMD_TOPIC,
    DEPTH_TOPIC,
)
from real.ros2.qos import sensor_qos

LOGGER = logging.getLogger("cable.real.robot_controller")

# The depth size the parkour person-mask + stair detector consume downstream.
_DEPTH_W, _DEPTH_H = 106, 60
_DEPTH_FRAME_ID = "front_d435_depth"


class RealRobotController:
    """Publish core/'s follow command + depth onto ROS 2; no joints, no SDK."""

    def __init__(self, args) -> None:
        self._args = args
        self._node: Optional[Node] = None
        self._cmd_pub = None
        self._depth_pub = None
        self._owns_rclpy = False  # did WE call rclpy.init()? (so shutdown is symmetric)

    # ------------------------------------------------------------------ lifecycle
    def initialize(self) -> bool:
        """Bring up the rclpy context + publishers. Returns True on success."""
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy = True
            self._node = Node("real_robot_controller")
            self._cmd_pub = self._node.create_publisher(
                Float32MultiArray, FOLLOW_CMD_TOPIC, sensor_qos()
            )
            self._depth_pub = self._node.create_publisher(
                Image, DEPTH_TOPIC, sensor_qos()
            )
            LOGGER.info(
                "RealRobotController up: publishing %s + %s",
                FOLLOW_CMD_TOPIC,
                DEPTH_TOPIC,
            )
            return True
        except Exception as exc:  # pragma: no cover - hardware/ROS path
            LOGGER.error("RealRobotController initialize failed: %s", exc)
            return False

    def is_ready(self) -> bool:
        return self._node is not None and rclpy.ok()

    # -------------------------------------------------------------------- command
    def move(
        self,
        trans_x: float,
        trans_y: float,
        rotation: float,
        stairs_detected: bool = False,
        yaw_err: float = 0.0,
        **kwargs,
    ) -> bool:
        """Publish one follow command. Mirrors RobotController.move()'s signature.

        ``trans_y`` is ignored (the follow controller never commands lateral). The
        extra perception kwargs (person_bbox, stairs_action_active, hold,
        person_detected, gap_m, depth_img) match exactly what ``core.main`` passes.
        """
        if not self.is_ready():
            LOGGER.warning("move() before initialize(); dropping command")
            return False

        cmd = FollowCommand.from_move_args(
            trans_x,
            rotation,
            yaw_err=yaw_err,
            stairs_detected=stairs_detected,
            stairs_action_active=bool(kwargs.get("stairs_action_active", False)),
            hold=bool(kwargs.get("hold", False)),
            person_detected=bool(kwargs.get("person_detected", False)),
            gap_m=kwargs.get("gap_m"),
            person_bbox=kwargs.get("person_bbox"),
        )
        # Stamp at publish with THIS node's clock (system time on real HW, use_sim_time
        # off) so the 50 Hz consumer can age-gate: a dead vision process stops refreshing
        # the stamp and the control node stops executing the last command. See
        # follow_command wire layout + low_level_control_node staleness gate.
        stamp = self._node.get_clock().now().nanoseconds * 1e-9
        self._cmd_pub.publish(Float32MultiArray(data=cmd.pack(stamp=stamp)))

        depth_img = kwargs.get("depth_img")
        if depth_img is not None:
            self._publish_depth(depth_img)
        return True

    def stop(self) -> bool:
        """Command a full stop (zero velocity)."""
        return self.move(0.0, 0.0, 0.0)

    def shutdown(self) -> bool:
        """Send a final stop, tear down the node, and release rclpy if we own it."""
        try:
            if self.is_ready():
                self.stop()
            if self._node is not None:
                self._node.destroy_node()
                self._node = None
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
                self._owns_rclpy = False
            return True
        except Exception as exc:  # pragma: no cover - hardware/ROS path
            LOGGER.error("RealRobotController shutdown error: %s", exc)
            return False

    # -------------------------------------------------------------------- helpers
    def _publish_depth(self, depth_img) -> None:
        """Resize raw depth to the 106x60 policy size and publish as 16UC1 Image.

        Person-masking is deliberately done downstream in the low-level process
        (``depth_to_policy``) using the bbox carried in the follow command, so the
        depth preprocessing lives in exactly one place next to its consumer.
        """
        try:
            arr = np.asarray(depth_img)
            if arr.size == 0:
                return
            if arr.shape[:2] != (_DEPTH_H, _DEPTH_W):
                import cv2  # lazy: only process A (vision) has cv2

                arr = cv2.resize(
                    arr, (_DEPTH_W, _DEPTH_H), interpolation=cv2.INTER_LINEAR
                )
            arr = np.ascontiguousarray(arr.astype(np.uint16))
            msg = Image()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = _DEPTH_FRAME_ID
            msg.height, msg.width = int(arr.shape[0]), int(arr.shape[1])
            msg.encoding = "16UC1"
            msg.is_bigendian = 0
            msg.step = int(arr.shape[1]) * 2
            msg.data = arr.tobytes()
            self._depth_pub.publish(msg)
        except Exception as exc:  # pragma: no cover - hardware/ROS path
            LOGGER.error("depth publish failed: %s", exc)
