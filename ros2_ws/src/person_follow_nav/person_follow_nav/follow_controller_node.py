import json
import math
import socket
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from person_follow_nav.ecs_logging_utils import (
    build_ecs_extra,
    get_ecs_logger,
    setup_ecs_file_logging,
)
from person_follow_nav.debug_trace_logger import CsvTraceLogger, DebugTraceLogger


@dataclass
class TargetPacket:
    timestamp: float
    valid: bool
    track_id: Optional[int]
    confidence: float
    x_base_m: Optional[float]
    y_base_m: Optional[float]
    valid_reason: Optional[str]
    block_reason: Optional[str]
    target_hold_active: Optional[bool]
    matched_visual_lock: Optional[bool]
    recent_visual_lock: Optional[bool]
    motion_lock_ready: Optional[bool]
    motion_lock_streak: Optional[int]
    motion_lock_frames: Optional[int]
    reacquire_active: Optional[bool]
    depth_valid: Optional[bool]
    depth_method: Optional[str]
    depth_distance_m: Optional[float]
    center_x: Optional[float]
    bbox_center_x: Optional[float]
    received_monotonic: float


@dataclass
class FilteredTarget:
    track_id: Optional[int]
    confidence: float
    x_base_m: float
    y_base_m: float


@dataclass
class OdomState:
    x: float
    y: float
    yaw: float


@dataclass
class GoalSignature:
    x: float
    y: float
    yaw: float


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_delta(a: float, b: float) -> float:
    return normalize_angle(a - b)


TARGET_VALID_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class PersonFollowController(Node):
    def __init__(self) -> None:
        super().__init__("person_follow_nav")

        self.declare_parameter("udp_host", "0.0.0.0")
        self.declare_parameter("udp_port", 41234)
        self.declare_parameter("target_timeout_sec", 0.8)
        self.declare_parameter("target_hold_sec", 0.7)
        self.declare_parameter("target_stale_grace_sec", 0.25)
        self.declare_parameter("render_backlog_age_sec", 0.35)
        self.declare_parameter("ema_alpha", 0.6)
        self.declare_parameter("desired_distance", 0.35)
        self.declare_parameter("update_rate_hz", 30.0)
        self.declare_parameter("max_goal_rate_hz", 12.0)
        self.declare_parameter("position_change_threshold_m", 0.02)
        self.declare_parameter("yaw_change_threshold_rad", 0.03)
        self.declare_parameter("path_step_m", 0.10)
        self.declare_parameter("follow_tolerance_m", 0.40)
        self.declare_parameter("bearing_hold_tolerance_rad", 0.15)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("target_pose_topic", "person_follow/target")
        self.declare_parameter("path_topic", "person_follow/path")
        self.declare_parameter("target_valid_topic", "person_follow/target_valid")
        self.declare_parameter("controller_id", "FollowPath")
        self.declare_parameter("goal_checker_id", "goal_checker")
        self.declare_parameter("ecs_log_dir", "logs")
        self.declare_parameter("enable_ecs_logging", False)
        # DEBUG-TRACE REMOVE-ME: sidecar structured trace folder.
        self.declare_parameter("debug_trace_dir", "")

        self._target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self._target_hold_sec = float(self.get_parameter("target_hold_sec").value)
        self._target_stale_grace_sec = max(
            0.0, float(self.get_parameter("target_stale_grace_sec").value)
        )
        self._render_backlog_age_sec = max(
            0.0, float(self.get_parameter("render_backlog_age_sec").value)
        )
        self._ema_alpha = float(self.get_parameter("ema_alpha").value)
        self._desired_distance = float(self.get_parameter("desired_distance").value)
        self._goal_rate_period = 1.0 / max(1e-3, float(self.get_parameter("max_goal_rate_hz").value))
        self._position_change_threshold_m = float(self.get_parameter("position_change_threshold_m").value)
        self._yaw_change_threshold_rad = float(self.get_parameter("yaw_change_threshold_rad").value)
        self._path_step_m = float(self.get_parameter("path_step_m").value)
        self._follow_tolerance_m = float(self.get_parameter("follow_tolerance_m").value)
        self._bearing_hold_tolerance_rad = float(
            self.get_parameter("bearing_hold_tolerance_rad").value
        )
        self._controller_id = self.get_parameter("controller_id").value
        self._goal_checker_id = self.get_parameter("goal_checker_id").value
        self._ecs_log_path, _ = setup_ecs_file_logging(
            service_name="person-follow-nav",
            event_dataset="cable.sidecar",
            log_dir=self.get_parameter("ecs_log_dir").value,
            file_prefix="person_follow_nav_ecs",
            enabled=bool(self.get_parameter("enable_ecs_logging").value),
        )
        self._ecs_logger = get_ecs_logger("sidecar.follow")
        # DEBUG-TRACE REMOVE-ME: capture target validity and staleness transitions in JSONL.
        self._debug_trace = DebugTraceLogger(
            trace_dir=str(self.get_parameter("debug_trace_dir").value),
            filename="sidecar_follow_trace.jsonl",
            source="sidecar.follow",
        )
        self._target_error_trace = CsvTraceLogger(
            trace_dir=str(self.get_parameter("debug_trace_dir").value),
            filename="sidecar_follow_target_errors.csv",
            fieldnames=[
                "ts_unix",
                "ts_monotonic",
                "measurement_valid",
                "reason",
                "holding_last_target",
                "track_id",
                "distance_error_m",
                "lateral_error_m",
                "packet_age_sec",
                "source_age_sec",
            ],
        )

        self._target_pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter("target_pose_topic").value, 10
        )
        self._path_pub = self.create_publisher(Path, self.get_parameter("path_topic").value, 10)
        self._target_valid_pub = self.create_publisher(
            Bool, self.get_parameter("target_valid_topic").value, TARGET_VALID_QOS
        )
        self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self._on_odom,
            10,
        )

        self._follow_path_client = ActionClient(self, FollowPath, "follow_path")
        self._goal_handle = None
        self._last_goal_signature: Optional[GoalSignature] = None
        self._last_goal_send_ts = 0.0

        self._odom_state: Optional[OdomState] = None
        self._latest_packet: Optional[TargetPacket] = None
        self._filtered_target: Optional[FilteredTarget] = None
        self._last_target_valid = False
        self._last_valid_target_ts = 0.0
        self._last_target_packet_signature: Optional[tuple[object, ...]] = None
        self._last_target_packet_log_ts = 0.0

        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.bind(
            (self.get_parameter("udp_host").value, int(self.get_parameter("udp_port").value))
        )
        self._udp_socket.setblocking(False)
        self.get_logger().info(
            f"Listening for vision targets on UDP "
            f"{self.get_parameter('udp_host').value}:{self.get_parameter('udp_port').value}"
        )
        if self._ecs_log_path is not None:
            self.get_logger().info(f"ECS analytics log: {self._ecs_log_path}")
        self._ecs_logger.info(
            "Person follow Nav session start",
            extra=build_ecs_extra(
                component="sidecar.follow",
                action="session_start",
                cable={
                    "target": {
                        "udp_host": self.get_parameter("udp_host").value,
                        "udp_port": int(self.get_parameter("udp_port").value),
                        "timeout_sec": self._target_timeout_sec,
                        "hold_sec": self._target_hold_sec,
                        "stale_grace_sec": self._target_stale_grace_sec,
                        "render_backlog_age_sec": self._render_backlog_age_sec,
                    },
                    "follow": {"desired_distance_m": self._desired_distance},
                },
            ),
        )
        self._debug_trace.log(
            "session_start",
            udp_host=str(self.get_parameter("udp_host").value),
            udp_port=int(self.get_parameter("udp_port").value),
            target_timeout_sec=float(self._target_timeout_sec),
            target_hold_sec=float(self._target_hold_sec),
            target_stale_grace_sec=float(self._target_stale_grace_sec),
            render_backlog_age_sec=float(self._render_backlog_age_sec),
            desired_distance_m=float(self._desired_distance),
        )

        update_period = 1.0 / max(1e-3, float(self.get_parameter("update_rate_hz").value))
        self.create_timer(update_period, self._on_timer)

    def _on_odom(self, msg: Odometry) -> None:
        self._odom_state = OdomState(
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=quaternion_to_yaw(
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            ),
        )

    def _on_timer(self) -> None:
        self._pump_udp_packets()
        packet, packet_reason, packet_age_sec, source_age_sec = self._evaluate_target_packet()
        holding_last_target = packet is None and self._can_hold_last_target()
        self._log_target_packet_state(
            packet_reason,
            self._latest_packet if packet is None else packet,
            holding_last_target=holding_last_target,
            packet_age_sec=packet_age_sec,
            source_age_sec=source_age_sec,
        )
        if packet is None:
            self._log_target_error_sample(
                reason=packet_reason,
                target=None,
                packet=self._latest_packet,
                holding_last_target=holding_last_target,
                packet_age_sec=packet_age_sec,
                source_age_sec=source_age_sec,
            )
            if holding_last_target:
                held_target = self._filtered_target
                if held_target is None:
                    return
                self._publish_target_valid(
                    True,
                    reason=packet_reason,
                    packet=self._latest_packet,
                    holding_last_target=True,
                )
                self._publish_target_pose(held_target)

                if self._odom_state is None:
                    return

                if self._is_within_follow_tolerance(held_target):
                    self._publish_empty_path()
                    self._cancel_goal()
                    return

                path, signature = self._build_follow_path(held_target, self._odom_state)
                self._path_pub.publish(path)

                now = time.monotonic()
                if self._should_send_goal(signature, now):
                    self._send_goal(path, signature, now)
                return

            self._filtered_target = None
            self._publish_target_valid(False, reason=packet_reason, packet=self._latest_packet)
            self._publish_empty_path()
            self._cancel_goal()
            return

        filtered_target = self._update_filtered_target(packet)
        self._last_valid_target_ts = time.monotonic()
        self._log_target_error_sample(
            reason=packet_reason,
            target=filtered_target,
            packet=packet,
            holding_last_target=False,
            packet_age_sec=packet_age_sec,
            source_age_sec=source_age_sec,
        )
        self._publish_target_valid(True, reason=packet_reason, packet=packet)
        self._publish_target_pose(filtered_target)

        if self._odom_state is None:
            return

        if self._is_within_follow_tolerance(filtered_target):
            self._publish_empty_path()
            self._cancel_goal()
            return

        path, signature = self._build_follow_path(filtered_target, self._odom_state)
        self._path_pub.publish(path)

        now = time.monotonic()
        if self._should_send_goal(signature, now):
            self._send_goal(path, signature, now)

    def _pump_udp_packets(self) -> None:
        while True:
            try:
                payload_bytes, _ = self._udp_socket.recvfrom(65535)
            except BlockingIOError:
                return
            except OSError as exc:
                self.get_logger().warning(f"UDP receive error: {exc}")
                self._ecs_logger.error(
                    "UDP receive error",
                    extra=build_ecs_extra(
                        component="sidecar.follow",
                        action="udp_receive_failed",
                        cable={"error": {"message": str(exc)}},
                    ),
                )
                return

            packet = self._parse_packet(payload_bytes)
            if packet is not None:
                self._latest_packet = packet
                self._ecs_logger.debug(
                    "UDP target packet received",
                    extra=build_ecs_extra(
                        component="sidecar.follow",
                        action="udp_packet_received",
                        cable={
                            "target": {
                                "track_id": packet.track_id,
                                "valid": packet.valid,
                                "confidence": packet.confidence,
                                "x_base_m": packet.x_base_m,
                                "y_base_m": packet.y_base_m,
                                "valid_reason": packet.valid_reason,
                                "block_reason": packet.block_reason,
                                "target_hold_active": packet.target_hold_active,
                                "matched_visual_lock": packet.matched_visual_lock,
                                "recent_visual_lock": packet.recent_visual_lock,
                                "motion_lock_ready": packet.motion_lock_ready,
                                "motion_lock_streak": packet.motion_lock_streak,
                                "motion_lock_frames": packet.motion_lock_frames,
                                "reacquire_active": packet.reacquire_active,
                                "depth_valid": packet.depth_valid,
                                "depth_method": packet.depth_method,
                                "depth_distance_m": packet.depth_distance_m,
                                "center_x": packet.center_x,
                                "bbox_center_x": packet.bbox_center_x,
                            }
                        },
                    ),
                )

    def _parse_packet(self, payload_bytes: bytes) -> Optional[TargetPacket]:
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.get_logger().warning("Ignoring malformed UDP target packet")
            self._ecs_logger.warning(
                "Malformed UDP target packet ignored",
                extra=build_ecs_extra(
                    component="sidecar.follow",
                    action="udp_packet_malformed",
                ),
            )
            return None

        def optional_float(value) -> Optional[float]:
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) else None

        def optional_bool(value) -> Optional[bool]:
            if value is None:
                return None
            return bool(value)

        def optional_int(value) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def optional_text(value) -> Optional[str]:
            if value is None:
                return None
            text = str(value)
            return text or None

        track_id = payload.get("track_id")
        if track_id is not None:
            track_id = int(track_id)

        return TargetPacket(
            timestamp=float(payload.get("timestamp", 0.0)),
            valid=bool(payload.get("valid", False)),
            track_id=track_id,
            confidence=float(payload.get("confidence", 0.0)),
            x_base_m=optional_float(payload.get("x_base_m")),
            y_base_m=optional_float(payload.get("y_base_m")),
            valid_reason=optional_text(payload.get("valid_reason")),
            block_reason=optional_text(payload.get("block_reason")),
            target_hold_active=optional_bool(payload.get("target_hold_active")),
            matched_visual_lock=optional_bool(payload.get("matched_visual_lock")),
            recent_visual_lock=optional_bool(payload.get("recent_visual_lock")),
            motion_lock_ready=optional_bool(payload.get("motion_lock_ready")),
            motion_lock_streak=optional_int(payload.get("motion_lock_streak")),
            motion_lock_frames=optional_int(payload.get("motion_lock_frames")),
            reacquire_active=optional_bool(payload.get("reacquire_active")),
            depth_valid=optional_bool(payload.get("depth_valid")),
            depth_method=optional_text(payload.get("depth_method")),
            depth_distance_m=optional_float(payload.get("depth_distance_m")),
            center_x=optional_float(payload.get("center_x")),
            bbox_center_x=optional_float(payload.get("bbox_center_x")),
            received_monotonic=time.monotonic(),
        )

    def _packet_source_age_sec(self, packet: TargetPacket) -> Optional[float]:
        if packet.timestamp <= 0.0:
            return None
        age_sec = time.time() - packet.timestamp
        if not math.isfinite(age_sec) or age_sec < 0.0:
            return None
        return float(age_sec)

    def _should_apply_stale_grace(
        self,
        packet: TargetPacket,
        packet_age_sec: float,
        source_age_sec: Optional[float],
    ) -> bool:
        if self._target_stale_grace_sec <= 0.0:
            return False
        if packet_age_sec > (self._target_timeout_sec + self._target_stale_grace_sec):
            return False
        if not packet.valid:
            return False
        if packet.valid_reason != "live_target":
            return False
        if packet.block_reason not in (None, ""):
            return False
        if packet.reacquire_active is True:
            return False
        if packet.motion_lock_ready is False:
            return False
        if source_age_sec is None or source_age_sec < self._render_backlog_age_sec:
            return False
        return True

    def _evaluate_target_packet(
        self,
    ) -> tuple[Optional[TargetPacket], str, Optional[float], Optional[float]]:
        packet = self._latest_packet
        if packet is None:
            return None, "no_packet", None, None

        packet_age_sec = time.monotonic() - packet.received_monotonic
        source_age_sec = self._packet_source_age_sec(packet)
        if packet_age_sec > self._target_timeout_sec:
            if self._should_apply_stale_grace(packet, packet_age_sec, source_age_sec):
                return packet, "packet_stale_grace_live_target", packet_age_sec, source_age_sec
            return None, "packet_stale", packet_age_sec, source_age_sec

        if not packet.valid:
            return None, "packet_marked_invalid", packet_age_sec, source_age_sec

        if packet.x_base_m is None or packet.y_base_m is None:
            return None, "packet_missing_coordinates", packet_age_sec, source_age_sec

        return packet, "packet_valid", packet_age_sec, source_age_sec

    def _can_hold_last_target(self) -> bool:
        return (
            self._filtered_target is not None
            and self._last_valid_target_ts > 0.0
            and (time.monotonic() - self._last_valid_target_ts) <= self._target_hold_sec
        )

    def _log_target_packet_state(
        self,
        reason: str,
        packet: Optional[TargetPacket],
        holding_last_target: bool,
        packet_age_sec: Optional[float],
        source_age_sec: Optional[float],
    ) -> None:
        packet_state = (
            "holding_last_target"
            if holding_last_target
            else (
                "live_packet"
                if reason in ("packet_valid", "packet_stale_grace_live_target")
                else "dropping_target"
            )
        )
        signature = (
            packet_state,
            reason,
            None if packet is None else packet.track_id,
            None if packet is None else packet.valid_reason,
            None if packet is None else packet.block_reason,
            bool(holding_last_target),
        )
        now = time.monotonic()
        if (
            signature == self._last_target_packet_signature
            and (now - self._last_target_packet_log_ts) < 1.0
        ):
            return

        packet_age_text = "n/a" if packet_age_sec is None else f"{packet_age_sec:.3f}"
        source_age_text = "n/a" if source_age_sec is None else f"{source_age_sec:.3f}"
        self.get_logger().info(
            "Target packet "
            f"state={packet_state} reason={reason} age={packet_age_text}s "
            f"source_age={source_age_text}s "
            f"track_id={None if packet is None else packet.track_id} "
            f"valid={None if packet is None else packet.valid} "
            f"upstream_valid_reason={None if packet is None else packet.valid_reason} "
            f"upstream_block_reason={None if packet is None else packet.block_reason} "
            f"x={None if packet is None else packet.x_base_m} "
            f"y={None if packet is None else packet.y_base_m} "
            f"hold={holding_last_target}"
        )
        self._ecs_logger.info(
            "Target packet state",
            extra=build_ecs_extra(
                component="sidecar.follow",
                action="target_packet_state",
                cable={
                    "target": {
                        "packet_state": packet_state,
                        "reason": reason,
                        "packet_age_sec": packet_age_sec,
                        "source_age_sec": source_age_sec,
                        "holding_last_target": bool(holding_last_target),
                        "track_id": None if packet is None else packet.track_id,
                        "valid": None if packet is None else bool(packet.valid),
                        "confidence": None if packet is None else packet.confidence,
                        "x_base_m": None if packet is None else packet.x_base_m,
                        "y_base_m": None if packet is None else packet.y_base_m,
                        "upstream_valid_reason": None if packet is None else packet.valid_reason,
                        "upstream_block_reason": None if packet is None else packet.block_reason,
                        "target_hold_active": None if packet is None else packet.target_hold_active,
                        "matched_visual_lock": None if packet is None else packet.matched_visual_lock,
                        "recent_visual_lock": None if packet is None else packet.recent_visual_lock,
                        "motion_lock_ready": None if packet is None else packet.motion_lock_ready,
                        "motion_lock_streak": None if packet is None else packet.motion_lock_streak,
                        "motion_lock_frames": None if packet is None else packet.motion_lock_frames,
                        "reacquire_active": None if packet is None else packet.reacquire_active,
                        "depth_valid": None if packet is None else packet.depth_valid,
                        "depth_method": None if packet is None else packet.depth_method,
                        "depth_distance_m": None if packet is None else packet.depth_distance_m,
                        "center_x": None if packet is None else packet.center_x,
                        "bbox_center_x": None if packet is None else packet.bbox_center_x,
                    }
                },
            ),
        )
        # DEBUG-TRACE REMOVE-ME: structured state transitions for packet validity and staleness.
        self._debug_trace.log(
            "target_packet_state",
            packet_state=packet_state,
            reason=reason,
            packet_age_sec=packet_age_sec,
            source_age_sec=source_age_sec,
            holding_last_target=bool(holding_last_target),
            track_id=None if packet is None else packet.track_id,
            valid=None if packet is None else bool(packet.valid),
            upstream_valid_reason=None if packet is None else packet.valid_reason,
            upstream_block_reason=None if packet is None else packet.block_reason,
            x_base_m=None if packet is None else packet.x_base_m,
            y_base_m=None if packet is None else packet.y_base_m,
        )
        self._last_target_packet_signature = signature
        self._last_target_packet_log_ts = now

    def _update_filtered_target(self, packet: TargetPacket) -> FilteredTarget:
        if (
            self._filtered_target is None
            or self._filtered_target.track_id != packet.track_id
        ):
            self._filtered_target = FilteredTarget(
                track_id=packet.track_id,
                confidence=packet.confidence,
                x_base_m=float(packet.x_base_m),
                y_base_m=float(packet.y_base_m),
            )
            return self._filtered_target

        alpha = min(max(self._ema_alpha, 0.0), 1.0)
        self._filtered_target.x_base_m = alpha * float(packet.x_base_m) + (1.0 - alpha) * self._filtered_target.x_base_m
        self._filtered_target.y_base_m = alpha * float(packet.y_base_m) + (1.0 - alpha) * self._filtered_target.y_base_m
        self._filtered_target.confidence = packet.confidence
        return self._filtered_target

    def _publish_target_valid(
        self,
        is_valid: bool,
        reason: Optional[str] = None,
        packet: Optional[TargetPacket] = None,
        holding_last_target: bool = False,
    ) -> None:
        self._target_valid_pub.publish(Bool(data=is_valid))
        if self._last_target_valid == is_valid:
            return
        self._last_target_valid = is_valid
        self._ecs_logger.info(
            "Target validity changed",
            extra=build_ecs_extra(
                component="sidecar.follow",
                action="target_valid_change",
                cable={
                    "target": {
                        "valid": bool(is_valid),
                        "reason": reason,
                        "holding_last_target": bool(holding_last_target),
                        "track_id": None if packet is None else packet.track_id,
                        "upstream_valid_reason": None if packet is None else packet.valid_reason,
                        "upstream_block_reason": None if packet is None else packet.block_reason,
                    }
                },
            ),
        )
        # DEBUG-TRACE REMOVE-ME: explicit valid/invalid transitions with reasons.
        self._debug_trace.log(
            "target_valid_change",
            valid=bool(is_valid),
            reason=reason,
            holding_last_target=bool(holding_last_target),
            track_id=None if packet is None else packet.track_id,
            upstream_valid_reason=None if packet is None else packet.valid_reason,
            upstream_block_reason=None if packet is None else packet.block_reason,
        )

    def _publish_target_pose(self, target: FilteredTarget) -> None:
        target_yaw = math.atan2(target.y_base_m, target.x_base_m)
        qx, qy, qz, qw = yaw_to_quaternion(target_yaw)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = target.x_base_m
        msg.pose.position.y = target.y_base_m
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self._target_pose_pub.publish(msg)
        self._ecs_logger.debug(
            "Filtered target published",
            extra=build_ecs_extra(
                component="sidecar.follow",
                action="target_pose_published",
                cable={
                    "target": {
                        "track_id": target.track_id,
                        "confidence": target.confidence,
                        "x_base_m": target.x_base_m,
                        "y_base_m": target.y_base_m,
                        "yaw_rad": target_yaw,
                    }
                },
            ),
        )

    def _log_target_error_sample(
        self,
        reason: str,
        target: Optional[FilteredTarget],
        packet: Optional[TargetPacket],
        holding_last_target: bool,
        packet_age_sec: Optional[float],
        source_age_sec: Optional[float],
    ) -> None:
        distance_error_m = float("nan")
        lateral_error_m = float("nan")
        measurement_valid = False
        if target is not None and reason == "packet_valid":
            distance_error_m = float(target.x_base_m - self._desired_distance)
            lateral_error_m = float(target.y_base_m)
            measurement_valid = True

        self._target_error_trace.log_row(
            ts_unix=time.time(),
            ts_monotonic=time.monotonic(),
            measurement_valid=measurement_valid,
            reason=reason,
            holding_last_target=holding_last_target,
            track_id=None if packet is None else packet.track_id,
            distance_error_m=distance_error_m,
            lateral_error_m=lateral_error_m,
            packet_age_sec=packet_age_sec,
            source_age_sec=source_age_sec,
        )

    def _is_within_follow_tolerance(self, target: FilteredTarget) -> bool:
        distance_error_m = math.hypot(target.x_base_m - self._desired_distance, target.y_base_m)
        return (
            distance_error_m <= max(0.0, self._follow_tolerance_m)
            and abs(self._target_bearing_rad(target)) <= max(0.0, self._bearing_hold_tolerance_rad)
        )

    def _target_bearing_rad(self, target: FilteredTarget) -> float:
        return math.atan2(target.y_base_m, target.x_base_m)

    def _build_follow_path(self, target: FilteredTarget, odom: OdomState) -> tuple[Path, GoalSignature]:
        dx_body = target.x_base_m - self._desired_distance
        dy_body = target.y_base_m

        cos_yaw = math.cos(odom.yaw)
        sin_yaw = math.sin(odom.yaw)
        dx_odom = cos_yaw * dx_body - sin_yaw * dy_body
        dy_odom = sin_yaw * dx_body + cos_yaw * dy_body

        goal_x = odom.x + dx_odom
        goal_y = odom.y + dy_odom
        goal_yaw = normalize_angle(odom.yaw + math.atan2(target.y_base_m, target.x_base_m))

        distance = math.hypot(goal_x - odom.x, goal_y - odom.y)
        num_points = max(2, min(8, int(math.ceil(distance / max(1e-3, self._path_step_m))) + 1))

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "odom"

        for index in range(num_points):
            ratio = index / float(num_points - 1)
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = odom.x + ratio * (goal_x - odom.x)
            pose.pose.position.y = odom.y + ratio * (goal_y - odom.y)
            interp_yaw = normalize_angle(odom.yaw + ratio * angle_delta(goal_yaw, odom.yaw))
            qx, qy, qz, qw = yaw_to_quaternion(interp_yaw)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)

        return path, GoalSignature(goal_x, goal_y, goal_yaw)

    def _should_send_goal(self, signature: GoalSignature, now: float) -> bool:
        if (now - self._last_goal_send_ts) < self._goal_rate_period:
            return False

        if self._last_goal_signature is None:
            return True

        position_delta = math.hypot(
            signature.x - self._last_goal_signature.x,
            signature.y - self._last_goal_signature.y,
        )
        yaw_delta = abs(angle_delta(signature.yaw, self._last_goal_signature.yaw))
        return (
            position_delta >= self._position_change_threshold_m
            or yaw_delta >= self._yaw_change_threshold_rad
        )

    def _send_goal(self, path: Path, signature: GoalSignature, now: float) -> None:
        if not self._follow_path_client.wait_for_server(timeout_sec=0.0):
            return

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = self._controller_id
        goal_msg.goal_checker_id = self._goal_checker_id

        send_future = self._follow_path_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._on_goal_response)
        self._last_goal_signature = signature
        self._last_goal_send_ts = now
        self._ecs_logger.info(
            "FollowPath goal sent",
            extra=build_ecs_extra(
                component="sidecar.follow",
                action="goal_sent",
                cable={
                    "nav": {
                        "goal": {
                            "x": signature.x,
                            "y": signature.y,
                            "yaw": signature.yaw,
                            "path_points": len(path.poses),
                        }
                    }
                },
            ),
        )

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._ecs_logger.warning(
                "FollowPath goal rejected",
                extra=build_ecs_extra(
                    component="sidecar.follow",
                    action="goal_rejected",
                ),
            )
            return

        self._goal_handle = goal_handle
        self._ecs_logger.info(
            "FollowPath goal accepted",
            extra=build_ecs_extra(
                component="sidecar.follow",
                action="goal_accepted",
            ),
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        result = future.result()
        self._goal_handle = None
        self._ecs_logger.info(
            "FollowPath goal result received",
            extra=build_ecs_extra(
                component="sidecar.follow",
                action="goal_result",
                cable={"nav": {"status": int(result.status)}},
            ),
        )

    def _cancel_goal(self) -> None:
        if self._goal_handle is None:
            self._last_goal_signature = None
            return

        self._goal_handle.cancel_goal_async()
        self._goal_handle = None
        self._last_goal_signature = None
        self._ecs_logger.info(
            "FollowPath goal cancelled",
            extra=build_ecs_extra(
                component="sidecar.follow",
                action="goal_cancelled",
            ),
        )

    def _publish_empty_path(self) -> None:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "odom"
        self._path_pub.publish(path)

    def destroy_node(self) -> bool:
        try:
            self._debug_trace.log("session_end")
            self._debug_trace.close()
            self._target_error_trace.close()
            self._ecs_logger.info(
                "Person follow Nav session end",
                extra=build_ecs_extra(
                    component="sidecar.follow",
                    action="session_end",
                ),
            )
            self._cancel_goal()
            self._udp_socket.close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PersonFollowController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
