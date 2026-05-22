import faulthandler
import math
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from go2_nav_bridge.ecs_logging_utils import (
    build_ecs_extra,
    get_ecs_logger,
    setup_ecs_file_logging,
)
from go2_nav_bridge.debug_trace_logger import DebugTraceLogger


@dataclass
class SportStateSnapshot:
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vz: float
    wz: float


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def normalize_network_interface(value: object) -> Optional[str]:
    text = str(value).strip()
    if not text or text.lower() == "auto":
        return None
    return text


TARGET_VALID_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class Go2NavBridge(Node):
    def __init__(self) -> None:
        super().__init__("go2_nav_bridge")

        self.declare_parameter("network_interface", "eth0")
        self.declare_parameter("sport_state_topic", "rt/sportmodestate")
        self.declare_parameter("cmd_vel_topic", "cmd_vel_smoothed")
        self.declare_parameter("target_valid_topic", "person_follow/target_valid")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("odom_publish_rate_hz", 50.0)
        self.declare_parameter("cmd_timeout_sec", 0.4)
        self.declare_parameter("move_repeat_rate_hz", 20.0)
        self.declare_parameter("min_move_command_x", 0.07)
        self.declare_parameter("min_move_command_y", 0.0)
        self.declare_parameter("min_move_command_wz", 0.0)
        self.declare_parameter("max_move_command_x", 1.0)
        self.declare_parameter("max_move_command_y", 0.4)
        self.declare_parameter("max_move_command_wz", 1.65)
        self.declare_parameter("move_scale_x", 1.0)
        self.declare_parameter("move_scale_y", 1.0)
        self.declare_parameter("move_scale_wz", 1.0)
        self.declare_parameter("sport_timeout_sec", 10.0)
        self.declare_parameter("startup_stand_up", True)
        self.declare_parameter("startup_stand_up_wait_sec", 2.0)
        self.declare_parameter("startup_balance_stand", True)
        self.declare_parameter("startup_balance_stand_wait_sec", 1.0)
        self.declare_parameter("ecs_log_dir", "logs")
        self.declare_parameter("enable_ecs_logging", False)
        # DEBUG-TRACE REMOVE-ME: bridge structured trace folder.
        self.declare_parameter("debug_trace_dir", "")

        self._network_interface = normalize_network_interface(
            self.get_parameter("network_interface").value
        )
        self._network_interface_label = self._network_interface or "auto"
        self._sport_state_topic = self.get_parameter("sport_state_topic").value
        self._odom_topic = self.get_parameter("odom_topic").value
        self._odom_frame_id = self.get_parameter("odom_frame_id").value
        self._base_frame_id = self.get_parameter("base_frame_id").value
        self._cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self._move_repeat_period = 1.0 / max(1e-3, float(self.get_parameter("move_repeat_rate_hz").value))
        self._min_move_command_x = float(self.get_parameter("min_move_command_x").value)
        self._min_move_command_y = float(self.get_parameter("min_move_command_y").value)
        self._min_move_command_wz = float(self.get_parameter("min_move_command_wz").value)
        self._max_move_command_x = float(self.get_parameter("max_move_command_x").value)
        self._max_move_command_y = float(self.get_parameter("max_move_command_y").value)
        self._max_move_command_wz = float(self.get_parameter("max_move_command_wz").value)
        self._move_scale_x = float(self.get_parameter("move_scale_x").value)
        self._move_scale_y = float(self.get_parameter("move_scale_y").value)
        self._move_scale_wz = float(self.get_parameter("move_scale_wz").value)
        self._ecs_log_path, _ = setup_ecs_file_logging(
            service_name="go2-nav-bridge",
            event_dataset="cable.sidecar",
            log_dir=self.get_parameter("ecs_log_dir").value,
            file_prefix="go2_nav_bridge_ecs",
            enabled=bool(self.get_parameter("enable_ecs_logging").value),
        )
        self._ecs_logger = get_ecs_logger("sidecar.bridge")
        # DEBUG-TRACE REMOVE-ME: capture watchdog/stop transitions in JSONL.
        self._debug_trace = DebugTraceLogger(
            trace_dir=str(self.get_parameter("debug_trace_dir").value),
            filename="sidecar_bridge_trace.jsonl",
            source="sidecar.bridge",
        )
        self._last_state_log_ts = 0.0
        self._last_cmd_ros_log_ts = 0.0
        self._last_move_ros_log_ts = 0.0
        self._last_watchdog_reason: Optional[str] = None

        self._state_lock = threading.Lock()
        self._latest_state: Optional[SportStateSnapshot] = None

        self._cmd_lock = threading.Lock()
        self._latest_cmd: Optional[Twist] = None
        self._latest_cmd_ts = 0.0
        self._last_move_ts = 0.0

        self._target_valid = False
        self._stop_sent = True

        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            Twist,
            self.get_parameter("cmd_vel_topic").value,
            self._on_cmd_vel,
            10,
        )
        self.create_subscription(
            Bool,
            self.get_parameter("target_valid_topic").value,
            self._on_target_valid,
            TARGET_VALID_QOS,
        )

        if self._ecs_log_path is not None:
            self.get_logger().info(f"ECS analytics log: {self._ecs_log_path}")
        self._log_init_step(
            "dds_channel_factory",
            "start",
            network_interface=self._network_interface_label,
        )
        try:
            ChannelFactoryInitialize(0, self._network_interface)
        except Exception as exc:
            error_message = str(exc) or exc.__class__.__name__
            self.get_logger().error(
                "Unitree DDS init failed for network interface "
                f"'{self._network_interface_label}'. On this robot the Unitree NIC is "
                "`eth0`; use `network_interface:=eth0` or leave `network_interface:=auto` "
                "for CycloneDDS autodetect."
            )
            self._log_init_step(
                "dds_channel_factory",
                "failed",
                network_interface=self._network_interface_label,
                error_message=error_message,
            )
            raise
        self._log_init_step(
            "dds_channel_factory",
            "complete",
            network_interface=self._network_interface_label,
        )
        self._log_init_step("sport_state_subscriber_create", "start")
        self._state_sub = ChannelSubscriber(self._sport_state_topic, SportModeState_)
        self._log_init_step("sport_state_subscriber_create", "complete")
        self._log_init_step("sport_state_subscriber_init", "start")
        self._state_sub.Init(self._on_sport_state, 10)
        self._log_init_step("sport_state_subscriber_init", "complete")

        self._log_init_step("sport_client_create", "start")
        self._sport = SportClient()
        self._log_init_step("sport_client_create", "complete")
        self._log_init_step("sport_client_timeout", "start")
        self._sport.SetTimeout(float(self.get_parameter("sport_timeout_sec").value))
        self._log_init_step("sport_client_timeout", "complete")
        self._log_init_step("sport_client_init", "start")
        self._sport.Init()
        self._log_init_step("sport_client_init", "complete")
        self._ecs_logger.info(
            "Go2 Nav bridge session start",
            extra=build_ecs_extra(
                component="sidecar.bridge",
                action="session_start",
                cable={
                    "robot": {
                        "network_interface": self._network_interface,
                        "network_interface_resolved": self._network_interface_label,
                        "cmd_timeout_sec": self._cmd_timeout_sec,
                    }
                },
            ),
        )
        self._debug_trace.log(
            "session_start",
            network_interface=self._network_interface_label,
            cmd_timeout_sec=float(self._cmd_timeout_sec),
            move_repeat_period_sec=float(self._move_repeat_period),
        )

        if bool(self.get_parameter("startup_balance_stand").value):
            if bool(self.get_parameter("startup_stand_up").value):
                try:
                    self._log_init_step("stand_up", "start")
                    result = self._sport.StandUp()
                    self.get_logger().info(f"StandUp startup command returned {result}")
                    self._log_init_step("stand_up", "complete", result_code=int(result))
                    self._ecs_logger.info(
                        "StandUp startup command sent",
                        extra=build_ecs_extra(
                            component="sidecar.bridge",
                            action="stand_up_startup",
                            cable={"robot": {"startup": {"step": "stand_up", "result_code": int(result)}}},
                        ),
                    )
                    time.sleep(float(self.get_parameter("startup_stand_up_wait_sec").value))
                except Exception as exc:
                    self.get_logger().error(f"StandUp startup command failed: {exc}")
                    self._log_init_step("stand_up", "failed", error_message=str(exc))
                    self._ecs_logger.error(
                        "StandUp startup command failed",
                        extra=build_ecs_extra(
                            component="sidecar.bridge",
                            action="stand_up_failed",
                            cable={"error": {"message": str(exc)}},
                        ),
                    )
            try:
                self._log_init_step("balance_stand", "start")
                result = self._sport.BalanceStand()
                self.get_logger().info(f"BalanceStand startup command returned {result}")
                self._log_init_step("balance_stand", "complete", result_code=int(result))
                self._ecs_logger.info(
                    "BalanceStand startup command sent",
                    extra=build_ecs_extra(
                        component="sidecar.bridge",
                        action="balance_stand_startup",
                        cable={"robot": {"result_code": int(result)}},
                    ),
                )
                time.sleep(float(self.get_parameter("startup_balance_stand_wait_sec").value))
            except Exception as exc:
                self.get_logger().error(f"BalanceStand startup command failed: {exc}")
                self._log_init_step("balance_stand", "failed", error_message=str(exc))
                self._ecs_logger.error(
                    "BalanceStand startup command failed",
                    extra=build_ecs_extra(
                        component="sidecar.bridge",
                        action="balance_stand_failed",
                        cable={"error": {"message": str(exc)}},
                    ),
                )

        odom_period = 1.0 / max(1e-3, float(self.get_parameter("odom_publish_rate_hz").value))
        self.create_timer(odom_period, self._publish_odom)
        self.create_timer(0.05, self._watchdog)

    def _log_init_step(self, step: str, phase: str, **fields: float | int | str) -> None:
        ros_log = self.get_logger().error if phase == "failed" else self.get_logger().info
        ecs_log = self._ecs_logger.error if phase == "failed" else self._ecs_logger.info
        ros_log(f"Bridge init step {step} {phase}")
        ecs_log(
            "Bridge init step",
            extra=build_ecs_extra(
                component="sidecar.bridge",
                action="init_step",
                cable={
                    "robot": {
                        "init": {
                            "step": step,
                            "phase": phase,
                            **fields,
                        }
                    }
                },
            ),
        )

    def _on_sport_state(self, msg: SportModeState_) -> None:
        snapshot = SportStateSnapshot(
            x=float(msg.position[0]),
            y=float(msg.position[1]),
            z=float(msg.position[2]),
            yaw=float(msg.imu_state.rpy[2]),
            vx=float(msg.velocity[0]),
            vy=float(msg.velocity[1]),
            vz=float(msg.velocity[2]),
            wz=float(msg.yaw_speed),
        )
        with self._state_lock:
            self._latest_state = snapshot
        now = time.monotonic()
        if (now - self._last_state_log_ts) >= 0.5:
            self.get_logger().info(
                "Odom snapshot "
                f"x={snapshot.x:.3f} y={snapshot.y:.3f} yaw={snapshot.yaw:.3f} "
                f"vx={snapshot.vx:.3f} vy={snapshot.vy:.3f} wz={snapshot.wz:.3f}"
            )
            self._ecs_logger.debug(
                "Go2 odom snapshot",
                extra=build_ecs_extra(
                    component="sidecar.bridge",
                    action="odom_snapshot",
                    cable={
                        "nav": {
                            "odom": {
                                "x": snapshot.x,
                                "y": snapshot.y,
                                "z": snapshot.z,
                                "yaw": snapshot.yaw,
                                "vx": snapshot.vx,
                                "vy": snapshot.vy,
                                "vz": snapshot.vz,
                                "wz": snapshot.wz,
                            }
                        }
                    },
                ),
            )
            self._last_state_log_ts = now

    def _on_target_valid(self, msg: Bool) -> None:
        target_valid = bool(msg.data)
        if self._target_valid == target_valid:
            return

        self._target_valid = target_valid
        self.get_logger().info(f"Target valid changed: {self._target_valid}")
        self._ecs_logger.info(
            "Target validity changed",
            extra=build_ecs_extra(
                component="sidecar.bridge",
                action="target_valid_change",
                cable={"target": {"valid": bool(self._target_valid)}},
            ),
        )
        if not self._target_valid:
            self._issue_stop("target_invalid_transition")

    def _on_cmd_vel(self, msg: Twist) -> None:
        with self._cmd_lock:
            self._latest_cmd = msg
            self._latest_cmd_ts = time.monotonic()
            cmd_ts = self._latest_cmd_ts

        if (cmd_ts - self._last_cmd_ros_log_ts) >= 0.5:
            self.get_logger().info(
                "cmd_vel_smoothed "
                f"vx={float(msg.linear.x):.3f} vy={float(msg.linear.y):.3f} "
                f"wz={float(msg.angular.z):.3f} target_valid={self._target_valid}"
            )
            self._last_cmd_ros_log_ts = cmd_ts

        if not self._target_valid:
            self._issue_stop("target_invalid_on_cmd_vel")
            return

        self._ecs_logger.debug(
            "Smoothed cmd_vel received",
            extra=build_ecs_extra(
                component="sidecar.bridge",
                action="cmd_vel_received",
                cable={
                    "robot": {
                        "command": {
                            "vx": float(msg.linear.x),
                            "vy": float(msg.linear.y),
                            "wz": float(msg.angular.z),
                        }
                    }
                },
            ),
        )
        self._send_move(msg)

    @staticmethod
    def _apply_command_floor(value: float, floor: float) -> float:
        if abs(value) < 1e-6:
            return 0.0
        if abs(value) < floor:
            return math.copysign(floor, value)
        return value

    def _send_move(self, msg: Twist) -> None:
        vx_raw = float(msg.linear.x)
        vy_raw = float(msg.linear.y)
        wz_raw = float(msg.angular.z)
        vx = self._apply_command_floor(
            vx_raw * self._move_scale_x, self._min_move_command_x
        )
        if self._max_move_command_x > 0.0:
            vx = min(max(vx, -self._max_move_command_x), self._max_move_command_x)
        vy = self._apply_command_floor(
            vy_raw * self._move_scale_y, self._min_move_command_y
        )
        if self._max_move_command_y > 0.0:
            vy = min(max(vy, -self._max_move_command_y), self._max_move_command_y)
        wz = self._apply_command_floor(
            wz_raw * self._move_scale_wz, self._min_move_command_wz
        )
        if self._max_move_command_wz > 0.0:
            wz = min(max(wz, -self._max_move_command_wz), self._max_move_command_wz)
        try:
            self._sport.Move(vx, vy, wz)
        except Exception as exc:
            self.get_logger().error(f"Move command failed: {exc}")
            self._ecs_logger.error(
                "Sport move command failed",
                extra=build_ecs_extra(
                    component="sidecar.bridge",
                    action="move_failed",
                    cable={
                        "robot": {
                            "command": {
                                "vx": vx,
                                "vy": vy,
                                "wz": wz,
                            }
                        },
                        "error": {"message": str(exc)},
                    },
                ),
            )
            return

        self._last_move_ts = time.monotonic()
        self._stop_sent = False
        self._last_watchdog_reason = None
        if (self._last_move_ts - self._last_move_ros_log_ts) >= 0.5:
            self.get_logger().info(
                "Move command sent "
                f"raw=({vx_raw:.3f}, {vy_raw:.3f}, {wz_raw:.3f}) "
                f"applied=({vx:.3f}, {vy:.3f}, {wz:.3f})"
            )
            self._last_move_ros_log_ts = self._last_move_ts
        self._ecs_logger.debug(
            "Sport move command sent",
            extra=build_ecs_extra(
                component="sidecar.bridge",
                action="move_command",
                cable={
                    "robot": {
                        "command": {
                            "vx": vx,
                            "vy": vy,
                            "wz": wz,
                        }
                    }
                },
            ),
        )

    def _issue_stop(self, reason: str) -> None:
        if self._stop_sent:
            if self._last_watchdog_reason != reason:
                self.get_logger().info(f"Stop already active, reason={reason}")
                self._last_watchdog_reason = reason
                self._debug_trace.log("stop_already_active", reason=reason)
            return
        try:
            self._sport.StopMove()
        except Exception as exc:
            self.get_logger().error(f"StopMove command failed: {exc}")
            self._ecs_logger.error(
                "StopMove command failed",
                extra=build_ecs_extra(
                    component="sidecar.bridge",
                    action="stop_failed",
                    cable={"error": {"message": str(exc)}},
                ),
            )
            self._debug_trace.log("stop_command_failed", reason=reason, error_message=str(exc))
            return
        self._stop_sent = True
        self._last_watchdog_reason = reason
        self.get_logger().info(f"StopMove command sent, reason={reason}")
        # DEBUG-TRACE REMOVE-ME: capture direct stop reasons that correlate with stalls/timeouts.
        self._debug_trace.log("stop_command_sent", reason=reason)
        self._ecs_logger.info(
            "StopMove command sent",
            extra=build_ecs_extra(
                component="sidecar.bridge",
                action="stop_command",
                cable={"robot": {"stop_reason": reason}},
            ),
        )

    def _watchdog(self) -> None:
        if not self._target_valid:
            self._issue_stop("target_invalid_watchdog")
            return

        with self._cmd_lock:
            latest_cmd = self._latest_cmd
            latest_cmd_ts = self._latest_cmd_ts

        now = time.monotonic()
        if latest_cmd is None or (now - latest_cmd_ts) > self._cmd_timeout_sec:
            self._issue_stop("cmd_timeout")
            return

        if (now - self._last_move_ts) >= self._move_repeat_period:
            self._send_move(latest_cmd)

    def _publish_odom(self) -> None:
        with self._state_lock:
            state = self._latest_state

        if state is None:
            return

        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_to_quaternion(state.yaw)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self._odom_frame_id
        odom_msg.child_frame_id = self._base_frame_id
        odom_msg.pose.pose.position.x = state.x
        odom_msg.pose.pose.position.y = state.y
        odom_msg.pose.pose.position.z = state.z
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.twist.twist.linear.x = state.vx
        odom_msg.twist.twist.linear.y = state.vy
        odom_msg.twist.twist.linear.z = state.vz
        odom_msg.twist.twist.angular.z = state.wz
        self._odom_pub.publish(odom_msg)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self._odom_frame_id
        tf_msg.child_frame_id = self._base_frame_id
        tf_msg.transform.translation.x = state.x
        tf_msg.transform.translation.y = state.y
        tf_msg.transform.translation.z = state.z
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(tf_msg)

    def destroy_node(self) -> bool:
        try:
            self._debug_trace.log("session_end")
            self._debug_trace.close()
            self._ecs_logger.info(
                "Go2 Nav bridge session end",
                extra=build_ecs_extra(
                    component="sidecar.bridge",
                    action="session_end",
                ),
            )
            self._issue_stop("shutdown")
            self._state_sub.Close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    faulthandler.enable(all_threads=True)
    rclpy.init(args=args)
    node = None
    try:
        node = Go2NavBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback_lines = traceback.format_exc().rstrip().splitlines()
        if node is None:
            for line in traceback_lines:
                print(line, flush=True)
        else:
            for line in traceback_lines:
                node.get_logger().error(line)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
