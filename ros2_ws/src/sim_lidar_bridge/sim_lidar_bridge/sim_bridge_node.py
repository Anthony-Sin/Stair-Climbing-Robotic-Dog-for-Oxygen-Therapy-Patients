"""Sim <-> ROS2 bridge for the Isaac Go2 environment.

Receives the *real* XT16 raycast point cloud + robot pose that isaac_env emits over
UDP (see Ros2BridgeCloudSender) and republishes them on the SAME ROS2 topics the
real robot uses, then forwards Nav2's smoothed command back to Isaac:

    Isaac (cast_scan cloud + pose) --UDP--> [this node] --> /xt16/lidar_points
                                                            /odom + TF
                              /cmd_vel_smoothed --> [this node] --UDP--> Isaac cmd port

So the real Nav2 / costmap / MPPI stack runs unchanged against sim data. This node
is sim-only: on the real robot the Hesai driver publishes /xt16/lidar_points and
go2_nav_bridge publishes /odom and drives the robot -- the same topics -- so going
to hardware is just "stop this node, start the real drivers", with Nav2 untouched.

The point cloud is published in the LiDAR sensor frame (x forward, y left, z up),
identical to the real XT16 convention, so the static base_link -> hesai_xt16
transform and the costmap config carry over verbatim.
"""

import base64
import json
import math
import socket
import zlib

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def _yaw_to_quat(yaw_rad: float):
    """(x, y, z, w) quaternion for a yaw-only rotation about +Z."""
    half = 0.5 * float(yaw_rad)
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _make_pointcloud2(stamp, frame_id: str, points_xyz: np.ndarray) -> PointCloud2:
    """Build an unorganized XYZ float32 PointCloud2 (little-endian, 12-byte points)."""
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * int(pts.shape[0])
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


class SimLidarBridge(Node):
    def __init__(self) -> None:
        super().__init__("sim_lidar_bridge")

        self.declare_parameter("udp_listen_host", "0.0.0.0")
        self.declare_parameter("udp_listen_port", 52003)
        self.declare_parameter("isaac_cmd_host", "127.0.0.1")
        self.declare_parameter("isaac_cmd_port", 52001)
        self.declare_parameter("points_topic", "/xt16/lidar_points")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_topic", "/cmd_vel_smoothed")
        self.declare_parameter("lidar_frame", "hesai_xt16")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")
        # XT16 mount offset in base_link (matches sim_lidar_xt16.Xt16Config).
        self.declare_parameter("mount_x_m", 0.0)
        self.declare_parameter("mount_y_m", 0.0)
        self.declare_parameter("mount_z_m", 0.10)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("poll_period_sec", 0.005)
        # Odometry drift (opt-in; all zero => exact ground-truth pose passthrough).
        # Real legged odometry accumulates unbounded error with distance travelled,
        # worst during stair climbs (foot slip); injecting it here exercises the
        # Nav2/costmap/MPPI stack against the drift the perfect Isaac pose hides.
        self.declare_parameter("odom_drift_slip", 0.0)         # frac. extra translation per step
        self.declare_parameter("odom_yaw_drift_per_m", 0.0)    # rad heading error added per metre
        self.declare_parameter("odom_climb_drift_gain", 0.0)   # extra slip multiplier per metre |dz|

        gp = self.get_parameter
        self._lidar_frame = str(gp("lidar_frame").value)
        self._base_frame = str(gp("base_frame").value)
        self._odom_frame = str(gp("odom_frame").value)
        self._publish_tf = bool(gp("publish_tf").value)
        self._isaac_cmd_dest = (str(gp("isaac_cmd_host").value), int(gp("isaac_cmd_port").value))

        # Odom-drift config + accumulator state.
        self._odom_slip = float(gp("odom_drift_slip").value)
        self._odom_yaw_per_m = float(gp("odom_yaw_drift_per_m").value)
        self._odom_climb_gain = float(gp("odom_climb_drift_gain").value)
        self._odom_drift_enabled = (self._odom_slip != 0.0 or self._odom_yaw_per_m != 0.0)
        self._gt_prev = None     # (x, y, yaw, z) ground truth at the previous packet
        self._odom_est = None    # [x, y, yaw] drifting estimate

        # UDP receive socket for the Isaac cloud/odom sidecar (non-blocking poll).
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 21)
        self._rx.bind((str(gp("udp_listen_host").value), int(gp("udp_listen_port").value)))
        self._rx.setblocking(False)

        # UDP send socket to forward Nav2 cmd_vel back into Isaac.
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._cloud_pub = self.create_publisher(
            PointCloud2, str(gp("points_topic").value), qos_profile_sensor_data
        )
        self._odom_pub = self.create_publisher(Odometry, str(gp("odom_topic").value), 10)
        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self.create_subscription(Twist, str(gp("cmd_topic").value), self._on_cmd_vel, 10)

        self._publish_static_mount_tf(
            float(gp("mount_x_m").value), float(gp("mount_y_m").value), float(gp("mount_z_m").value)
        )

        # Previous pose for odom twist finite-difference.
        self._prev = None  # (x, y, yaw, ts)
        self._last_seq = -1
        self._cloud_count = 0

        self.create_timer(float(gp("poll_period_sec").value), self._poll_udp)
        self.get_logger().info(
            "sim_lidar_bridge up: listening on "
            f"{gp('udp_listen_host').value}:{gp('udp_listen_port').value}, "
            f"publishing {gp('points_topic').value} + {gp('odom_topic').value}, "
            f"forwarding {gp('cmd_topic').value} -> {self._isaac_cmd_dest}"
        )
        if self._odom_drift_enabled:
            self.get_logger().warn(
                "odom drift ENABLED (sim-only): "
                f"slip={self._odom_slip} yaw_per_m={self._odom_yaw_per_m} "
                f"climb_gain={self._odom_climb_gain} -- /odom is NOT ground truth"
            )

    # ------------------------------------------------------------------ TF
    def _publish_static_mount_tf(self, mx: float, my: float, mz: float) -> None:
        if not self._publish_tf:
            return
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self._base_frame
        tf.child_frame_id = self._lidar_frame
        tf.transform.translation.x = float(mx)
        tf.transform.translation.y = float(my)
        tf.transform.translation.z = float(mz)
        tf.transform.rotation.w = 1.0  # sensor axes aligned with base_link
        self._static_tf.sendTransform(tf)

    def _broadcast_odom_tf(self, stamp, x: float, y: float, z: float, qz: float, qw: float) -> None:
        if not self._publish_tf:
            return
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self._odom_frame
        tf.child_frame_id = self._base_frame
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.translation.z = float(z)
        tf.transform.rotation.z = float(qz)
        tf.transform.rotation.w = float(qw)
        self._tf.sendTransform(tf)

    # ------------------------------------------------------------- cmd_vel
    def _on_cmd_vel(self, msg: Twist) -> None:
        # Mirror SimRobotController: robot body frame m/s + rad/s, reverse-X
        # suppressed (the Go2 high-level Move never reverses on X either).
        vx = max(0.0, float(msg.linear.x))
        payload = {
            "vx": vx,
            "vy": float(msg.linear.y),
            "wz": float(msg.angular.z),
            "stairs_detected": False,
        }
        try:
            self._tx.sendto(json.dumps(payload).encode("utf-8"), self._isaac_cmd_dest)
        except Exception as exc:  # noqa: BLE001 - log and keep the node alive
            self.get_logger().warn(f"cmd_vel forward failed: {exc}")

    # --------------------------------------------------------------- ingest
    def _poll_udp(self) -> None:
        latest = None
        # Drain the socket; only the newest scan is worth publishing this tick.
        while True:
            try:
                data, _addr = self._rx.recvfrom(1 << 16)
            except BlockingIOError:
                break
            except OSError:
                break
            latest = data
        if latest is None:
            return
        try:
            pkt = json.loads(latest.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"bad cloud packet: {exc}")
            return
        self._publish_packet(pkt)

    def _publish_packet(self, pkt: dict) -> None:
        seq = int(pkt.get("seq", -1))
        if seq == self._last_seq:
            return
        self._last_seq = seq

        pose = pkt.get("pose", {}) or {}
        x = float(pose.get("x", 0.0))
        y = float(pose.get("y", 0.0))
        z = float(pose.get("z", 0.0))
        yaw = math.radians(float(pose.get("yaw_deg", 0.0)))
        ts = float(pkt.get("ts", 0.0))

        # Apply optional odom drift. When disabled this returns the GT pose exactly,
        # so the published /odom + TF are unchanged. The point cloud is in the LiDAR
        # sensor frame and is intentionally NOT drifted.
        gt_x, gt_y = x, y
        x, y, yaw = self._apply_odom_drift(x, y, yaw, z)

        stamp = self.get_clock().now().to_msg()

        # --- Point cloud (real raycast hits, sensor frame) -------------------
        try:
            raw = zlib.decompress(base64.b64decode(pkt.get("points", "")))
            pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 3)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"cloud decode failed: {exc}")
            pts = np.zeros((0, 3), dtype=np.float32)
        self._cloud_pub.publish(_make_pointcloud2(stamp, self._lidar_frame, pts))

        # --- Odometry + TF ---------------------------------------------------
        _qx, _qy, qz, qw = _yaw_to_quat(yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Finite-difference twist from successive poses (odom frame -> body frame).
        if self._prev is not None:
            px, py, pyaw, pts_ts = self._prev
            d = ts - pts_ts
            if d > 1e-4:
                wx = (x - px) / d
                wy = (y - py) / d
                dyaw = math.atan2(math.sin(yaw - pyaw), math.cos(yaw - pyaw))
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                odom.twist.twist.linear.x = cos_y * wx + sin_y * wy
                odom.twist.twist.linear.y = -sin_y * wx + cos_y * wy
                odom.twist.twist.angular.z = dyaw / d
        self._prev = (x, y, yaw, ts)

        self._odom_pub.publish(odom)
        self._broadcast_odom_tf(stamp, x, y, z, qz, qw)

        self._cloud_count += 1
        if self._cloud_count % 50 == 1:
            drift_note = ""
            if self._odom_drift_enabled:
                drift_note = f" drift_err={math.hypot(x - gt_x, y - gt_y):.3f}m"
            self.get_logger().info(
                f"bridged scan seq={seq} pts={int(pts.shape[0])} "
                f"pose=({x:.2f},{y:.2f},yaw={math.degrees(yaw):.1f}){drift_note}"
            )

    def _apply_odom_drift(self, x: float, y: float, yaw: float, z: float):
        """Return a drifting odom pose estimate (default: exact GT passthrough).

        Integrates the ground-truth per-packet increments into a separate estimate
        with (a) translation slip that scales the travelled distance and (b) a
        heading error that accumulates with distance -- the dominant real legged-
        odometry error -- optionally amplified while climbing (|dz|). With the
        drift params zero this is an exact passthrough, so /odom + TF are unchanged.
        """
        if not self._odom_drift_enabled:
            return x, y, yaw
        if self._gt_prev is None or self._odom_est is None:
            self._gt_prev = (x, y, yaw, z)
            self._odom_est = [x, y, yaw]
            return x, y, yaw
        px, py, pyaw, pz = self._gt_prev
        self._gt_prev = (x, y, yaw, z)
        dx, dy = x - px, y - py
        dyaw = math.atan2(math.sin(yaw - pyaw), math.cos(yaw - pyaw))
        ddist = math.hypot(dx, dy)
        slip = self._odom_slip * (1.0 + self._odom_climb_gain * abs(z - pz))
        # Accumulate the true turn plus a distance-proportional heading error.
        new_yaw = self._odom_est[2] + dyaw + self._odom_yaw_per_m * ddist
        self._odom_est[2] = math.atan2(math.sin(new_yaw), math.cos(new_yaw))
        # Rotate the GT translation increment by the accumulated heading error and
        # scale it by the slip, then integrate into the estimate.
        yaw_err = self._odom_est[2] - yaw
        c, s = math.cos(yaw_err), math.sin(yaw_err)
        self._odom_est[0] += (c * dx - s * dy) * (1.0 + slip)
        self._odom_est[1] += (s * dx + c * dy) * (1.0 + slip)
        return self._odom_est[0], self._odom_est[1], self._odom_est[2]

    def destroy_node(self) -> bool:
        try:
            self._rx.close()
            self._tx.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimLidarBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
