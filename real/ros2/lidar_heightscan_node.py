"""LiDAR PointCloud2 -> PGTT body-frame heightscan + a forward stair-distance signal.

Runs SEPARATELY from the 50 Hz control node so parsing the cloud (~10 Hz, N points)
never adds jitter to the real-time loop. Publishes:
  * /go2/heightscan      Float32MultiArray(99): the raw body-frame elevation grid the
                         low-level node rebuilds PGTT's height_fn from.
  * /go2/stair_detection Float32MultiArray([detected, count, leading_edge, riser_dist]):
                         a forward riser distance for the handoff's approach engage.

Only meaningful in ``heightscan_mode: lidar``; in ``flat`` mode this node need not run
(the low-level node then uses a flat height_fn). SKU + extrinsic come from params.
"""
from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray

from real.perception.heightscan_provider import HeightscanProvider
from real.perception.pointcloud_interface import pointcloud2_to_xyz, to_base, get_extrinsic
from real.ros2.qos import sensor_qos
from go2_locomotion.pgtt_heightmap import PGTT_DIST_X, PGTT_N_ROWS, PGTT_N_COLS

_CENTER_COL = (PGTT_N_COLS - 1) // 2
_BASE_ROW = (PGTT_N_ROWS - 1) // 2   # grid row under the base (forward offset 0)
_RISER_M = 0.05   # terrain rising more than this ahead counts as the first riser


class LidarHeightscanNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_heightscan")
        lidar_topic = str(self.declare_parameter("lidar_topic", "/livox/lidar").value)
        sku = str(self.declare_parameter("lidar_sku", "mid360").value)
        mode = str(self.declare_parameter("heightscan_mode", "lidar").value)
        scale = float(self.declare_parameter("heightscan_scale", 1.0).value)

        self._extr = get_extrinsic(sku)
        self._provider = HeightscanProvider(mode=mode, scale=scale)
        self._hs_pub = self.create_publisher(Float32MultiArray, "/go2/heightscan", sensor_qos())
        self._stair_pub = self.create_publisher(Float32MultiArray, "/go2/stair_detection", sensor_qos())
        self.create_subscription(PointCloud2, lidar_topic, self._on_cloud, sensor_qos())
        self.get_logger().info(f"lidar_heightscan up: {lidar_topic} (sku={sku}, mode={mode})")

    def _on_cloud(self, msg) -> None:
        pts = to_base(pointcloud2_to_xyz(msg), self._extr)
        if pts.shape[0] == 0:
            return
        self._provider.update_from_points(pts)
        grid = self._provider.grid_raw()
        self._hs_pub.publish(Float32MultiArray(data=[float(v) for v in grid]))
        self._stair_pub.publish(Float32MultiArray(data=self._stair_signal(grid)))

    def _stair_signal(self, grid_flat) -> list:
        """First forward riser distance from the center-column elevation profile."""
        g = np.asarray(grid_flat, dtype=np.float32).reshape(PGTT_N_ROWS, PGTT_N_COLS)
        ground = float(np.min(g))
        riser_dist = float("nan")
        count = 0
        # Rows from base (i=_BASE_ROW) forward (i decreasing) -> dist (_BASE_ROW-i)*dist_x.
        for i in range(_BASE_ROW, -1, -1):
            if float(g[i, _CENTER_COL]) - ground > _RISER_M:
                riser_dist = float((_BASE_ROW - i) * PGTT_DIST_X)
                count = 1
                break
        detected = 1.0 if count > 0 else 0.0
        leading = riser_dist if count > 0 else float("nan")
        return [detected, float(count), leading, riser_dist]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarHeightscanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
