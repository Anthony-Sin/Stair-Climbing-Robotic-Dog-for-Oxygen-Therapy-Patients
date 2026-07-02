"""One abstraction over the LiDAR SKU (Livox Mid-360 vs Hesai XT16) for the heightscan.

The two candidate LiDARs publish ``sensor_msgs/PointCloud2`` on different topics with
different mount extrinsics, but the heightscan provider only ever wants an ``(N,3)``
array in the robot BASE frame. This module is that single seam: parse the cloud,
apply the per-SKU lidar->base extrinsic. Swapping SKUs is a config change, not a code
change -- the provider never learns which LiDAR is fitted.

EXTRINSICS BELOW ARE PLACEHOLDERS -- measure the real lidar->base transform on the
robot and set it in real_robot.yaml. A wrong extrinsic shifts the whole elevation map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LidarExtrinsic:
    """Rigid transform from the LiDAR frame to the robot base frame: p_base = R @ p + t."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


# Placeholder mounts (identity rotation + an approximate height above base). MEASURE.
_EXTRINSICS = {
    "xt16": LidarExtrinsic(t=np.array([0.0, 0.0, 0.10], dtype=np.float32)),
    "mid360": LidarExtrinsic(t=np.array([0.0, 0.0, 0.10], dtype=np.float32)),
}


def get_extrinsic(sku: str) -> LidarExtrinsic:
    return _EXTRINSICS.get(str(sku).lower(), LidarExtrinsic())


def to_base(points_lidar: np.ndarray, extrinsic: LidarExtrinsic) -> np.ndarray:
    """Transform an ``(N,3)`` LiDAR-frame cloud into the base frame (pure)."""
    p = np.asarray(points_lidar, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return (p[:, :3] @ np.asarray(extrinsic.R, dtype=np.float32).T) + np.asarray(extrinsic.t, dtype=np.float32)


def pointcloud2_to_xyz(msg: Any) -> np.ndarray:
    """``sensor_msgs/PointCloud2`` -> ``(N,3)`` float32 (thin ROS-side parse).

    Uses ``sensor_msgs_py.point_cloud2`` when available (the normal Foxy path) and
    drops non-finite points. Returns an empty array on any parse failure so the
    heightscan provider simply keeps its last grid.
    """
    try:
        from sensor_msgs_py import point_cloud2 as pc2  # type: ignore

        # read_points_numpy was added in ROS2 Iron/Humble and does NOT exist on Foxy.
        # Use read_points (available in all ROS2 releases) instead.
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        arr = np.array(list(gen), dtype=np.float32).reshape(-1, 3)
        return arr[np.isfinite(arr).all(axis=1)]
    except Exception:
        return np.empty((0, 3), dtype=np.float32)
