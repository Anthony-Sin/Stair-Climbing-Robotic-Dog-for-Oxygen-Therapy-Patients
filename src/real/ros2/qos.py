"""Centralized ROS 2 QoS profiles for the Go2 port (Foxy).

In Foxy a publisher/subscriber QoS mismatch silently drops every message -- no
error, no warning. Pinning the profiles in one place (and asserting receipt in
preflight) prevents the classic "node is up but receives nothing" failure.

  - sensor_qos  : BestEffort + KeepLast(1). For high-rate latest-wins streams
                  (/lowstate, camera, PointCloud2, /go2/cmd_custom). The consumer
                  always acts on the freshest sample; an occasional drop is fine.
  - reliable_qos: Reliable + KeepLast(1). For commands that must not be dropped
                  (/lowcmd to the robot).
"""
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy


def sensor_qos(depth: int = 1) -> QoSProfile:
    """BestEffort, keep only the latest `depth` samples (high-rate sensor streams)."""
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def reliable_qos(depth: int = 1) -> QoSProfile:
    """Reliable, keep the latest `depth` samples (must-deliver commands, e.g. /lowcmd)."""
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )
