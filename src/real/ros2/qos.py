"""Centralized ROS 2 QoS profiles for the Go2 port (Foxy).

In Foxy a publisher/subscriber QoS mismatch silently drops every message -- no
error, no warning. Pinning the profiles in one place (and asserting receipt in
preflight) prevents the classic "node is up but receives nothing" failure.

  - sensor_qos  : BestEffort + KeepLast(1). For high-rate latest-wins streams
                  (/lowstate, camera, PointCloud2, /go2/cmd_custom). The consumer
                  always acts on the freshest sample; an occasional drop is fine.
  - reliable_qos: Reliable + KeepLast(1). For commands that must not be dropped
                  (/lowcmd to the robot).
  - latched_qos : Reliable + TRANSIENT_LOCAL + KeepLast(1). For one-shot state a
                  late/restarted subscriber must still receive (e.g. the sport-mode
                  "released" gate). BOTH ends MUST use this: a VOLATILE/BestEffort
                  subscriber never receives a TRANSIENT_LOCAL publisher's latched
                  history -- it just waits forever, robot flat on the ground, no error.
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


def latched_qos(depth: int = 1) -> QoSProfile:
    """Reliable + TRANSIENT_LOCAL (latched): late/restarted subscribers still get the
    last published value. Use for one-shot state that a consumer must not miss even if
    it joins after the publish (the sport-mode "released" gate). The publisher AND the
    subscriber must both use this profile."""
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )
