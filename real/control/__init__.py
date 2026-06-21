"""Portable real-robot control logic (no rclpy, no unitree_sdk2).

These modules are imported by the ROS 2 nodes in ``real/ros2`` but contain no ROS 2
I/O themselves, so the pure logic (command wire format, LowState->obs adapter,
LowCmd field building + CRC, dual-policy runner, safety watchdog) is unit-testable
on a plain host with only numpy.
"""
