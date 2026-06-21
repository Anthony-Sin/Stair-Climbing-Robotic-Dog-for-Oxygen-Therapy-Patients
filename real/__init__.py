"""Real Unitree Go2 EDU deployment package (Ubuntu 20.04, ROS 2 Foxy).

Native-ROS2 port of the proven sim controller. Layout:
  control/      portable controller logic (no rclpy): the move()/stop() seam that
                core/ drives, the LowState->articulation adapter, the LowCmd
                builder (CRC/mode/remap), the dual-policy runner, the watchdog.
  ros2/         thin rclpy node shells (the only place that does ROS 2 I/O).
  perception/   depth->policy + LiDAR->heightscan providers.
  logging/      real-hardware telemetry (re-emits the sim fall_diag schema).
  verification/ standalone preflight + run ingestion.

Shared locomotion math lives in the repo-root ``go2_locomotion`` package, imported
by both the sim and this port. ``unitree_sdk2`` is confined to
``ros2/sport_startup_node`` (the one unavoidable Motion-Switcher release).
"""
