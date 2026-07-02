"""ROS 2 (rclpy) node shells -- the ONLY modules in real/ that do ROS 2 I/O.

Each node adapts topics <-> the portable logic in ``real/control`` and
``real/perception``. ``sport_startup_node`` is additionally the only module that
touches ``unitree_sdk2`` (Motion-Switcher sport-mode release).
"""
