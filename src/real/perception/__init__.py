"""Real-sensor perception: depth->policy preprocessing and LiDAR->heightscan.

Pure-numpy providers consumed by the ROS 2 nodes. ``heightscan_provider`` synthesizes
PGTT's 99-value heightscan from a LiDAR point cloud (the sim's ground-truth terrain
oracle has no real equivalent); ``depth_to_policy`` turns the RealSense depth into the
106x60 person-masked frame the stair detector consumes.
"""
