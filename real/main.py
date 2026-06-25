import sys
import os

# Resolve root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add repo root to path so `from core.X import` and `from go2_locomotion.X import` resolve.
# Do NOT add real/bot (superseded sdk2-DDS controller, causes shadow imports) or
# sim/isaac (Isaac-specific modules cause import collisions on the real robot).
sys.path.insert(0, ROOT_DIR)

# Ensure the --sim flag is NOT passed since this is the hardware entrypoint
if "--sim" in sys.argv:
    sys.argv.remove("--sim")

# Native ROS 2 transport is THE real Go2 EDU path (rclpy + unitree_ros2). Default it
# on so the hardware entrypoint routes through RealRobotController -> the low-level
# control node, not the legacy unitree_sdk2-DDS RobotController.
if "--ros2" not in sys.argv:
    sys.argv.append("--ros2")

# Set default model paths for the physical robot if not explicitly provided
if "--trt-engine" not in sys.argv:
    sys.argv.extend(["--trt-engine", "real/models/yolo11n-pose-fp16.trt"])

if "--stairs-model" not in sys.argv:
    # TODO: convert yolov8s-worldv2.pt to TRT for ~3-5x faster inference on Orin.
    # Until then, PyTorch inference may run <5 Hz vs >30 Hz for the TRT pose model,
    # causing stairs_action_active to lag the 50 Hz control loop.
    sys.argv.extend(["--stairs-model", "real/models/yolov8s-worldv2.pt"])

from core.main import main

if __name__ == "__main__":
    main()
