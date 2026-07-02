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
    # Auto-select: prefer an on-device TensorRT engine when present (~3-5x faster on
    # Orin -- PyTorch inference runs <5 Hz vs >30 Hz, lagging the 50 Hz control loop),
    # fall back to the PyTorch weights otherwise. The .engine is built ON the robot by
    # real/models/export_stairs_trt.py (it bakes the YOLO-World vocab in at export;
    # yolo_stairs_inference skips set_classes for .engine paths accordingly).
    _stairs_engine = "real/models/yolov8s-worldv2.engine"
    _stairs_pt = "real/models/yolov8s-worldv2.pt"
    _stairs_model = _stairs_engine if os.path.exists(_stairs_engine) else _stairs_pt
    sys.argv.extend(["--stairs-model", _stairs_model])

from core.main import main

if __name__ == "__main__":
    main()
