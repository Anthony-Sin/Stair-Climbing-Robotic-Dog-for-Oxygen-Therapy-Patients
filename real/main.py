import sys
import os

# Resolve root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add root, core, and real/bot directories to python path
sys.path.extend([
    ROOT_DIR,
    os.path.join(ROOT_DIR, "core"),
    os.path.join(ROOT_DIR, "real", "bot"),
    os.path.join(ROOT_DIR, "sim", "isaac")
])

# Ensure the --sim flag is NOT passed since this is the hardware entrypoint
if "--sim" in sys.argv:
    sys.argv.remove("--sim")

# Set default model paths for the physical robot if not explicitly provided
if "--trt-engine" not in sys.argv:
    sys.argv.extend(["--trt-engine", "real/models/yolo11n-pose-fp16.trt"])

if "--stairs-model" not in sys.argv:
    sys.argv.extend(["--stairs-model", "real/models/yolov8s-worldv2.pt"])

from core.main import main

if __name__ == "__main__":
    main()
