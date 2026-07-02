"""Export the YOLO-World stairs detector to a TensorRT engine (BAKES the vocab in).

WHY: on the Jetson Orin the PyTorch ``yolov8s-worldv2.pt`` stairs model runs <5 Hz,
lagging the 50 Hz control loop; a TensorRT engine runs ~3-5x faster. ``real/main.py``
auto-selects ``real/models/yolov8s-worldv2.engine`` when it exists.

A TensorRT-exported YOLO-World engine has its class vocabulary BAKED IN at export time
(``set_classes`` cannot be called at runtime on an engine -- it fails / is a no-op), so
we MUST call ``set_classes([...])`` here BEFORE ``export``, with the SAME class list
``core.vision.yolo_stairs_inference.YoloStairsInference.initialize`` uses. Keep the two
in sync; if the runtime class list ever changes, re-run this exporter.

-----------------------------------------------------------------------------------
RUN ON THE ROBOT (NVIDIA Jetson Orin), INSIDE THE RUNTIME DOCKER CONTAINER. Do NOT run
it on the dev host -- ultralytics + TensorRT + the Orin GPU are only present there.

The host repo root is mounted at the container's ``/workspace`` (working dir
``/workspace``; see docker/start_follow_system.sh ``-v "$REPO_ROOT:/workspace" -w
/workspace``). Container-relative command:

    python3 real/models/export_stairs_trt.py

  (host form / the file being run: <repo_root>/real/models/export_stairs_trt.py)

It writes ``real/models/yolov8s-worldv2.engine`` next to the ``.pt`` weights (i.e.
``/workspace/real/models/yolov8s-worldv2.engine`` in the container). After it succeeds,
``real/main.py`` picks the engine up automatically on the next run -- no flag needed.
-----------------------------------------------------------------------------------
"""
from __future__ import annotations

import os

# The SAME open-vocabulary class list used at runtime in
# core.vision.yolo_stairs_inference.YoloStairsInference.initialize -- keep in sync so
# the baked-in vocab matches what the controller expects.
STAIRS_CLASSES = ["stairs", "staircase", "steps", "brick stairs", "brick steps", "concrete stairs"]

# Weights in / engine out (repo-relative == container /workspace-relative).
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
PT_PATH = os.path.join(_MODELS_DIR, "yolov8s-worldv2.pt")
ENGINE_PATH = os.path.join(_MODELS_DIR, "yolov8s-worldv2.engine")


def export(pt_path: str = PT_PATH, *, half: bool = True, imgsz: int = 640) -> str:
    """Load the YOLO-World .pt, bake the vocab via set_classes, export to a TRT engine.

    ``half=True`` builds an FP16 engine (matches the other on-device TRT models). Returns
    the exported engine path. ultralytics is imported LAZILY so this module ``py_compile``s
    on the dev host where ultralytics/TensorRT are not installed.
    """
    from ultralytics import YOLOWorld  # lazy: only present in the robot container

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"stairs weights not found: {pt_path}")

    model = YOLOWorld(pt_path)
    # Bake the runtime vocabulary in BEFORE export (cannot set_classes on the engine).
    model.set_classes(STAIRS_CLASSES)
    print(f"[export_stairs_trt] baking classes {STAIRS_CLASSES} and exporting to TensorRT (half={half}, imgsz={imgsz})...")
    out = model.export(format="engine", half=half, imgsz=imgsz)
    print(f"[export_stairs_trt] exported engine -> {out}")
    return str(out)


if __name__ == "__main__":
    export()
