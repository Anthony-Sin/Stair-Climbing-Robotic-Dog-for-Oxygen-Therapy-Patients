"""Detection smoke test: YOLO-World must find objects in a textured render.

Needs the YOLO-World weights, a GPU, and a verification image, so it SKIPS
cleanly when any are absent (it is also skipped from host collection by
tests/conftest.py). Paths are overridable via env vars rather than hard-coded to
the Docker container, so the test runs wherever the assets live:

    YOLO_WORLD_MODEL      (default: sim/models/yolo/yolov8x-worldv2.pt)
    DETECTION_TEST_IMAGE  (default: log/verification_test.png)

Run directly (python tests/test_detection_on_sim_textured.py) or via pytest.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# YOLO-World open-vocabulary classes the textured-stairs render should surface.
STAIR_CLASSES = [
    "stairs", "staircase", "steps", "brick stairs",
    "brick steps", "brick", "brick pattern", "person",
]

MODEL_PATH = os.environ.get(
    "YOLO_WORLD_MODEL",
    os.path.join(_REPO, "sim", "models", "yolo", "yolov8x-worldv2.pt"),
)
IMAGE_PATH = os.environ.get(
    "DETECTION_TEST_IMAGE",
    os.path.join(_REPO, "log", "verification_test.png"),
)


def test_detects_objects_in_textured_render():
    import pytest

    try:
        import cv2
        from ultralytics import YOLOWorld
    except ImportError as exc:
        pytest.skip(f"detection dependencies unavailable: {exc}")
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"YOLO-World weights not found: {MODEL_PATH} (set YOLO_WORLD_MODEL)")
    if not os.path.exists(IMAGE_PATH):
        pytest.skip(f"verification image not found: {IMAGE_PATH} (set DETECTION_TEST_IMAGE)")

    img = cv2.imread(IMAGE_PATH)
    assert img is not None, f"OpenCV could not decode the verification image: {IMAGE_PATH}"

    model = YOLOWorld(MODEL_PATH)
    model.set_classes(STAIR_CLASSES)
    results = model.predict(img, conf=0.001, verbose=False)

    assert results, "YOLO-World returned no prediction results object"
    boxes = results[0].boxes
    n = 0 if boxes is None else len(boxes)
    assert n > 0, (
        f"YOLO-World detected 0 objects at conf=0.001 on the textured-stairs image "
        f"({IMAGE_PATH}); the detector is blind or misconfigured. classes={STAIR_CLASSES}"
    )

    names = results[0].names
    detected = sorted({names[int(b.cls[0])] for b in boxes})
    print(f"detected {n} object(s); classes present: {detected}")


if __name__ == "__main__":
    try:
        test_detects_objects_in_textured_render()
        print("PASS")
        _code = 0
    except BaseException as exc:  # noqa: BLE001
        if type(exc).__name__ == "Skipped":
            print(f"SKIP: {exc}")
            _code = 0
        else:
            print(f"FAIL: {exc}")
            _code = 1
    sys.exit(_code)
