import threading
import time
import logging
from collections import deque
from typing import Deque, Dict, Any
import numpy as np

# Configure local logging
LOGGER = logging.getLogger("cable.vision.yolo_stairs")

# Stairs query vocabulary (unchanged). Kept FIRST in the combined class list so a
# detection's class index < len(STAIRS_CLASSES) marks it as a stair, not furniture.
STAIRS_CLASSES = ["stairs", "staircase", "steps", "brick stairs", "brick steps", "concrete stairs"]
# Household-furniture vocabulary, appended only when detect_obstacles=True. These feed
# the reactive obstacle avoidance (control.obstacle_avoidance); the staircase stays in
# STAIRS_CLASSES so the dog still climbs stairs rather than dodging them.
OBSTACLE_CLASSES = ["couch", "sofa", "armchair", "chair", "table", "coffee table",
                    "cabinet", "bookshelf", "television", "potted plant"]

class YoloStairsInference:
    """
    Handles parallel open-vocabulary stairs detection using YOLO-World.
    Runs predictions on a separate thread to maintain main loop speed.

    With ``detect_obstacles=True`` the SAME inference also returns furniture
    obstacle boxes (``result["obstacles"]``) split from the stairs by class index --
    one model pass, stairs behaviour unchanged.
    """
    def __init__(
        self,
        model_path: str = "src/sim/models/yolo/yolov8x-worldv2.pt",
        confidence: float = 0.20,
        verbose: bool = False,
        consistency_frames: int = 5,
        consistency_required: int = 3,
        detect_obstacles: bool = False,
        obstacle_confidence: float = 0.25,
    ):
        self.verbose = verbose
        self.confidence = confidence
        self.model_path = model_path
        self.model = None
        self.detect_obstacles = bool(detect_obstacles)
        self.obstacle_confidence = float(obstacle_confidence)
        # Class list actually set on the model + the stairs/furniture split point.
        self._class_names = list(STAIRS_CLASSES) + (list(OBSTACLE_CLASSES) if self.detect_obstacles else [])
        self._n_stair_classes = len(STAIRS_CLASSES)
        self.consistency_frames = max(1, int(consistency_frames))
        self.consistency_required = max(1, min(int(consistency_required), self.consistency_frames))
        self._positive_history: Deque[bool] = deque(maxlen=self.consistency_frames)

        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_result = {
            "detected": False,
            "raw_detected": False,
            "bbox": None,
            "conf": 0.0,
            "positive_count": 0,
            "consistency_frames": self.consistency_frames,
            "consistency_required": self.consistency_required,
            "obstacles": [],
        }
        self._thread = None
        self._stop_event = threading.Event()
        self._new_frame_event = threading.Event()

    def initialize(self) -> bool:
        """Initialize the YOLO-World model and set its query classes."""
        import os
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
        LOGGER.info(f"Loading YOLO-World model from {self.model_path}...")
        try:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(self.model_path)
            # A TensorRT-exported YOLO-World engine has its vocabulary BAKED IN at export
            # time; calling set_classes() on it fails or is a no-op. Skip it for
            # .engine/.trt paths (classes were baked by real/models/export_stairs_trt.py).
            # The .pt path behavior is unchanged.
            _is_engine = str(self.model_path).lower().endswith((".engine", ".trt"))
            if _is_engine:
                LOGGER.info(
                    "YOLO-World engine detected (%s): classes are baked into the engine; "
                    "skipping set_classes()", self.model_path,
                )
            else:
                try:
                    # Define queries/classes dynamically. Stairs first, then (optionally)
                    # furniture -- the split index is self._n_stair_classes.
                    self.model.set_classes(self._class_names)
                    LOGGER.info("YOLO-World initialized and classes set to %s (obstacles=%s)",
                                self._class_names, self.detect_obstacles)
                except Exception as e:
                    LOGGER.warning("set_classes() failed (%s); proceeding with model's existing vocab", e)
            
            # Start background worker thread
            self._thread = threading.Thread(
                target=self._run_inference_loop,
                name="yolo-stairs-worker",
                daemon=True
            )
            self._thread.start()
            return True
        except Exception as e:
            LOGGER.error(f"Failed to initialize YOLO-World stairs detector: {e}")
            return False

    def update_frame(self, image: np.ndarray) -> None:
        """Update the latest image frame for prediction."""
        if image is None:
            return
        with self._lock:
            self._latest_image = image.copy()
            self._new_frame_event.set()

    def get_latest_result(self) -> Dict[str, Any]:
        """Thread-safe retrieval of the latest stairs detection results."""
        with self._lock:
            return dict(self._latest_result)

    def stop(self) -> None:
        """Stop the background inference thread."""
        self._stop_event.set()
        self._new_frame_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run_inference_loop(self) -> None:
        """Background loop running YOLO-World predictions on the latest frame."""
        loop_counter = 0
        while not self._stop_event.is_set():
            self._new_frame_event.wait()
            self._new_frame_event.clear()
            
            if self._stop_event.is_set():
                break

            with self._lock:
                if self._latest_image is None:
                    continue
                img = self._latest_image.copy()

            try:
                # Perform prediction on the frame using device parameter if available in model config
                # Predict returns a list of Results objects
                results = self.model.predict(
                    img,
                    conf=self.confidence,
                    verbose=False,
                    device=None # Auto-select CUDA if available
                )
                
                detected = False
                best_bbox = None
                best_conf = 0.0
                obstacles = []

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        conf_array = boxes.conf.cpu().numpy()
                        xyxy = boxes.xyxy.cpu().numpy()
                        # Class index -> stair vs furniture (indices < _n_stair_classes are stairs).
                        if boxes.cls is not None:
                            cls_array = boxes.cls.cpu().numpy().astype(int)
                        else:
                            cls_array = np.zeros(len(conf_array), dtype=int)
                        # Best STAIRS box only (class-aware) -- preserves the stairs detection
                        # when furniture classes are also present. With obstacles off, every
                        # box is a stair, so this is identical to the old argmax-over-all.
                        best_stair_idx = -1
                        best_stair_conf = 0.0
                        for i in range(len(conf_array)):
                            ci = int(cls_array[i]) if i < len(cls_array) else 0
                            cf = float(conf_array[i])
                            if ci < self._n_stair_classes:
                                if cf > best_stair_conf:
                                    best_stair_conf = cf
                                    best_stair_idx = i
                            elif self.detect_obstacles and cf >= self.obstacle_confidence:
                                obstacles.append({
                                    "bbox": [float(v) for v in xyxy[i].tolist()],
                                    "conf": cf,
                                    "label": (self._class_names[ci]
                                              if ci < len(self._class_names) else str(ci)),
                                })
                        best_conf = best_stair_conf  # for the raw-confidence log below
                        if best_stair_idx >= 0 and best_stair_conf >= self.confidence:
                            detected = True
                            best_bbox = [float(v) for v in xyxy[best_stair_idx].tolist()]

                loop_counter += 1
                if loop_counter % 50 == 0:
                    LOGGER.info(f"[yolo-stairs-worker] Raw best confidence: {best_conf:.4f} (threshold: {self.confidence:.4f})")

                self._positive_history.append(bool(detected))
                positive_count = sum(1 for item in self._positive_history if item)
                consistent_detected = (
                    positive_count >= self.consistency_required
                    and len(self._positive_history) >= self.consistency_required
                )

                with self._lock:
                    self._latest_result = {
                        "detected": bool(consistent_detected),
                        "raw_detected": bool(detected),
                        "bbox": best_bbox,
                        "conf": best_conf,
                        "positive_count": int(positive_count),
                        "consistency_frames": self.consistency_frames,
                        "consistency_required": self.consistency_required,
                        "obstacles": obstacles,
                        "ts_unix": time.time(),
                        "ts_monotonic": time.monotonic(),
                    }
                    
            except Exception as e:
                # Log the error but keep the thread alive
                LOGGER.warning(f"Error during parallel YOLO-World stairs prediction: {e}")
                time.sleep(0.05)
