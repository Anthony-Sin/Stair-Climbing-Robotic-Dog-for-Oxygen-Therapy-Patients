import threading
import time
import logging
from collections import deque
from typing import Deque, Dict, Any
import numpy as np

# Configure local logging
LOGGER = logging.getLogger("cable.vision.yolo_stairs")

class YoloStairsInference:
    """
    Handles parallel open-vocabulary stairs detection using YOLO-World.
    Runs predictions on a separate thread to maintain main loop speed.
    """
    def __init__(
        self,
        model_path: str = "sim/models/yolo/yolov8x-worldv2.pt",
        confidence: float = 0.20,
        verbose: bool = False,
        consistency_frames: int = 5,
        consistency_required: int = 3,
    ):
        self.verbose = verbose
        self.confidence = confidence
        self.model_path = model_path
        self.model = None
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
            # Define queries/classes dynamically
            self.model.set_classes(["stairs", "staircase", "steps", "brick stairs", "brick steps", "concrete stairs"])
            
            LOGGER.info("YOLO-World initialized and classes set to ['stairs', 'staircase', 'steps', 'brick stairs', 'brick steps', 'concrete stairs']")
            
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

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        # Find the highest confidence detection
                        conf_array = boxes.conf.cpu().numpy()
                        if len(conf_array) > 0:
                            best_idx = int(np.argmax(conf_array))
                            best_conf = float(conf_array[best_idx])
                            if best_conf >= self.confidence:
                                detected = True
                                best_bbox = boxes.xyxy[best_idx].cpu().numpy().tolist()

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
                        "ts_unix": time.time(),
                        "ts_monotonic": time.monotonic(),
                    }
                    
            except Exception as e:
                # Log the error but keep the thread alive
                LOGGER.warning(f"Error during parallel YOLO-World stairs prediction: {e}")
                time.sleep(0.05)
