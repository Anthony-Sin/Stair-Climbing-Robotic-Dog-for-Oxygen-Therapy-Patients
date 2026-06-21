"""Async OpenCV preview/recording worker.

Extracted verbatim from main.py: a background thread that renders the preview
window and writes preview frames without blocking the control loop.
"""
import queue
import threading
import cv2
from typing import Any, Dict, List, Optional

from core.hud.visualization import RotationDebugWindow


class _AsyncPreviewWorker:
    """Runs OpenCV preview rendering in a dedicated thread."""

    def __init__(self, enabled: bool, show_rotation_debug: bool):
        self._enabled = bool(enabled)
        self._show_rotation_debug = bool(show_rotation_debug)
        self._frame_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self._event_queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rotation_debug = RotationDebugWindow() if self._show_rotation_debug else None
        self._dropped_frames = 0
        self._window_name = "TensorRT Detections"

    @property
    def dropped_frames(self) -> int:
        return int(self._dropped_frames)

    def start(self) -> None:
        if not self._enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="preview-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, frame, rotation_error_deg, rotation_cmd,
               rotation_tolerance, edge_penalty) -> None:
        if not self._enabled:
            return
        payload: Dict[str, Any] = {
            "frame": frame,
            "rotation_error_deg": float(rotation_error_deg),
            "rotation_cmd": float(rotation_cmd),
            "rotation_tolerance": float(rotation_tolerance),
            "edge_penalty": float(edge_penalty),
        }
        try:
            self._frame_queue.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            _ = self._frame_queue.get_nowait()
            self._dropped_frames += 1
        except queue.Empty:
            pass
        try:
            self._frame_queue.put_nowait(payload)
        except queue.Full:
            self._dropped_frames += 1

    def poll_events(self) -> List[str]:
        events: List[str] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _run(self) -> None:
        while not self._stop_event.is_set():
            payload: Optional[Dict[str, Any]] = None
            try:
                payload = self._frame_queue.get(timeout=0.03)
            except queue.Empty:
                payload = None
            try:
                if payload is not None:
                    cv2.imshow(self._window_name, payload["frame"])
                    if self._rotation_debug is not None:
                        self._rotation_debug.render(
                            payload["rotation_error_deg"],
                            payload["rotation_cmd"],
                            payload["rotation_tolerance"],
                            payload["edge_penalty"],
                        )
                key = cv2.waitKey(1) & 0xFF
            except Exception:
                self._event_queue.put("preview_error")
                self._stop_event.set()
                break
            if key == ord('q'):
                self._event_queue.put("quit")
            elif key == ord('p'):
                self._event_queue.put("toggle_preparation")
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass
        if self._rotation_debug is not None:
            try:
                cv2.destroyWindow(self._rotation_debug.window_name)
            except Exception:
                pass
