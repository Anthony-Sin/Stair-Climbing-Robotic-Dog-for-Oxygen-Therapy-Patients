"""Async OpenCV preview/recording worker.

Extracted verbatim from main.py: a background thread that renders the preview
window and writes preview frames without blocking the control loop.
"""
import os
import queue
import shutil
import subprocess
import threading
import cv2
from typing import Any, Callable, Dict, List, Optional

from core.hud.visualization import RotationDebugWindow


def _resolve_ffmpeg() -> Optional[str]:
    """Return a usable ffmpeg path (system PATH, then imageio-ffmpeg), or None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class _AsyncPreviewRecorder:
    """Draw the HUD overlay + encode the preview MP4 on a BACKGROUND thread.

    In headless runs the preview video is the only preview output, and doing the
    draw_detections + draw_frame_overlays + VideoWriter.write SYNCHRONOUSLY on the
    control loop cost ~69 ms/frame (measured: run_sim_20260630_003448_571), pinning
    the controller to ~4 FPS -- too slow to track a patient turning at a close
    standoff (the box swings out of the 69 deg frame between samples). This worker
    takes a SNAPSHOT of the per-frame draw inputs and does ALL of that work off the
    hot path, so the control loop only pays a shallow copy. Drop-if-busy: if the
    encoder falls behind, frames are dropped (a slightly choppier debug video) rather
    than ever blocking control.
    """

    def __init__(self, video_path: str, fps: float,
                 draw_detections_fn: Callable, draw_overlays_fn: Callable,
                 save_images_dir: Optional[str] = None,
                 realtime: bool = True) -> None:
        self._video_path = video_path
        self._fps = max(1.0, float(fps))
        self._draw_detections = draw_detections_fn
        self._draw_overlays = draw_overlays_fn
        self._save_images_dir = save_images_dir
        # Retime the finished mp4 to SIM-time playback so the opencv preview plays at the
        # natural robot speed and shares ONE clock with Isaac's scene_view.mp4 (which already
        # records at sim-realtime). The control loop runs at a few real frames/sec but the
        # preview is encoded at a fixed --preview-save-fps, so the same instant lands at very
        # different timestamps between the two previews. We use the per-frame sim_t stamped by
        # Isaac (frame_meta["sim_t"]); without it (e.g. real robot) the retime is skipped.
        self._realtime = bool(realtime)
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._writer = None
        self._dropped = 0
        self._written = 0
        self._first_event: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        # Sim-time span of the encoded frames (from Isaac frame_meta["sim_t"]), used to
        # retime the preview to sim-realtime playback.
        self._sim_t_first: Optional[float] = None
        self._sim_t_last: Optional[float] = None
        self._retimed = False

    @property
    def dropped(self) -> int:
        return int(self._dropped)

    @property
    def written(self) -> int:
        return int(self._written)

    def take_first_event(self) -> Optional[Dict[str, Any]]:
        """One-shot info about the started video (path/shape) for trace logging."""
        with self._lock:
            ev, self._first_event = self._first_event, None
            return ev

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="preview-recorder", daemon=True)
        self._thread.start()

    def submit(self, payload: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            pass
        try:
            _ = self._queue.get_nowait()
            self._dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._dropped += 1

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)  # wake the thread
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
        if self._realtime:
            self._retime_to_realtime()

    def _retime_to_realtime(self) -> None:
        """Stretch the finished preview mp4 to SIM-time playback via an ffmpeg stream-copy
        ``-itsscale`` remux (no re-encode), so it plays at the natural robot speed and matches
        Isaac's scene_view.mp4. No-op when ffmpeg or the per-frame sim_t is unavailable (e.g.
        the real robot, where playback is already real-time)."""
        if self._retimed:
            return
        self._retimed = True
        if (self._sim_t_first is None or self._sim_t_last is None or self._written < 4
                or not self._video_path or not os.path.exists(self._video_path)):
            return
        span = self._sim_t_last - self._sim_t_first
        if span <= 0.0:
            return
        measured_fps = (self._written - 1) / span  # encoded frames per SIM-second
        if measured_fps <= 0.0 or abs(measured_fps - self._fps) <= 0.05 * self._fps:
            return
        exe = _resolve_ffmpeg()
        if not exe:
            return
        ratio = self._fps / measured_fps  # retime so 1 video-sec == 1 sim-sec
        tmp = self._video_path + ".rt.mp4"
        cmd = [exe, "-y", "-loglevel", "error", "-itsscale", f"{ratio:.6f}",
               "-i", self._video_path, "-c", "copy", "-movflags", "+faststart", tmp]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=True, timeout=120)
            os.replace(tmp, self._video_path)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

    def _ensure_writer(self, frame_shape) -> bool:
        if self._writer is not None:
            return True
        import platform as _platform
        h, w = int(frame_shape[0]), int(frame_shape[1])
        codecs = ("avc1", "mp4v") if _platform.system() == "Windows" else ("mp4v",)
        video_dir = os.path.dirname(self._video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            vw = cv2.VideoWriter(self._video_path, fourcc, self._fps, (w, h))
            if vw.isOpened():
                self._writer = vw
                with self._lock:
                    self._first_event = {"path": self._video_path, "fps": self._fps,
                                         "frame_shape": [h, w]}
                return True
            vw.release()
        return False

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                break
            try:
                combined = self._draw_detections(**payload["detections_kwargs"])
                self._draw_overlays(combined, **payload["overlays_kwargs"])
                if self._ensure_writer(combined.shape):
                    self._writer.write(combined)
                    self._written += 1
                    _st = payload.get("sim_t")
                    if _st is not None:
                        if self._sim_t_first is None:
                            self._sim_t_first = float(_st)
                        self._sim_t_last = float(_st)
                if self._save_images_dir:
                    cv2.imwrite(
                        os.path.join(self._save_images_dir,
                                     f"opencv_preview_{int(payload.get('frame_idx', 0)):06d}.jpg"),
                        combined,
                    )
            except Exception:
                # Never let a preview error kill the control loop; just drop the frame.
                self._dropped += 1
        # Drain anything queued at stop so the tail of the run is recorded.
        while True:
            try:
                payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if payload is None:
                continue
            try:
                combined = self._draw_detections(**payload["detections_kwargs"])
                self._draw_overlays(combined, **payload["overlays_kwargs"])
                if self._ensure_writer(combined.shape):
                    self._writer.write(combined)
                    self._written += 1
                    _st = payload.get("sim_t")
                    if _st is not None:
                        if self._sim_t_first is None:
                            self._sim_t_first = float(_st)
                        self._sim_t_last = float(_st)
            except Exception:
                self._dropped += 1


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
