"""Video recording writer with an HD ffmpeg-pipe backend and a cv2 mp4v fallback.

Two backends, chosen at the first written frame (so the writer is sized to the
real render):

  * ``ffmpeg`` pipe (libx264 / H.264) -- PREFERRED when an ffmpeg binary resolves.
    Accepts up to the requested ``max_resolution`` (e.g. 1280x720 / 1920x1080), so
    the recording cameras are NOT limited to the bundled-encoder macroblock ceiling.
  * ``mp4v`` via ``cv2.VideoWriter`` -- FALLBACK (Isaac's bundled FFMPEG mpeg4).
    Frames are downscaled to <= ``_MP4V_MAX_PIXELS`` so the writer can open, because
    mpeg4 rejects 1080p with -22 (EINVAL) and avc1/H.264 is unavailable in the
    bundled OpenCV (wrong openh264 DLL).

The caller does the camera-specific ``get_rgb()`` / colour conversion and passes a
BGR ndarray to :meth:`write`. Finalisation (:meth:`release`) closes the ffmpeg
stdin and waits, which writes the mp4 ``moov`` atom -- the same guarantee the cv2
``.release()`` provides -- so the graceful-stop path keeps producing playable mp4s.

Pure-Python / host-testable: imports of cv2 / numpy are deferred so the backend
selection logic can be unit-tested without OpenCV or Isaac.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from typing import Callable, Optional, Tuple

# Per-frame size ceiling for the legacy mp4v fallback (mirrors isaac_env's
# _RECORD_MAX_PIXELS: ~768x432 stays under the proven-good mpeg4 macroblock count).
_MP4V_MAX_PIXELS = 768 * 432


def resolve_ffmpeg() -> Optional[str]:
    """Return a usable ffmpeg executable path, or None. Prefers a system ffmpeg on
    PATH, then the static binary shipped by imageio-ffmpeg (commonly present in
    Isaac/conda envs)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def parse_resolution(text: str, default: Tuple[int, int] = (1280, 720)) -> Tuple[int, int]:
    """Parse a ``WxH`` string (e.g. ``"1920x1080"``) into an even-dimensioned tuple."""
    try:
        parts = str(text).lower().replace(" ", "").split("x")
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            return default
        return _even(w), _even(h)
    except Exception:
        return default


def _even(n: int) -> int:
    return max(2, (int(n) // 2) * 2)


def fit_resolution(w: int, h: int, max_pixels: int) -> Tuple[int, int]:
    """Largest aspect-preserving, even-dimensioned (w, h) with w*h <= max_pixels."""
    if w <= 0 or h <= 0:
        return 2, 2
    if w * h <= max_pixels:
        return _even(w), _even(h)
    scale = (max_pixels / float(w * h)) ** 0.5
    return _even(round(w * scale)), _even(round(h * scale))


class RecordingWriter:
    """Lazy-opening video writer. ``write(bgr)`` opens the backend on the first
    frame and resizes every frame to the chosen size; ``release()`` finalises."""

    def __init__(
        self,
        path: str,
        fps: float,
        *,
        encoder: str = "auto",
        max_resolution: Tuple[int, int] = (1280, 720),
        role: str = "video",
        log: Optional[Callable[..., None]] = None,
    ) -> None:
        self.path = str(path)
        self.fps = max(1.0, float(fps))
        self.encoder = (encoder or "auto").lower()
        self.max_w = int(max_resolution[0])
        self.max_h = int(max_resolution[1])
        self.role = str(role)
        self._log = log or (lambda *a, **k: None)
        self.backend: Optional[str] = None  # "ffmpeg" | "mp4v"
        self._proc = None
        self._cv2_writer = None
        self._size: Optional[Tuple[int, int]] = None
        self.started = False
        self.failed = False
        self.frames = 0

    # -- backend order -----------------------------------------------------
    def _backend_order(self):
        if self.encoder == "mp4v":
            return ("mp4v",)
        if self.encoder == "ffmpeg":
            # forced ffmpeg, but keep mp4v as a last resort so we never silently
            # drop the video if ffmpeg cannot spawn.
            return ("ffmpeg", "mp4v")
        return ("ffmpeg", "mp4v")  # auto

    # -- lazy open ---------------------------------------------------------
    def _open(self, frame) -> bool:
        h, w = frame.shape[:2]
        for backend in self._backend_order():
            if backend == "ffmpeg":
                tw, th = fit_resolution(w, h, self.max_w * self.max_h)
                if self._open_ffmpeg(tw, th):
                    self.backend = "ffmpeg"
                    self._size = (tw, th)
                    break
            elif backend == "mp4v":
                tw, th = fit_resolution(w, h, _MP4V_MAX_PIXELS)
                if self._open_mp4v(tw, th):
                    self.backend = "mp4v"
                    self._size = (tw, th)
                    break
        if self.backend is None:
            self.failed = True
            self._log(
                logging.WARNING,
                "recording_codec_failed",
                f"No backend could open the {self.role} video writer; "
                f"{os.path.basename(self.path)} will be missing.",
                role=self.role,
                path=self.path,
                source_resolution=f"{w}x{h}",
                fps=round(self.fps, 2),
            )
            return False
        self.started = True
        self._log(
            logging.INFO,
            "recording_started",
            f"{self.role} recording started",
            role=self.role,
            path=self.path,
            backend=self.backend,
            resolution=f"{self._size[0]}x{self._size[1]}",
            fps=round(self.fps, 2),
        )
        return True

    def _open_ffmpeg(self, w: int, h: int) -> bool:
        exe = resolve_ffmpeg()
        if not exe:
            return False
        cmd = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{self.fps:.4f}", "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            self.path,
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return self._proc.stdin is not None
        except Exception:
            self._proc = None
            return False

    def _open_mp4v(self, w: int, h: int) -> bool:
        try:
            import cv2
        except Exception:
            return False
        codecs = ("avc1", "mp4v") if platform.system() == "Windows" else ("mp4v",)
        for codec in codecs:
            vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec), self.fps, (int(w), int(h)))
            if vw.isOpened():
                self._cv2_writer = vw
                return True
            vw.release()
        return False

    # -- write -------------------------------------------------------------
    def write(self, bgr) -> None:
        if self.failed or bgr is None:
            return
        if self.backend is None and not self._open(bgr):
            return
        import cv2

        tw, th = self._size  # type: ignore[misc]
        if (int(bgr.shape[1]), int(bgr.shape[0])) != (tw, th):
            bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
        if self.backend == "ffmpeg":
            try:
                import numpy as np

                self._proc.stdin.write(np.ascontiguousarray(bgr, dtype=np.uint8).tobytes())
                self.frames += 1
            except Exception as exc:
                self.failed = True
                self._log(
                    logging.WARNING,
                    "recording_write_failed",
                    f"{self.role} ffmpeg pipe write failed; "
                    f"{os.path.basename(self.path)} may be truncated",
                    role=self.role,
                    error=str(exc),
                )
        else:
            self._cv2_writer.write(bgr)
            self.frames += 1

    # -- finalize ----------------------------------------------------------
    def release(self) -> bool:
        """Finalise the file (writes the mp4 moov atom). Returns whether any frame
        was recorded. Safe to call more than once."""
        if self.backend == "ffmpeg" and self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=15)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._cv2_writer is not None:
            try:
                self._cv2_writer.release()
            except Exception:
                pass
            self._cv2_writer = None
        return self.started and self.frames > 0
