from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class CameraFrame:
    color_image: np.ndarray
    depth_image: np.ndarray
    serial: str
    frame_number: int
    timestamp_ms: float


class RealSenseCameraClient:
    """Lightweight RealSense RGB-D client for the ROS 2 sidecar container."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        rotate: int = 0,
        camera_serials: Optional[Sequence[str]] = None,
        warmup_attempts: int = 3,
        startup_timeout_ms: int = 5000,
        frame_timeout_ms: int = 1000,
        align_depth_to_color: bool = True,
        verbose: bool = True,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.rotate = int(rotate)
        self.camera_serials = list(camera_serials or [])
        self.warmup_attempts = int(warmup_attempts)
        self.startup_timeout_ms = int(startup_timeout_ms)
        self.frame_timeout_ms = int(frame_timeout_ms)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.verbose = bool(verbose)

        self.pipeline: Optional[rs.pipeline] = None
        self.profile: Optional[rs.pipeline_profile] = None
        self.active_serial: Optional[str] = None
        self.active_resolution: Optional[Tuple[int, int]] = None
        self.align = rs.align(rs.stream.color) if self.align_depth_to_color else None
        self._last_frame_meta: Dict[str, object] = {
            "success": False,
            "wait_ms": 0.0,
            "timeout_ms": self.frame_timeout_ms,
            "error": None,
        }

    def _log(self, message: str, *, force: bool = False) -> None:
        if self.verbose or force:
            stream = sys.stderr if force else sys.stdout
            print(message, file=stream)

    def _get_connected_cameras(self) -> Sequence[str]:
        ctx = rs.context()
        serials = []
        for index in range(len(ctx.devices)):
            serial = ctx.devices[index].get_info(rs.camera_info.serial_number)
            serials.append(serial)
            self._log(f"[RealSenseCameraClient] Detected camera {index}: {serial}")
        return serials

    def _iter_resolutions(self) -> Sequence[Tuple[int, int]]:
        preferred = [(self.width, self.height)]
        fallback = [(1280, 720), (848, 480), (640, 480), (424, 240), (320, 240)]
        for candidate in fallback:
            if candidate not in preferred:
                preferred.append(candidate)
        return preferred

    def _setup_pipeline(self, serial: str, width: int, height: int) -> Tuple[rs.pipeline, rs.config]:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, self.fps)
        return pipeline, config

    def start(self) -> None:
        if self.pipeline is not None:
            return

        connected = list(self._get_connected_cameras())
        if not connected:
            raise RuntimeError("No RealSense cameras detected")

        serials = self.camera_serials or connected
        serials = [serial for serial in serials if serial in connected]
        if not serials:
            raise RuntimeError(
                f"Requested camera serial(s) not found: {list(self.camera_serials)}. "
                f"Connected cameras: {connected}"
            )

        for serial in serials:
            for width, height in self._iter_resolutions():
                pipeline = None
                try:
                    self._log(
                        f"[RealSenseCameraClient] Trying {serial} at {width}x{height}@{self.fps}"
                    )
                    pipeline, config = self._setup_pipeline(serial, width, height)
                    profile = pipeline.start(config)

                    for attempt in range(1, self.warmup_attempts + 1):
                        frame = self.get_frame(
                            pipeline=pipeline,
                            timeout_ms=self.startup_timeout_ms,
                        )
                        if frame is not None:
                            self.pipeline = pipeline
                            self.profile = profile
                            self.active_serial = serial
                            self.active_resolution = (width, height)
                            self._log(
                                f"[RealSenseCameraClient] Started {serial} "
                                f"at {width}x{height}@{self.fps}"
                            )
                            return
                        self._log(
                            f"[RealSenseCameraClient] No frame yet for {serial} "
                            f"(attempt {attempt}/{self.warmup_attempts})"
                        )
                        time.sleep(0.15)
                    pipeline.stop()
                except Exception as exc:
                    self._log(
                        f"[RealSenseCameraClient] Failed to start {serial} "
                        f"at {width}x{height}: {exc}"
                    )
                    if pipeline is not None:
                        try:
                            pipeline.stop()
                        except Exception:
                            pass

        raise RuntimeError("Failed to start RealSense camera at any tested resolution")

    def stop(self) -> None:
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        finally:
            self.pipeline = None
            self.profile = None
            self.active_serial = None
            self.active_resolution = None

    def __enter__(self) -> "RealSenseCameraClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _rotate_image(self, image: np.ndarray) -> np.ndarray:
        if self.rotate == 90:
            return np.rot90(image, k=3)
        if self.rotate == 180:
            return np.rot90(image, k=2)
        if self.rotate == 270:
            return np.rot90(image, k=1)
        return image

    def get_intrinsics(self) -> Dict[str, float]:
        if self.profile is None:
            raise RuntimeError("Camera client is not started")

        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.ppx
        cy = intrinsics.ppy
        width = intrinsics.width
        height = intrinsics.height

        if self.rotate == 90:
            return {"fx": fy, "fy": fx, "cx": cy, "cy": width - cx, "width": height, "height": width}
        if self.rotate == 180:
            return {"fx": fx, "fy": fy, "cx": width - cx, "cy": height - cy, "width": width, "height": height}
        if self.rotate == 270:
            return {"fx": fy, "fy": fx, "cx": height - cy, "cy": cx, "width": height, "height": width}
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}

    def get_last_frame_meta(self) -> Dict[str, object]:
        return dict(self._last_frame_meta)

    def get_frame(
        self,
        *,
        pipeline: Optional[rs.pipeline] = None,
        timeout_ms: Optional[int] = None,
    ) -> Optional[CameraFrame]:
        active_pipeline = pipeline or self.pipeline
        if active_pipeline is None:
            raise RuntimeError("Camera client is not started")

        wait_timeout_ms = self.frame_timeout_ms if timeout_ms is None else int(timeout_ms)
        start_ts = time.perf_counter()

        try:
            frames = active_pipeline.wait_for_frames(timeout_ms=wait_timeout_ms)
            if self.align is not None:
                frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                self._last_frame_meta = {
                    "success": False,
                    "wait_ms": float(elapsed_ms),
                    "timeout_ms": wait_timeout_ms,
                    "error": "missing_color_or_depth",
                }
                return None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            if self.rotate in (90, 180, 270):
                color_image = self._rotate_image(color_image)
                depth_image = self._rotate_image(depth_image)

            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            self._last_frame_meta = {
                "success": True,
                "wait_ms": float(elapsed_ms),
                "timeout_ms": wait_timeout_ms,
                "error": None,
            }
            return CameraFrame(
                color_image=color_image,
                depth_image=depth_image,
                serial=self.active_serial or "",
                frame_number=int(color_frame.get_frame_number()),
                timestamp_ms=float(color_frame.get_timestamp()),
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            self._last_frame_meta = {
                "success": False,
                "wait_ms": float(elapsed_ms),
                "timeout_ms": wait_timeout_ms,
                "error": str(exc),
            }
            self._log(f"[RealSenseCameraClient] Error getting frame: {exc}")
            return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight RealSense RGB-D capture client")
    parser.add_argument("--serial", action="append", dest="serials", help="Preferred RealSense serial; repeat to provide fallbacks")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--frames", type=int, default=1, help="Number of frames to capture before exiting")
    parser.add_argument("--quiet", action="store_true", help="Suppress informational logs")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    client = RealSenseCameraClient(
        width=args.width,
        height=args.height,
        fps=args.fps,
        rotate=args.rotate,
        camera_serials=args.serials,
        verbose=not args.quiet,
    )

    try:
        with client:
            intrinsics = client.get_intrinsics()
            print(
                "[RealSenseCameraClient] Active camera "
                f"serial={client.active_serial} resolution={client.active_resolution} intrinsics={intrinsics}"
            )
            captured = 0
            while captured < args.frames:
                frame = client.get_frame()
                if frame is None:
                    continue
                captured += 1
                print(
                    "[RealSenseCameraClient] Frame "
                    f"{captured}/{args.frames} serial={frame.serial} "
                    f"color_shape={frame.color_image.shape} depth_shape={frame.depth_image.shape} "
                    f"depth_dtype={frame.depth_image.dtype} frame_number={frame.frame_number}"
                )
        return 0
    except Exception as exc:
        print(f"[RealSenseCameraClient] Failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
