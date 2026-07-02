#!/usr/bin/env python3

"""RealSense device discovery and dual-camera startup with resolution fallback."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pyrealsense2 as rs  # type: ignore[import-not-found]
from dual_camera_system import DualCameraSystem

from dual_camera_viewer_config import Resolution


def detect_two_realsense_serials(swap_cameras: bool = False) -> Tuple[str, str]:
    ctx = rs.context()
    devices = ctx.query_devices()

    serials: List[str] = []
    for dev in devices:
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
        except Exception:
            continue
        if serial:
            serials.append(serial)

    if len(serials) < 2:
        raise RuntimeError(f"Found {len(serials)} RealSense device(s). Need at least 2 connected.")

    serial1, serial2 = serials[0], serials[1]
    if swap_cameras:
        serial1, serial2 = serial2, serial1

    return serial1, serial2


def build_resolution_attempts(preferred: Resolution) -> List[Resolution]:
    attempts: List[Resolution] = [preferred]
    fallbacks: Sequence[Resolution] = (
        (1280, 720),
        (848, 480),
        (640, 480),
        (424, 240),
        (320, 240),
    )
    for resolution in fallbacks:
        if resolution not in attempts:
            attempts.append(resolution)
    return attempts


def start_dual_system_with_fallback(
    serial1: str,
    serial2: str,
    calibration_file: str,
    fps: int,
    preferred_resolution: Resolution,
    force_recalibrate: bool = False,
) -> Tuple[DualCameraSystem, Resolution]:

    attempts = build_resolution_attempts(preferred_resolution)
    last_error: Exception | None = None

    for width, height in attempts:
        system = DualCameraSystem(
            serial1=serial1,
            serial2=serial2,
            calibration_file=calibration_file,
            width=width,
            height=height,
            fps=fps,
            force_recalibrate=force_recalibrate,
        )
        try:
            system.start()
            print(f"Started DualCameraSystem at {width}x{height} @ {fps}fps")
            return system, (width, height)
        except Exception as exc:
            last_error = exc
            try:
                system.stop()
            except Exception:
                pass
            print(f"Failed to start at {width}x{height}: {exc}")

    raise RuntimeError(
        "Unable to start dual cameras at any supported fallback resolution. "
        f"Last error: {last_error}"
    )
