#!/usr/bin/env python3

"""
Dual RealSense Charuco-only merge quality viewer.

Features
--------
- Auto-detects two connected RealSense devices
- Starts both cameras at the same resolution with fallback retries
- Uses DualCameraSystem for calibration / global-frame conversion
- Supports camera rotation CLI: 0 / 90 / 180 / 270 (clockwise)
- OpenCV-only live view:
        [0] Camera preview        – both colour feeds side-by-side
        [1] Charuco merge heatmap – board-corner 3D error (display-X vs Z)
        [2] Status strip          – detection/gate/metric diagnostics
- Metrics are computed ONLY from common Charuco IDs seen by both cameras.

Controls
--------
- Press 'q' in the OpenCV preview window to quit.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs  # type: ignore[import-not-found]
from dual_camera_system import DualCameraSystem


Resolution = Tuple[int, int]
Pixel = Tuple[int, int]

# ── visual tunables ──────────────────────────────────────────────────────────
_PREVIEW_HEIGHT = 560
_PANEL_WIDTH = 700
_PANEL_BG = (22, 27, 34)
_PANEL_GRID = (48, 54, 61)
_PANEL_TEXT = (201, 209, 217)
_STATUS_PANEL_H = 150

_BOARD_MIN_COMMON_CORNERS = 8
_BOARD_MIN_VALID_3D_CORNERS = 6
_BOARD_MIN_SPAN_RATIO_X = 0.20
_BOARD_MIN_SPAN_RATIO_Y = 0.20
_BOARD_HEATMAP_BINS_X = 32
_BOARD_HEATMAP_BINS_Z = 20
_BOARD_PLOT_EVERY = 2
_BOARD_MAX_ERROR_MM = 60.0
_BOARD_GOOD_MM = 35.0
_BOARD_WARN_MM = 75.0


def _default_log_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"charuco_merge_debug_{ts}.jsonl"


class DebugLogger:
    def __init__(self, path: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self._fh = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def log(self, event: str, payload: Dict[str, object]) -> None:
        if not self.enabled or self._fh is None:
            return
        row = {
            "ts_unix": time.time(),
            "event": event,
            "payload": payload,
        }
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _axis_bounds(a: np.ndarray, b: np.ndarray, pad_ratio: float = 0.08) -> Tuple[float, float]:
    vals = np.concatenate([a, b]) if a.size or b.size else np.array([0.0], dtype=np.float32)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    span = vmax - vmin
    if span < 1e-6:
        center = 0.5 * (vmin + vmax)
        half = 0.25
        return center - half, center + half
    pad = span * pad_ratio
    return vmin - pad, vmax + pad


def _status_color(status: str) -> Tuple[int, int, int]:
    if status == "OK":
        return (100, 210, 120)
    if status == "INSUFFICIENT_BOARD_COVERAGE":
        return (66, 181, 245)
    if status == "BOARD_NOT_DETECTED_BOTH":
        return (74, 90, 255)
    if status == "BOARD_MODE_DISABLED":
        return (139, 148, 158)
    return (139, 148, 158)


def _quality_label(median_mm: Optional[float]) -> str:
    if median_mm is None:
        return "N/A"
    if median_mm < _BOARD_GOOD_MM:
        return "GOOD"
    if median_mm < _BOARD_WARN_MM:
        return "WARN"
    return "BAD"


def _quality_color(label: str) -> Tuple[int, int, int]:
    if label == "GOOD":
        return (100, 210, 120)
    if label == "WARN":
        return (66, 181, 245)
    if label == "BAD":
        return (74, 90, 255)
    return (139, 148, 158)


def _error_to_color(err_mm: float, max_err_mm: float) -> Tuple[int, int, int]:
    # Piecewise green->yellow->red for intuitive mm-error view.
    t = float(np.clip(err_mm / max(1e-6, max_err_mm), 0.0, 1.0))
    if t < 0.5:
        a = t / 0.5
        b = int((1.0 - a) * 110 + a * 80)
        g = int((1.0 - a) * 200 + a * 210)
        r = int((1.0 - a) * 90 + a * 230)
    else:
        a = (t - 0.5) / 0.5
        b = int((1.0 - a) * 80 + a * 75)
        g = int((1.0 - a) * 210 + a * 80)
        r = int((1.0 - a) * 230 + a * 255)
    return (b, g, r)


def _render_heatmap_panel(metrics: Dict[str, object], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), _PANEL_BG, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 72, 14, 44, 36
    plot_w = max(1, width - margin_l - margin_r)
    plot_h = max(1, height - margin_t - margin_b)

    bins_x = int(metrics["bins_x"])
    bins_z = int(metrics["bins_z"])
    x_min = float(metrics["x_min"])
    x_max = float(metrics["x_max"])
    z_min = float(metrics["z_min"])
    z_max = float(metrics["z_max"])
    overlap_mask = metrics["overlap_mask"]
    cell_error_mm = metrics["cell_error_mm"]
    max_err_mm = float(metrics["max_error_mm"])

    cell_w = max(1, plot_w // max(1, bins_x))
    cell_h = max(1, plot_h // max(1, bins_z))

    cv2.rectangle(panel, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (33, 38, 45), 1)

    for iz in range(bins_z):
        y0 = margin_t + iz * cell_h
        y1 = min(margin_t + (iz + 1) * cell_h, margin_t + plot_h)
        if y0 >= margin_t + plot_h:
            continue
        for ix in range(bins_x):
            x0 = margin_l + ix * cell_w
            x1 = min(margin_l + (ix + 1) * cell_w, margin_l + plot_w)
            if x0 >= margin_l + plot_w:
                continue

            if bool(overlap_mask[iz, ix]):
                err_mm = float(cell_error_mm[iz, ix])
                color = _error_to_color(err_mm, max_err_mm) if np.isfinite(err_mm) else (63, 68, 76)
            else:
                color = (42, 47, 54)
            cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)

    for frac in (0.25, 0.5, 0.75):
        xg = margin_l + int(frac * plot_w)
        zg = margin_t + int(frac * plot_h)
        cv2.line(panel, (xg, margin_t), (xg, margin_t + plot_h), _PANEL_GRID, 1)
        cv2.line(panel, (margin_l, zg), (margin_l + plot_w, zg), _PANEL_GRID, 1)

    cv2.putText(panel, "Charuco-only heatmap (display-X vs Z)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _PANEL_TEXT, 1, cv2.LINE_AA)
    tick_color = (139, 148, 158)
    cv2.putText(panel, f"{x_min:+.2f}", (margin_l - 10, margin_t + plot_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, tick_color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"{x_max:+.2f}", (margin_l + plot_w - 34, margin_t + plot_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, tick_color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"{z_max:+.2f}", (8, margin_t + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, tick_color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"{z_min:+.2f}", (6, margin_t + plot_h), cv2.FONT_HERSHEY_SIMPLEX, 0.36, tick_color, 1, cv2.LINE_AA)
    cv2.putText(panel, "display-X (m)", (margin_l + max(4, plot_w // 2 - 40), height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _PANEL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, "Z depth (m)", (8, margin_t + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.36, _PANEL_TEXT, 1, cv2.LINE_AA)
    return panel


def _status_panel(metrics: Dict[str, object], width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), _PANEL_BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (36, 41, 49), 1)

    status = str(metrics["status"])
    board_detected = bool(metrics["board_detected"])
    common_count = int(metrics["common_count"])
    valid_3d_count = int(metrics["valid_3d_count"])
    span1x = float(metrics["span_ratio_cam1_x"])
    span1y = float(metrics["span_ratio_cam1_y"])
    span2x = float(metrics["span_ratio_cam2_x"])
    span2y = float(metrics["span_ratio_cam2_y"])
    median_mm = metrics["median_mm"]
    p90_mm = metrics["p90_mm"]
    quality = _quality_label(None if median_mm is None else float(median_mm))

    cv2.putText(panel, "Board-only merge diagnostics", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _PANEL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.47, _status_color(status), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (480, 8), (690, 44), (33, 38, 45), -1)
    cv2.rectangle(panel, (480, 8), (690, 44), (48, 54, 61), 1)
    cv2.putText(
        panel,
        f"Quality: {quality}",
        (492, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        _quality_color(quality),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(panel, f"Board both: {'yes' if board_detected else 'no'}", (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.44, _PANEL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Common IDs: {common_count}", (210, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.44, _PANEL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Valid 3D: {valid_3d_count}", (390, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.44, _PANEL_TEXT, 1, cv2.LINE_AA)

    cv2.putText(
        panel,
        f"Median: {median_mm:.1f} mm" if median_mm is not None else "Median: -",
        (10, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        _PANEL_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"P90: {p90_mm:.1f} mm" if p90_mm is not None else "P90: -",
        (210, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        _PANEL_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(panel, f"Span cam1 x/y: {span1x:.2f}/{span1y:.2f}", (390, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.44, _PANEL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Span cam2 x/y: {span2x:.2f}/{span2y:.2f}", (390, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.44, _PANEL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"Thresholds: GOOD < {_BOARD_GOOD_MM:.0f}mm, WARN < {_BOARD_WARN_MM:.0f}mm, BAD >= {_BOARD_WARN_MM:.0f}mm",
        (10, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (139, 148, 158),
        1,
        cv2.LINE_AA,
    )
    return panel


def _rotate_xy_for_display(points_xyz: np.ndarray, angle: int) -> np.ndarray:
    if points_xyz.size == 0 or angle == 0:
        return points_xyz

    out = points_xyz.copy()
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]

    if angle == 90:
        out[:, 0] = -y
        out[:, 1] = x
    elif angle == 180:
        out[:, 0] = -x
        out[:, 1] = -y
    elif angle == 270:
        out[:, 0] = y
        out[:, 1] = -x
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual RealSense Charuco-only merge viewer.")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="Clockwise rotation applied to preview and display axes.")
    parser.add_argument("--calibration-file", type=str, default="dual_camera_calibration.json", help="Path to dual-camera calibration JSON.")
    parser.add_argument("--force-recalibrate", action="store_true", help="Bypass any existing calibration file and recalibrate.")
    parser.add_argument("--num-pairs", type=int, default=35, help="Target number of valid image pairs for calibration.")
    parser.add_argument("--width", type=int, default=1280, help="Preferred width for startup attempt.")
    parser.add_argument("--height", type=int, default=720, help="Preferred height for startup attempt.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate for both cameras.")
    parser.add_argument("--swap-cameras", action="store_true", help="Swap auto-detected camera order if physical left/right are reversed.")

    parser.set_defaults(board_only_heatmap=True)
    parser.add_argument("--board-only-heatmap", dest="board_only_heatmap", action="store_true", help="Enable Charuco board-only heatmap mode (default).")
    parser.add_argument("--no-board-only-heatmap", dest="board_only_heatmap", action="store_false", help="Disable Charuco board-only heatmap mode.")

    parser.add_argument("--board-min-common-corners", type=int, default=_BOARD_MIN_COMMON_CORNERS, help="Minimum common Charuco IDs required.")
    parser.add_argument("--board-min-valid-3d-corners", type=int, default=_BOARD_MIN_VALID_3D_CORNERS, help="Minimum matched IDs with valid 3D in both cameras.")
    parser.add_argument("--board-min-span-ratio-x", type=float, default=_BOARD_MIN_SPAN_RATIO_X, help="Minimum matched-corner X span ratio per camera.")
    parser.add_argument("--board-min-span-ratio-y", type=float, default=_BOARD_MIN_SPAN_RATIO_Y, help="Minimum matched-corner Y span ratio per camera.")
    parser.add_argument("--board-heatmap-bins-x", type=int, default=_BOARD_HEATMAP_BINS_X, help="Horizontal bins for board heatmap.")
    parser.add_argument("--board-heatmap-bins-z", type=int, default=_BOARD_HEATMAP_BINS_Z, help="Vertical bins for board heatmap.")
    parser.add_argument("--board-plot-every", type=int, default=_BOARD_PLOT_EVERY, help="Refresh board panel every N frames.")
    parser.add_argument("--board-max-error-mm", type=float, default=_BOARD_MAX_ERROR_MM, help="Heatmap color cap in mm.")
    parser.add_argument("--debug-log-file", type=str, default=_default_log_path(), help="JSONL debug log path.")
    parser.add_argument("--debug-log-every", type=int, default=1, help="Write detailed debug rows every N board updates.")
    parser.add_argument("--debug-log", action="store_true", default=True, help="Enable detailed debug logging (default: on).")
    parser.add_argument("--no-debug-log", dest="debug_log", action="store_false", help="Disable detailed debug logging.")
    return parser.parse_args()


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


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation angle: {angle}")


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_LINEAR)


def _extract_common_charuco(
    color1: np.ndarray,
    color2: np.ndarray,
    system: DualCameraSystem,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    corners1, ids1 = system.detect_charuco(color1)
    corners2, ids2 = system.detect_charuco(color2)

    if corners1 is None or ids1 is None or corners2 is None or ids2 is None:
        return False, np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int32)

    ids1_flat = ids1.reshape(-1).astype(np.int32)
    ids2_flat = ids2.reshape(-1).astype(np.int32)
    common_ids = np.intersect1d(ids1_flat, ids2_flat)
    if common_ids.size == 0:
        return True, np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64), common_ids

    order1 = {int(cid): idx for idx, cid in enumerate(ids1_flat.tolist())}
    order2 = {int(cid): idx for idx, cid in enumerate(ids2_flat.tolist())}
    idx1 = np.array([order1[int(cid)] for cid in common_ids], dtype=np.int32)
    idx2 = np.array([order2[int(cid)] for cid in common_ids], dtype=np.int32)

    pix1 = corners1[idx1].reshape(-1, 2).astype(np.float64)
    pix2 = corners2[idx2].reshape(-1, 2).astype(np.float64)
    return True, pix1, pix2, common_ids


def _span_ratio(points_xy: np.ndarray, width: int, height: int) -> Tuple[float, float]:
    if points_xy.shape[0] == 0:
        return 0.0, 0.0
    span_x = float(np.max(points_xy[:, 0]) - np.min(points_xy[:, 0]))
    span_y = float(np.max(points_xy[:, 1]) - np.min(points_xy[:, 1]))
    return span_x / max(1.0, float(width)), span_y / max(1.0, float(height))


def _empty_metrics(args: argparse.Namespace) -> Dict[str, object]:
    bx = max(2, args.board_heatmap_bins_x)
    bz = max(2, args.board_heatmap_bins_z)
    return {
        "board_detected": False,
        "common_count": 0,
        "valid_3d_count": 0,
        "span_ratio_cam1_x": 0.0,
        "span_ratio_cam1_y": 0.0,
        "span_ratio_cam2_x": 0.0,
        "span_ratio_cam2_y": 0.0,
        "median_mm": None,
        "p90_mm": None,
        "status": "BOARD_NOT_DETECTED_BOTH",
        "bins_x": bx,
        "bins_z": bz,
        "x_min": -0.25,
        "x_max": 0.25,
        "z_min": 0.0,
        "z_max": 1.0,
        "overlap_mask": np.zeros((bz, bx), dtype=np.bool_),
        "cell_error_mm": np.full((bz, bx), np.nan, dtype=np.float32),
        "max_error_mm": float(max(1.0, args.board_max_error_mm)),
        "debug": {},
    }


def _charuco_board_metrics(
    system: DualCameraSystem,
    color1: np.ndarray,
    color2: np.ndarray,
    frames: Tuple[Optional[np.ndarray], Optional[rs.depth_frame], Optional[np.ndarray], Optional[rs.depth_frame]],
    rotate: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    metrics = _empty_metrics(args)

    if not bool(args.board_only_heatmap):
        metrics["status"] = "BOARD_MODE_DISABLED"
        return metrics

    board_detected, pix1_f, pix2_f, common_ids = _extract_common_charuco(color1, color2, system)
    metrics["board_detected"] = board_detected
    common_count = int(common_ids.size)
    metrics["common_count"] = common_count
    metrics["debug"] = {
        "common_ids": common_ids.astype(np.int32).tolist(),
        "gate_checks": {
            "min_common_required": int(args.board_min_common_corners),
            "min_valid_3d_required": int(args.board_min_valid_3d_corners),
            "min_span_ratio_x_required": float(args.board_min_span_ratio_x),
            "min_span_ratio_y_required": float(args.board_min_span_ratio_y),
        },
    }
    if not board_detected:
        return metrics

    h1, w1 = color1.shape[:2]
    h2, w2 = color2.shape[:2]
    span1x, span1y = _span_ratio(pix1_f, w1, h1)
    span2x, span2y = _span_ratio(pix2_f, w2, h2)
    metrics["span_ratio_cam1_x"] = span1x
    metrics["span_ratio_cam1_y"] = span1y
    metrics["span_ratio_cam2_x"] = span2x
    metrics["span_ratio_cam2_y"] = span2y

    gate_ok = (
        common_count >= int(args.board_min_common_corners)
        and span1x >= float(args.board_min_span_ratio_x)
        and span1y >= float(args.board_min_span_ratio_y)
        and span2x >= float(args.board_min_span_ratio_x)
        and span2y >= float(args.board_min_span_ratio_y)
    )
    metrics["debug"]["gate_values"] = {
        "common_count": common_count,
        "span_ratio_cam1_x": span1x,
        "span_ratio_cam1_y": span1y,
        "span_ratio_cam2_x": span2x,
        "span_ratio_cam2_y": span2y,
    }
    if not gate_ok:
        metrics["status"] = "INSUFFICIENT_BOARD_COVERAGE"
        return metrics

    pix1_i = np.round(pix1_f).astype(np.int32)
    pix2_i = np.round(pix2_f).astype(np.int32)
    pixels1: List[Pixel] = [(int(u), int(v)) for u, v in pix1_i.tolist()]
    pixels2: List[Pixel] = [(int(u), int(v)) for u, v in pix2_i.tolist()]

    p1 = system.pixel_to_3d_batch(pixels1, camera_id=1, frames=frames)
    p2 = system.pixel_to_3d_batch(pixels2, camera_id=2, frames=frames)

    finite_mask = np.isfinite(p1).all(axis=1) & np.isfinite(p2).all(axis=1)
    valid_3d_count = int(np.count_nonzero(finite_mask))
    metrics["valid_3d_count"] = valid_3d_count
    metrics["debug"]["depth_valid_mask"] = finite_mask.astype(np.int32).tolist()
    if valid_3d_count < int(args.board_min_valid_3d_corners):
        metrics["status"] = "INSUFFICIENT_BOARD_COVERAGE"
        return metrics

    p1v = p1[finite_mask]
    p2v = p2[finite_mask]
    err_mm = np.linalg.norm(p1v - p2v, axis=1) * 1000.0
    metrics["median_mm"] = float(np.median(err_mm))
    metrics["p90_mm"] = float(np.percentile(err_mm, 90.0))
    delta_xyz_mm = (p1v - p2v) * 1000.0
    metrics["debug"]["error_stats_mm"] = {
        "min": float(np.min(err_mm)),
        "max": float(np.max(err_mm)),
        "mean": float(np.mean(err_mm)),
        "std": float(np.std(err_mm)),
        "median": float(np.median(err_mm)),
        "p90": float(np.percentile(err_mm, 90.0)),
        "p95": float(np.percentile(err_mm, 95.0)),
    }
    metrics["debug"]["per_corner"] = {
        "corner_ids_valid": common_ids[finite_mask].astype(np.int32).tolist(),
        "error_mm": err_mm.astype(np.float32).tolist(),
        "delta_x_mm": delta_xyz_mm[:, 0].astype(np.float32).tolist(),
        "delta_y_mm": delta_xyz_mm[:, 1].astype(np.float32).tolist(),
        "delta_z_mm": delta_xyz_mm[:, 2].astype(np.float32).tolist(),
        "p1_xyz_m": p1v.astype(np.float32).tolist(),
        "p2_xyz_m": p2v.astype(np.float32).tolist(),
    }

    p_mid = 0.5 * (p1v + p2v)
    p_mid = _rotate_xy_for_display(p_mid, rotate)

    bins_x = max(2, int(args.board_heatmap_bins_x))
    bins_z = max(2, int(args.board_heatmap_bins_z))
    x_min, x_max = _axis_bounds(p_mid[:, 0], np.array([], dtype=np.float64))
    z_min, z_max = _axis_bounds(p_mid[:, 2], np.array([], dtype=np.float64))

    bx = np.floor((p_mid[:, 0] - x_min) / max(1e-9, x_max - x_min) * bins_x).astype(np.int32)
    bz = np.floor((p_mid[:, 2] - z_min) / max(1e-9, z_max - z_min) * bins_z).astype(np.int32)
    bx = np.clip(bx, 0, bins_x - 1)
    bz = np.clip(bz, 0, bins_z - 1)

    overlap_mask = np.zeros((bins_z, bins_x), dtype=np.bool_)
    cell_error_mm = np.full((bins_z, bins_x), np.nan, dtype=np.float32)
    for iz in range(bins_z):
        for ix in range(bins_x):
            m = (bx == ix) & (bz == iz)
            if np.any(m):
                overlap_mask[iz, ix] = True
                cell_error_mm[iz, ix] = float(np.median(err_mm[m]))

    metrics.update(
        {
            "status": "OK",
            "bins_x": bins_x,
            "bins_z": bins_z,
            "x_min": x_min,
            "x_max": x_max,
            "z_min": z_min,
            "z_max": z_max,
            "overlap_mask": overlap_mask,
            "cell_error_mm": cell_error_mm,
            "max_error_mm": float(max(1.0, args.board_max_error_mm)),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    logger = DebugLogger(args.debug_log_file, enabled=bool(args.debug_log))
    print(f"Debug log: {logger.path if args.debug_log else 'disabled'}")

    serial1, serial2 = detect_two_realsense_serials(args.swap_cameras)
    print(f"Using camera serials: cam1={serial1}, cam2={serial2}")

    system: DualCameraSystem | None = None

    try:
        window_name = "Dual Camera Merge Viewer (q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        system, used_resolution = start_dual_system_with_fallback(
            serial1=serial1,
            serial2=serial2,
            calibration_file=args.calibration_file,
            fps=args.fps,
            preferred_resolution=(args.width, args.height),
            force_recalibrate=args.force_recalibrate,
        )
        print(f"Resolution selected: {used_resolution[0]}x{used_resolution[1]}")
        logger.log(
            "session_start",
            {
                "serial1": serial1,
                "serial2": serial2,
                "resolution": [int(used_resolution[0]), int(used_resolution[1])],
                "fps": int(args.fps),
                "rotate": int(args.rotate),
                "args": vars(args),
            },
        )

        if not system.is_calibrated:
            print("No calibration loaded. Starting calibration...")
            success = system.calibrate(preview_rotate=args.rotate, num_valid_pairs=args.num_pairs)
            if not success:
                raise RuntimeError("Calibration was aborted or failed.")
        calib_report_obj = getattr(system, "calibration_report", {})
        if callable(calib_report_obj):
            calib_report = calib_report_obj()
        elif isinstance(calib_report_obj, dict):
            calib_report = dict(calib_report_obj)
        else:
            calib_report = {}
        logger.log(
            "calibration_info",
            {
                "is_calibrated": bool(system.is_calibrated),
                "calibration_report": calib_report,
                "rotation_cam2_to_cam1": (system.rotation.tolist() if system.rotation is not None else None),
                "translation_cam2_to_cam1": (system.translation.reshape(-1).tolist() if system.translation is not None else None),
            },
        )

        frame_idx = 0
        board_eval_idx = 0
        plot_panel = np.full((_PREVIEW_HEIGHT, _PANEL_WIDTH, 3), _PANEL_BG, dtype=np.uint8)

        while True:
            frames = system.get_aligned_frames()
            color1, _, color2, _ = frames
            if color1 is None or color2 is None:
                continue

            if frame_idx % max(1, args.board_plot_every) == 0:
                metrics = _charuco_board_metrics(system, color1, color2, frames, args.rotate, args)
                board_eval_idx += 1
                if args.debug_log and (board_eval_idx % max(1, args.debug_log_every) == 0):
                    logger.log(
                        "board_eval",
                        {
                            "frame_idx": int(frame_idx),
                            "board_eval_idx": int(board_eval_idx),
                            "status": str(metrics["status"]),
                            "quality_label": _quality_label(
                                None if metrics["median_mm"] is None else float(metrics["median_mm"])
                            ),
                            "board_detected": bool(metrics["board_detected"]),
                            "common_count": int(metrics["common_count"]),
                            "valid_3d_count": int(metrics["valid_3d_count"]),
                            "span_ratio_cam1_x": float(metrics["span_ratio_cam1_x"]),
                            "span_ratio_cam1_y": float(metrics["span_ratio_cam1_y"]),
                            "span_ratio_cam2_x": float(metrics["span_ratio_cam2_x"]),
                            "span_ratio_cam2_y": float(metrics["span_ratio_cam2_y"]),
                            "median_mm": (None if metrics["median_mm"] is None else float(metrics["median_mm"])),
                            "p90_mm": (None if metrics["p90_mm"] is None else float(metrics["p90_mm"])),
                            "debug": metrics.get("debug", {}),
                        },
                    )
                heat_panel = _render_heatmap_panel(metrics, _PANEL_WIDTH, _PREVIEW_HEIGHT - _STATUS_PANEL_H)
                status_panel = _status_panel(metrics, _PANEL_WIDTH, _STATUS_PANEL_H)
                plot_panel = cv2.vconcat([heat_panel, status_panel])

            frame_idx += 1

            disp1 = _resize_to_height(rotate_image(color1, args.rotate), _PREVIEW_HEIGHT)
            disp2 = _resize_to_height(rotate_image(color2, args.rotate), _PREVIEW_HEIGHT)
            cv2.rectangle(disp1, (0, 0), (disp1.shape[1] - 1, 6), (60, 134, 255), -1)
            cv2.rectangle(disp2, (0, 0), (disp2.shape[1] - 1, 6), (90, 90, 255), -1)

            preview_bgr = cv2.hconcat([disp1, disp2])
            combined_view = cv2.hconcat([preview_bgr, plot_panel])
            cv2.resizeWindow(window_name, combined_view.shape[1], combined_view.shape[0])
            cv2.imshow(window_name, combined_view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        logger.log("session_end", {})
        logger.close()
        if system is not None:
            system.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
