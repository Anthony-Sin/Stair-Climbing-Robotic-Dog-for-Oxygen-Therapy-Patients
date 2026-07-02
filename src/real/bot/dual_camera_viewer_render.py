#!/usr/bin/env python3

"""Rendering/overlay helpers: color mapping, heatmap/status panels, image ops."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from dual_camera_viewer_config import (
    _BOARD_GOOD_MM,
    _BOARD_WARN_MM,
    _PANEL_BG,
    _PANEL_GRID,
    _PANEL_TEXT,
)


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
