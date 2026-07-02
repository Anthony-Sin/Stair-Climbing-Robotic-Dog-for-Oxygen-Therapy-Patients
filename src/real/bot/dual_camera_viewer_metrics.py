#!/usr/bin/env python3

"""Frame processing, Charuco extraction, board merge metrics, and JSONL logging."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrealsense2 as rs  # type: ignore[import-not-found]
from dual_camera_system import DualCameraSystem

from dual_camera_viewer_config import Pixel


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
