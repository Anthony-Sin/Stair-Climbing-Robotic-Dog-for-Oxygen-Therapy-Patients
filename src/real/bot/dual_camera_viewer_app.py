#!/usr/bin/env python3

"""CLI argument parsing and the interactive dual-camera viewer main loop."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from dual_camera_system import DualCameraSystem

from dual_camera_viewer_config import (
    _BOARD_HEATMAP_BINS_X,
    _BOARD_HEATMAP_BINS_Z,
    _BOARD_MAX_ERROR_MM,
    _BOARD_MIN_COMMON_CORNERS,
    _BOARD_MIN_SPAN_RATIO_X,
    _BOARD_MIN_SPAN_RATIO_Y,
    _BOARD_MIN_VALID_3D_CORNERS,
    _BOARD_PLOT_EVERY,
    _PANEL_BG,
    _PANEL_WIDTH,
    _PREVIEW_HEIGHT,
    _STATUS_PANEL_H,
)
from dual_camera_viewer_metrics import (
    DebugLogger,
    _charuco_board_metrics,
    _default_log_path,
)
from dual_camera_viewer_pipeline import (
    detect_two_realsense_serials,
    start_dual_system_with_fallback,
)
from dual_camera_viewer_render import (
    _quality_label,
    _render_heatmap_panel,
    _resize_to_height,
    _status_panel,
    rotate_image,
)


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
