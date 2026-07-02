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

from dual_camera_viewer_app import (  # noqa: F401
    main,
    parse_args,
)
from dual_camera_viewer_config import (  # noqa: F401
    Pixel,
    Resolution,
    _BOARD_GOOD_MM,
    _BOARD_HEATMAP_BINS_X,
    _BOARD_HEATMAP_BINS_Z,
    _BOARD_MAX_ERROR_MM,
    _BOARD_MIN_COMMON_CORNERS,
    _BOARD_MIN_SPAN_RATIO_X,
    _BOARD_MIN_SPAN_RATIO_Y,
    _BOARD_MIN_VALID_3D_CORNERS,
    _BOARD_PLOT_EVERY,
    _BOARD_WARN_MM,
    _PANEL_BG,
    _PANEL_GRID,
    _PANEL_TEXT,
    _PANEL_WIDTH,
    _PREVIEW_HEIGHT,
    _STATUS_PANEL_H,
)
from dual_camera_viewer_metrics import (  # noqa: F401
    DebugLogger,
    _axis_bounds,
    _charuco_board_metrics,
    _default_log_path,
    _empty_metrics,
    _extract_common_charuco,
    _rotate_xy_for_display,
    _span_ratio,
)
from dual_camera_viewer_pipeline import (  # noqa: F401
    build_resolution_attempts,
    detect_two_realsense_serials,
    start_dual_system_with_fallback,
)
from dual_camera_viewer_render import (  # noqa: F401
    _error_to_color,
    _quality_color,
    _quality_label,
    _render_heatmap_panel,
    _resize_to_height,
    _status_color,
    _status_panel,
    rotate_image,
)


if __name__ == "__main__":
    main()
