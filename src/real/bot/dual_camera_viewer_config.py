#!/usr/bin/env python3

"""Type aliases and visual/gate tunables for the dual-camera merge viewer."""

from __future__ import annotations

from typing import Tuple


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
