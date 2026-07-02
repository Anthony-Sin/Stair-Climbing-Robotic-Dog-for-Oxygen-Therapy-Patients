"""Shared constants and optional cv2/numpy imports for analyze_run.

Moved verbatim from analyze_run.py as part of a pure structural split.
Import cv2/np/_CV2_AVAILABLE from here so the optional-dependency probe
runs exactly once.
"""

from __future__ import annotations

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    np = None   # type: ignore[assignment]
    _CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAIR_START_X_M = 2.0   # fixed across all presets (see isaac_env.py)
VIDEO_NAMES = ("scene_view.mp4", "topdown.mp4", "lidar_preview.mp4", "opencv_preview.mp4")
VERIFICATION_PNGS = ("verification_start.png", "verification_end.png")
THUMB_W, THUMB_H = 384, 216
LABEL_HEIGHT = 22
FRAMES_PER_VIDEO = 4   # start, 1/3, 2/3, end
