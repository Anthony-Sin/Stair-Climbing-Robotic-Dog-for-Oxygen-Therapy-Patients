"""Shared layout/geometry constants and preset defaults for the stair-sweep presenter.

Split out of ``sweep_present.py`` (single-responsibility): the canvas/grid geometry,
per-riser labels, verdict colors, and the commercial-footprint defaults the sweep holds
constant. Also re-exports the analyze_climb thresholds so this package has ONE source of
truth for them.
"""
import os
import sys

# Sibling import (analyze_climb.py lives next to this file) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_climb import STAIR_BASE_X, STEP_RUN, FALL_TILT_DEG, COLLAPSE_H_M  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(REPO_ROOT, "log")

# Commercial footprint the sweep holds constant (only the riser changes). Used as defaults
# when the per-run stair_preset_configured event cannot be parsed -- these match the preset
# run_stair_sweep.ps1 hard-codes, so the fallback is correct for this sweep.
DEFAULT_STEP_DEPTH_M = STEP_RUN     # 0.305
DEFAULT_STEP_COUNT = 14
DEFAULT_TOP_EDGE_X = 6.27           # forward x of the top step edge (constant across risers)

# Candidate hero clips per episode, best first. scene_view = cinematic chase (hero shot);
# topdown = autofit overview; follow_view = tracking chase.
VIDEO_PREFERENCE = ("scene_view.mp4", "topdown.mp4", "follow_view.mp4")

# Short human label per riser (m -> tag). Falls back to inches if unmatched.
RISER_SHORT = {
    0.100: '4"  gentle rise',
    0.125: '5"  hospital std',
    0.150: '6"  ADA maximum',
    0.178: '7"  IBC standard',
    0.198: '7.75"  comm. max',
}

VERDICT_COLORS = {
    "CLEAN": "#2e7d32",
    "COLLIDED": "#ef6c00",
    "FELL": "#c62828",
    "NO REACH": "#546e7a",
    "INCOMPLETE": "#9e9e9e",
    "NO DATA": "#bdbdbd",
}

# montage geometry
CANVAS_W, CANVAS_H = 1920, 1080
COLS, ROWS = 2, 3
CELL_W, CELL_H = CANVAS_W // COLS, CANVAS_H // ROWS   # 960 x 360
CAPTION_BAR_H = 64
