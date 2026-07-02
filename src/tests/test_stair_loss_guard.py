"""Regression tests for the person-loss stair forward-drive guard (host-safe).

Covers the P0-4 safety hole from the review: the STAIR_LOSS_FLOOR path used to drive a
blind forward floor UP the stairs gated only on the STALE last-known patient gap, so a
patient who stopped on the step just ahead while detection dropped got walked into. The
new live near-field guard (_stair_loss_forward_block) blocks a close body/wall while NOT
blocking a real stair riser (which also reads "near" but has a strong vertical gradient).
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.control.stair_policy import _stair_loss_forward_block
from core.vision.depth_processor import DepthProcessor

_H, _W = 720, 1280


class _Args:
    # Only the fields the guard reads. obstacle_stop_distance in metres; ROI ratios match
    # the front-obstacle gate defaults.
    obstacle_roi_width_ratio = 0.24
    obstacle_roi_height_ratio = 0.42
    obstacle_stop_distance = 0.6


def _roi_bounds():
    _, info = DepthProcessor.central_roi_nearest_depth(
        np.zeros((_H, _W), np.uint16),
        width_ratio=_Args.obstacle_roi_width_ratio,
        height_ratio=_Args.obstacle_roi_height_ratio,
    )
    return info["roi"]  # (x1, y1, x2, y2)


def _blank():
    return np.zeros((_H, _W), np.uint16)


def test_close_body_slab_blocks():
    # A uniform slab (a person) at 0.5 m fills the near ROI: close + flat -> BLOCK.
    x1, y1, x2, y2 = _roi_bounds()
    img = _blank()
    img[y1:y2, x1:x2] = 500  # mm
    dbg = {}
    assert _stair_loss_forward_block(_Args(), img, dbg) is True
    assert dbg["stairs_loss_nearfield_block"] is True
    assert dbg["stairs_loss_nearfield_riser"] is False


def test_close_riser_does_not_block():
    # A riser: near face (0.3 m) across the top ~30% of the ROI, tread receding to ~7 m
    # below -> close BUT a strong downward gradient => recognised as a riser, NOT blocked.
    x1, y1, x2, y2 = _roi_bounds()
    img = _blank()
    n = y2 - y1
    near_rows = int(0.3 * n)
    for i in range(n):
        if i < near_rows:
            val = 300
        else:
            val = int(300 + (i - near_rows) / max(1, (n - near_rows)) * 6700)
        img[y1 + i, x1:x2] = val
    dbg = {}
    assert _stair_loss_forward_block(_Args(), img, dbg) is False
    assert dbg["stairs_loss_nearfield_riser"] is True


def test_far_return_does_not_block():
    # Nothing close ahead (uniform 3 m) -> no block; the climb floor is allowed.
    x1, y1, x2, y2 = _roi_bounds()
    img = _blank()
    img[y1:y2, x1:x2] = 3000  # mm = 3 m > stop distance
    dbg = {}
    assert _stair_loss_forward_block(_Args(), img, dbg) is False
    assert dbg["stairs_loss_nearfield_block"] is False


def test_no_depth_frame_does_not_block():
    # No depth this frame -> the guard is a no-op (the other loss guards still apply).
    assert _stair_loss_forward_block(_Args(), None, {}) is False


if __name__ == "__main__":
    test_close_body_slab_blocks()
    test_close_riser_does_not_block()
    test_far_return_does_not_block()
    test_no_depth_frame_does_not_block()
    print("OK")
