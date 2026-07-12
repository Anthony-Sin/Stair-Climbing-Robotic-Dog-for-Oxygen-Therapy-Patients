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


# --- 2026-07-12 review: run-18 wedge (run_sim_20260712_103237_267, sim_t=40.88) --------
# At close range (<~0.45 m) a genuine riser FACE fills the whole ROI -- no tread is
# visible below it to produce the downward gradient the riser test relies on -- so it
# reads exactly like a flat body/wall slab (observed stairs_loss_nearfield_depth_m=0.304,
# gradient=-1.0). The context override lets a COMMITTED climb whose own confirmed
# depth-stair leading edge agrees with this near reading (|0.304-0.302|=0.002 m in run
# 18) reclassify it as the riser, not a wall. These tests use the SAME uniform-slab ROI
# as test_close_body_slab_blocks (flat -> would block on the gradient test alone) to
# isolate the override logic.


def _flat_near_slab(depth_m: float):
    x1, y1, x2, y2 = _roi_bounds()
    img = _blank()
    img[y1:y2, x1:x2] = int(depth_m * 1000)  # m -> mm
    return img


def test_committed_context_with_agreeing_leading_edge_releases_block():
    # At/beyond NEARFIELD_RISER_HOLD_STANDOFF_M (0.45) the run-18 context override
    # releases the block so the loss floor can drive toward the riser.
    img = _flat_near_slab(0.50)
    dbg = {}
    blocked = _stair_loss_forward_block(
        _Args(), img, dbg,
        committed_climb=True,
        depth_stair_confirmed=True,
        depth_stair_leading_edge_m=0.52,
    )
    assert blocked is False
    assert dbg["stairs_loss_nearfield_leading_edge_agree"] is True
    assert dbg["stairs_loss_nearfield_riser"] is True


def test_committed_context_inside_hold_standoff_reblocks():
    # Run 21 (run_sim_20260712_112519_790): the stage-5 climber cannot mount from a
    # dead-stand press at 0.30 m (climb_stalled all retries, worst pitch 7.3 deg =
    # never reared), so INSIDE NEARFIELD_RISER_HOLD_STANDOFF_M the override no longer
    # applies -- the dog holds at approach distance and engages with run-up instead.
    img = _flat_near_slab(0.304)
    dbg = {}
    blocked = _stair_loss_forward_block(
        _Args(), img, dbg,
        committed_climb=True,
        depth_stair_confirmed=True,
        depth_stair_leading_edge_m=0.302,
    )
    assert blocked is True
    assert dbg["stairs_loss_nearfield_leading_edge_agree"] is True
    assert dbg["stairs_loss_nearfield_riser"] is False


def test_context_override_requires_committed_climb():
    # Same confirmed + agreeing leading edge, but NOT a committed climb window -> still
    # blocked (fails toward blocking, CLAUDE.md 8.8).
    img = _flat_near_slab(0.304)
    dbg = {}
    blocked = _stair_loss_forward_block(
        _Args(), img, dbg,
        committed_climb=False,
        depth_stair_confirmed=True,
        depth_stair_leading_edge_m=0.302,
    )
    assert blocked is True
    assert dbg["stairs_loss_nearfield_leading_edge_agree"] is False


def test_context_override_requires_confirmed_structure():
    # Committed + agreeing leading edge, but the geometric gate did NOT confirm a
    # multi-riser structure this frame -> still blocked. A stray near reading with no
    # real staircase behind it must not be reclassified.
    img = _flat_near_slab(0.304)
    dbg = {}
    blocked = _stair_loss_forward_block(
        _Args(), img, dbg,
        committed_climb=True,
        depth_stair_confirmed=False,
        depth_stair_leading_edge_m=0.302,
    )
    assert blocked is True
    assert dbg["stairs_loss_nearfield_leading_edge_agree"] is False


def test_context_override_requires_leading_edge_agreement():
    # Committed + confirmed, but the confirmed structure's leading edge does NOT agree
    # with this near reading (a genuine body/wall coincidentally close while a staircase
    # is separately confirmed elsewhere in the ROI) -> still blocked.
    img = _flat_near_slab(0.304)
    dbg = {}
    blocked = _stair_loss_forward_block(
        _Args(), img, dbg,
        committed_climb=True,
        depth_stair_confirmed=True,
        depth_stair_leading_edge_m=1.2,
    )
    assert blocked is True
    assert dbg["stairs_loss_nearfield_leading_edge_agree"] is False


def test_context_override_missing_leading_edge_still_blocks():
    # Committed + confirmed, but no leading-edge reading this frame (None) -> cannot
    # agree with anything -> still blocked.
    img = _flat_near_slab(0.304)
    dbg = {}
    blocked = _stair_loss_forward_block(
        _Args(), img, dbg,
        committed_climb=True,
        depth_stair_confirmed=True,
        depth_stair_leading_edge_m=None,
    )
    assert blocked is True


def test_context_override_never_downgrades_a_genuine_riser_read():
    # The gradient test already passes (genuine riser profile, as in
    # test_close_riser_does_not_block) -- the override only ever RELEASES a would-block
    # verdict, it never has anything to do once is_riser is already True.
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
    blocked = _stair_loss_forward_block(
        _Args(), img, dbg,
        committed_climb=False,
        depth_stair_confirmed=False,
        depth_stair_leading_edge_m=None,
    )
    assert blocked is False
    assert dbg["stairs_loss_nearfield_riser"] is True


if __name__ == "__main__":
    test_close_body_slab_blocks()
    test_close_riser_does_not_block()
    test_far_return_does_not_block()
    test_no_depth_frame_does_not_block()
    test_committed_context_with_agreeing_leading_edge_releases_block()
    test_context_override_requires_committed_climb()
    test_context_override_requires_confirmed_structure()
    test_context_override_requires_leading_edge_agreement()
    test_context_override_missing_leading_edge_still_blocks()
    test_context_override_never_downgrades_a_genuine_riser_read()
    print("OK")
