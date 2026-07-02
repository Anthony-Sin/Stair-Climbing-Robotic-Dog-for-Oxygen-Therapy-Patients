"""Regression tests for the extracted depth stair gate (host-safe, no Isaac).

The 2,216-line control loop had ZERO tests over it, including the exact code of both
incident-8.3 field bugs. Those are now in the pure evaluate_depth_stair_gate, pinned here:

  1. UNITS (P2-2): the detector treats its grid as METRES but the D435 depth is uint16
     MILLIMETRES. The gate converts mm->m; feeding a metre grid (as if it were mm) must
     NOT confirm (it did, silently, on 0 frames before the fix).
  2. PERSON FALSE-STAIR (incident 8.3): a person slab ~0.6 m ahead back-projects into a
     stack of fake risers -> confirmed on flat ground; masking the person's bbox fixes it.
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling test helpers

from go2_locomotion.pgtt_stair_handoff import DepthStairDetector, HandoffConfig  # noqa: E402
from core.control.stair_policy import evaluate_depth_stair_gate, depth_stair_latch_allowed  # noqa: E402
from test_pgtt_stair_handoff import synth_staircase_depth  # noqa: E402

_H, _W = 60, 106


def _detector():
    return DepthStairDetector(HandoffConfig())


def _min_count():
    return int(HandoffConfig().stair_min_count)


def _person_slab_mm(depth_mm=600.0, band_frac=0.6, r0=6, r1=54):
    """A vertical person slab at a constant depth in the central column band (uint16 mm)."""
    D = np.zeros((_H, _W), dtype=np.float32)
    c0 = int(round(_W * (0.5 - band_frac / 2.0)))
    c1 = int(round(_W * (0.5 + band_frac / 2.0)))
    D[r0:r1, c0:c1] = depth_mm
    return D, (c0, r0, c1, r1)


def test_millimetre_grid_confirms_but_metre_grid_does_not():
    """The gate expects MILLIMETRES: a mm staircase confirms; the same values fed as if
    they were metres (i.e. actually a metre grid) get *0.001'd into micro-depths and the
    detector's range filter empties -> not confirmed. This pins the P2-2 units contract."""
    stair_m = synth_staircase_depth()          # metres, as Isaac/the detector's own frame
    stair_mm = stair_m * 1000.0                 # what the D435 actually delivers (uint16 mm)
    g_mm = evaluate_depth_stair_gate(stair_mm, None, _detector(), min_count=_min_count())
    assert g_mm.confirmed, f"a millimetre staircase must confirm after mm->m: {g_mm.result}"
    g_m = evaluate_depth_stair_gate(stair_m, None, _detector(), min_count=_min_count())
    assert not g_m.confirmed, "a metre grid fed as mm must NOT confirm (wrong units caught)"


def test_person_slab_false_stair_is_fixed_by_masking():
    """Incident 8.3: an unmasked person slab reads as a multi-riser staircase on flat
    ground; zeroing the person's bbox before detect() removes the false stairs."""
    slab_mm, bbox = _person_slab_mm()
    g_unmasked = evaluate_depth_stair_gate(slab_mm, None, _detector(), min_count=_min_count())
    assert g_unmasked.confirmed, "sanity: the unmasked person slab must reproduce the false stair"
    assert g_unmasked.result["stair_count"] >= 2

    g_masked = evaluate_depth_stair_gate(slab_mm, list(bbox), _detector(), min_count=_min_count())
    assert g_masked.person_masked
    assert not g_masked.confirmed, f"masking the person must clear the false stair: {g_masked.result}"


def test_gate_does_not_mutate_input():
    """The mm->m + mask must operate on a copy; the caller's depth frame stays untouched."""
    slab_mm, bbox = _person_slab_mm()
    before = slab_mm.copy()
    evaluate_depth_stair_gate(slab_mm, list(bbox), _detector(), min_count=_min_count())
    assert np.array_equal(slab_mm, before), "gate must not mutate the input depth image"


def test_gate_returns_raw_detector_result():
    """The raw detector dict is returned (the loop reuses its leading_edge downstream)."""
    stair_mm = synth_staircase_depth() * 1000.0
    g = evaluate_depth_stair_gate(stair_mm, None, _detector(), min_count=_min_count())
    assert "leading_edge_distance" in g.result and "stair_count" in g.result


def test_depth_latch_gated_on_recent_yolo():
    """The depth-only latch (incident 8.3 residual fix): a depth confirmation may latch stair
    mode ONLY when YOLO has corroborated stairs within persist_sec. This is what stops
    near-floor slivers latching stair mode on flat ground and killing plain-follow."""
    P = 8.0  # persist_sec
    # Depth confirms but YOLO has NEVER seen stairs (the run_..142645 case: false latch at
    # frame 14, YOLO's first real detection frame 517) -> BLOCKED.
    assert depth_stair_latch_allowed(depth_confirmed=True, now=100.0, last_yolo_stair_ts=-1e9, persist_sec=P) is False
    # Depth confirms and YOLO saw stairs 2 s ago (real staircase, close range) -> allowed.
    assert depth_stair_latch_allowed(depth_confirmed=True, now=100.0, last_yolo_stair_ts=98.0, persist_sec=P) is True
    # YOLO went stale (> persist_sec ago) -> depth may no longer carry the latch.
    assert depth_stair_latch_allowed(depth_confirmed=True, now=100.0, last_yolo_stair_ts=90.0, persist_sec=P) is False
    # No depth confirmation -> never latches regardless of YOLO recency.
    assert depth_stair_latch_allowed(depth_confirmed=False, now=100.0, last_yolo_stair_ts=99.9, persist_sec=P) is False


if __name__ == "__main__":
    test_millimetre_grid_confirms_but_metre_grid_does_not()
    test_person_slab_false_stair_is_fixed_by_masking()
    test_gate_does_not_mutate_input()
    test_gate_returns_raw_detector_result()
    test_depth_latch_gated_on_recent_yolo()
    print("ALL DEPTH STAIR GATE TESTS PASS")
