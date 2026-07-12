"""Regression tests for the 2026-07-12 run-18 wedge review (CLAUDE.md ledger).

Runs 15-18 all settled short of riser 1 (base riser at x=2.0) without ever engaging the
staircase. Run 18 (run_sim_20260712_103237_267) got the dog to within 0.30 m of riser 1 --
closer than any prior run -- then wedged behind a cluster of guards that misread the riser
itself. Host-safe (no Isaac/hardware deps): exercises the pure functions core/main.py wires
into the STAIR_LOSS_FLOOR dispatch and the too-close stance-lock.

  Fix A. _stair_loss_forward_block's context override (test_stair_loss_guard.py) --
      the near-field riser-vs-wall gradient test structurally cannot see a tread when the
      riser face fills the whole ROI at close range, so it reads a genuine riser exactly
      like a flat wall. Covered separately in test_stair_loss_guard.py; not repeated here.
  Fix B. too_close_riser_gap_suppressed (this file) -- a too-close stance-lock grounded in
      a STALE median gap must not fire when that gap agrees with a confirmed staircase's
      own leading edge while nobody is even detected (mirrors incident 8.3 in the opposite
      direction: there a person was misread as stairs; here a stair reading risks being
      misread as the person).

Decisive frame numbers (vision_main_trace.jsonl, frame_timing, sim_t=40.88,
run_sim_20260712_103237_267): stairs_loss_nearfield_depth_m=0.304,
depth_stair_leading_edge_m=0.302, standoff_gap_ctrl_m=0.7246, standoff_lower_bound_m=0.85,
person_detected=False, depth_stair_confirmed=True. NOTE (verified numerically, CLAUDE.md
"verify geometric claims numerically"): standoff_gap_ctrl_m (0.7246, a MEDIAN of
pre-loss person-gap history) does NOT itself agree with depth_stair_leading_edge_m (0.302,
diff 0.42 m >> NEARFIELD_LEADING_EDGE_AGREE_M) at this specific frame -- it is a stale but
genuine last-seen person gap, not a riser-contaminated reading, so too_close_riser_gap_
suppressed correctly does NOT fire on THIS frame's live fused gap (test_run18_fused_gap_
does_not_agree_with_leading_edge below encodes exactly that). The suppression is exercised
here instead against the CLASS of contamination it targets: a fused gap that DOES land on
the confirmed leading edge while the person is not detected.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.control.stair_policy import (  # noqa: E402
    NEARFIELD_LEADING_EDGE_AGREE_M,
    too_close_riser_gap_suppressed,
)


def test_riser_gap_suppressed_when_agreeing_and_person_not_detected():
    # Fused gap lands on the confirmed staircase's own leading edge while nobody is
    # detected -- that gap is the riser, not the patient. Suppress.
    assert too_close_riser_gap_suppressed(
        person_detected=False,
        gap_for_hold=0.31,
        depth_stair_confirmed=True,
        depth_stair_leading_edge_m=0.30,
    ) is True


def test_not_suppressed_when_person_detected():
    # Ordinary case: a person IS visible and close -- the too-close stance-lock must be
    # untouched regardless of any coincidental staircase reading.
    assert too_close_riser_gap_suppressed(
        person_detected=True,
        gap_for_hold=0.31,
        depth_stair_confirmed=True,
        depth_stair_leading_edge_m=0.30,
    ) is False


def test_not_suppressed_without_confirmed_structure():
    # A near reading with no independently-confirmed multi-riser structure behind it is
    # not enough -- fails toward keeping the hold (CLAUDE.md 8.8).
    assert too_close_riser_gap_suppressed(
        person_detected=False,
        gap_for_hold=0.31,
        depth_stair_confirmed=False,
        depth_stair_leading_edge_m=0.30,
    ) is False


def test_not_suppressed_when_gap_disagrees_with_leading_edge():
    # The fused gap and the confirmed staircase's leading edge describe two different
    # surfaces -- do not suppress a proximity hold on an unrelated close reading.
    assert too_close_riser_gap_suppressed(
        person_detected=False,
        gap_for_hold=1.5,
        depth_stair_confirmed=True,
        depth_stair_leading_edge_m=0.30,
    ) is False


def test_not_suppressed_on_missing_readings():
    assert too_close_riser_gap_suppressed(
        person_detected=False, gap_for_hold=None,
        depth_stair_confirmed=True, depth_stair_leading_edge_m=0.30,
    ) is False
    assert too_close_riser_gap_suppressed(
        person_detected=False, gap_for_hold=0.31,
        depth_stair_confirmed=True, depth_stair_leading_edge_m=None,
    ) is False


def test_run18_fused_gap_does_not_agree_with_leading_edge():
    # Run 18's ACTUAL wedge-frame numbers (vision_main_trace.jsonl, sim_t=40.88):
    # standoff_gap_ctrl_m=0.7246 is a stale median of PRE-loss person-gap samples, not a
    # riser-contaminated reading -- verified numerically (CLAUDE.md "verify geometric
    # claims numerically") rather than assumed. The diff to the leading edge (0.302) is
    # 0.4226 m, far outside NEARFIELD_LEADING_EDGE_AGREE_M, so the suppression correctly
    # stays OFF here: this frame's too_close_hold=True reflects a genuinely stale (but not
    # riser-fused) last-seen gap, not the bug this fix targets. (The too_close assertion
    # is inert for THIS dispatch branch regardless -- STAIR_LOSS_FLOOR's controller.move
    # call sends its own literal hold=False -- see the Fix C analysis for the full chain.)
    assert too_close_riser_gap_suppressed(
        person_detected=False,
        gap_for_hold=0.7245999865531921,
        depth_stair_confirmed=True,
        depth_stair_leading_edge_m=0.302,
    ) is False


def test_agreement_tolerance_boundary():
    edge = 0.30
    just_inside = edge + NEARFIELD_LEADING_EDGE_AGREE_M - 1e-6
    just_outside = edge + NEARFIELD_LEADING_EDGE_AGREE_M + 1e-3
    assert too_close_riser_gap_suppressed(
        person_detected=False, gap_for_hold=just_inside,
        depth_stair_confirmed=True, depth_stair_leading_edge_m=edge,
    ) is True
    assert too_close_riser_gap_suppressed(
        person_detected=False, gap_for_hold=just_outside,
        depth_stair_confirmed=True, depth_stair_leading_edge_m=edge,
    ) is False


if __name__ == "__main__":
    test_riser_gap_suppressed_when_agreeing_and_person_not_detected()
    test_not_suppressed_when_person_detected()
    test_not_suppressed_without_confirmed_structure()
    test_not_suppressed_when_gap_disagrees_with_leading_edge()
    test_not_suppressed_on_missing_readings()
    test_run18_fused_gap_does_not_agree_with_leading_edge()
    test_agreement_tolerance_boundary()
    print("OK")
