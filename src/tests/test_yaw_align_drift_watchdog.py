"""Regression tests for the yaw-align drift watchdog (host-safe).

Task (2026-07-12, run-28 review follow-up, run_sim_20260712_141230_357): the controller-side
edge_block veto that used to gate the post-crest face-the-patient rotation
(landing_face_patient_align / landing_visible_person_centering, core/control/stair_policy.py)
is removed -- see those functions' EDGE-GUARD PRECEDENCE docstring paragraphs. Its replacement
is a SIM-SIDE watchdog (go2_locomotion/yaw_align_drift.py) that measures the robot's actual
planar displacement during an in-place yaw-align turn and PERMANENTLY revokes the carve-out if
it ever exceeds a small bound -- a genuine in-place turn should not translate the body at all.
See that module's docstring for the full root cause / rationale.
"""
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from go2_locomotion.yaw_align_drift import (
    YawAlignDriftConfig,
    YawAlignDriftWatchdog,
)


def _watchdog(drift_max_m=0.15):
    return YawAlignDriftWatchdog(YawAlignDriftConfig(drift_max_m=drift_max_m))


def test_not_aligning_never_allows_and_has_no_anchor():
    wd = _watchdog()
    r = wd.update(aligning=False, x=0.0, y=0.0)
    assert r.allow_align is False
    assert r.tripped_this_frame is False
    assert r.tripped is False
    assert r.drift_m == 0.0
    assert wd.tripped is False


def test_first_aligning_frame_anchors_and_allows_with_zero_drift():
    # The very first frame of a rotation burst has nothing to compare against yet -- allowed,
    # zero measured drift.
    wd = _watchdog()
    r = wd.update(aligning=True, x=1.0, y=2.0)
    assert r.allow_align is True
    assert r.tripped_this_frame is False
    assert r.drift_m == 0.0


def test_no_trip_while_staying_under_the_threshold():
    wd = _watchdog(drift_max_m=0.15)
    r1 = wd.update(aligning=True, x=1.0, y=2.0)  # anchor at (1.0, 2.0)
    assert r1.allow_align is True
    # Small in-place jitter, well under the 0.15 m threshold.
    r2 = wd.update(aligning=True, x=1.05, y=2.02)
    assert r2.allow_align is True
    assert r2.tripped_this_frame is False
    assert r2.tripped is False
    assert math.isclose(r2.drift_m, math.hypot(0.05, 0.02), rel_tol=1e-9)
    r3 = wd.update(aligning=True, x=1.10, y=2.05)
    assert r3.allow_align is True
    assert r3.tripped is False


def test_trip_when_drift_exceeds_threshold_is_permanent():
    wd = _watchdog(drift_max_m=0.15)
    wd.update(aligning=True, x=0.0, y=0.0)  # anchor at origin
    r_trip = wd.update(aligning=True, x=0.20, y=0.0)  # 0.20 m > 0.15 m threshold
    assert r_trip.allow_align is False
    assert r_trip.tripped_this_frame is True
    assert r_trip.tripped is True
    assert math.isclose(r_trip.drift_m, 0.20, rel_tol=1e-9)
    assert wd.tripped is True
    # The NEXT frame (even with no further drift) is not a fresh trip -- tripped_this_frame is
    # one-shot -- but allow_align stays permanently False.
    r_after = wd.update(aligning=True, x=0.20, y=0.0)
    assert r_after.allow_align is False
    assert r_after.tripped_this_frame is False
    assert r_after.tripped is True
    # Permanent even across a not-aligning frame and a brand-new engagement afterward.
    r_idle = wd.update(aligning=False, x=0.0, y=0.0)
    assert r_idle.allow_align is False
    assert r_idle.tripped is True
    r_new_burst = wd.update(aligning=True, x=5.0, y=5.0)
    assert r_new_burst.allow_align is False
    assert r_new_burst.tripped is True
    assert wd.tripped is True


def test_reanchors_on_each_new_engagement_drift_is_per_burst():
    # A dog that walks between two separate rotation bursts must not trip on the WALK -- only
    # on drift measured WITHIN a burst, from that burst's own anchor.
    wd = _watchdog(drift_max_m=0.15)
    r1 = wd.update(aligning=True, x=0.0, y=0.0)  # burst 1 anchor
    assert r1.allow_align is True
    r2 = wd.update(aligning=False, x=0.0, y=0.0)  # burst 1 ends (engagement released)
    assert r2.allow_align is False
    # Ordinary follow walks the dog 2 m away entirely OUTSIDE any alignment burst.
    r_walk = wd.update(aligning=False, x=2.0, y=0.0)
    assert r_walk.allow_align is False
    assert wd.tripped is False
    # Burst 2 starts FAR from burst 1's anchor -- must re-anchor HERE, not compare against the
    # stale (0.0, 0.0) anchor (which would immediately read as a 2 m drift and false-trip).
    r3 = wd.update(aligning=True, x=2.0, y=0.0)  # fresh anchor at (2.0, 0.0)
    assert r3.allow_align is True
    assert r3.tripped_this_frame is False
    assert r3.drift_m == 0.0
    assert wd.tripped is False
    # Small drift within burst 2, measured from the NEW anchor -- still fine.
    r4 = wd.update(aligning=True, x=2.05, y=0.0)
    assert r4.allow_align is True
    assert math.isclose(r4.drift_m, 0.05, rel_tol=1e-9)


def test_no_reanchor_once_tripped():
    wd = _watchdog(drift_max_m=0.15)
    wd.update(aligning=True, x=0.0, y=0.0)
    r_trip = wd.update(aligning=True, x=1.0, y=0.0)  # trips
    assert r_trip.tripped is True
    # Release and a brand-new engagement far away must NOT be treated as a fresh, allowed
    # burst -- the permanent trip dominates regardless of position.
    wd.update(aligning=False, x=0.0, y=0.0)
    r_new = wd.update(aligning=True, x=100.0, y=100.0)
    assert r_new.allow_align is False
    assert r_new.tripped is True
    assert r_new.tripped_this_frame is False


def test_zero_arg_is_a_deliberate_hard_off():
    # CLAUDE.md 8.1 zero-as-disabled-sentinel: tested on the RAW threshold, not a derived
    # value -- the carve-out can never be honored, even with zero measured drift, even on the
    # very first frame of a burst (a naive ">" drift check alone could never implement a hard
    # off, since a stationary robot would never exceed a literal 0.0 m threshold).
    wd = _watchdog(drift_max_m=0.0)
    r1 = wd.update(aligning=True, x=0.0, y=0.0)
    assert r1.allow_align is False
    assert r1.tripped_this_frame is False
    r2 = wd.update(aligning=True, x=0.0, y=0.0)  # zero drift, still denied
    assert r2.allow_align is False
    assert r2.tripped_this_frame is False
    # Negative is treated the same as exactly zero (raw value <= 0.0).
    wd_neg = _watchdog(drift_max_m=-1.0)
    r_neg = wd_neg.update(aligning=True, x=0.0, y=0.0)
    assert r_neg.allow_align is False


def test_displacement_math_uses_euclidean_distance():
    wd = _watchdog(drift_max_m=1.0)
    wd.update(aligning=True, x=0.0, y=0.0)
    # 3-4-5 triangle -> drift should read 5.0, not e.g. |dx|+|dy| (Manhattan) == 7.0.
    r = wd.update(aligning=True, x=3.0, y=4.0)
    assert math.isclose(r.drift_m, 5.0, rel_tol=1e-9)
    assert r.allow_align is False  # 5.0 > 1.0 threshold -- trips
    assert r.tripped_this_frame is True


def test_reset_clears_trip_and_anchor():
    wd = _watchdog(drift_max_m=0.15)
    wd.update(aligning=True, x=0.0, y=0.0)
    wd.update(aligning=True, x=1.0, y=0.0)  # trips
    assert wd.tripped is True
    wd.reset()
    assert wd.tripped is False
    # Behaves like a brand-new instance post-reset: first aligning frame re-anchors cleanly.
    r = wd.update(aligning=True, x=1.0, y=0.0)
    assert r.allow_align is True
    assert r.drift_m == 0.0


if __name__ == "__main__":
    test_not_aligning_never_allows_and_has_no_anchor()
    test_first_aligning_frame_anchors_and_allows_with_zero_drift()
    test_no_trip_while_staying_under_the_threshold()
    test_trip_when_drift_exceeds_threshold_is_permanent()
    test_reanchors_on_each_new_engagement_drift_is_per_burst()
    test_no_reanchor_once_tripped()
    test_zero_arg_is_a_deliberate_hard_off()
    test_displacement_math_uses_euclidean_distance()
    test_reset_clears_trip_and_anchor()
    print("OK")
