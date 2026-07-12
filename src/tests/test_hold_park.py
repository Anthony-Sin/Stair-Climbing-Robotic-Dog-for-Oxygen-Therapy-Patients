"""Host-safe (no Isaac deps) tests for go2_locomotion/hold_park.py -- the D1 sustained-hold
PARK state machine (run-12 review, 2026-07-12, CLAUDE.md incident 8.15/8.16 continuation).

The isaac_env.py WIRING (drive-gain swap, joint-target slew, skip/resume rl_policy.step()) is
verified by code reading + `python -m compileall` (mirrors the F1 clamp precedent documented
in test_stair_speed_guards.py's module docstring -- isaac_env.py imports Isaac/omniverse and
cannot be imported on a plain host, incident 8.4). This file covers the PURE decision state
machine exhaustively instead.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # go2_locomotion package lives at the repo root

from go2_locomotion.hold_park import (  # noqa: E402
    HoldParkConfig,
    HoldParkController,
)


def _cfg(park_after_sec=2.5, tilt_max_rad=0.14, slew_sec=0.7):
    return HoldParkConfig(
        park_after_sec=park_after_sec, tilt_max_rad=tilt_max_rad, slew_sec=slew_sec,
    )


# --------------------------------------------------------------------------------------
# Below the sustained-hold threshold: normal walking, no engagement.
# --------------------------------------------------------------------------------------

def test_below_threshold_stays_walking():
    ctl = HoldParkController(_cfg(park_after_sec=2.5))
    for _ in range(10):
        d = ctl.update(0.1, hold_requested=True, tilt_rad=0.0)  # 1.0s total < 2.5s
        assert d.state == "walk"
        assert d.run_policy is True
        assert d.engaged_this_frame is False
        assert d.released_this_frame is False
        assert d.slew_alpha is None


def test_hold_elapsed_telemetry_accumulates():
    ctl = HoldParkController(_cfg(park_after_sec=2.5))
    d1 = ctl.update(0.5, hold_requested=True, tilt_rad=0.0)
    d2 = ctl.update(0.5, hold_requested=True, tilt_rad=0.0)
    assert abs(d1.hold_elapsed_sec - 0.5) < 1e-9
    assert abs(d2.hold_elapsed_sec - 1.0) < 1e-9


def test_any_non_hold_frame_resets_the_accumulator():
    """ANY non-hold frame resets the accumulator (task brief). A partial hold below
    threshold, interrupted, must not carry over toward the next hold stretch."""
    ctl = HoldParkController(_cfg(park_after_sec=2.5))
    for _ in range(20):
        ctl.update(0.1, hold_requested=True, tilt_rad=0.0)  # 2.0s accrued, still < 2.5s
    d = ctl.update(0.1, hold_requested=False, tilt_rad=0.0)
    assert d.state == "walk"
    assert d.hold_elapsed_sec == 0.0
    # Resume holding: must need the FULL park_after_sec again, not just the remaining 0.5s.
    d2 = ctl.update(2.0, hold_requested=True, tilt_rad=0.0)  # 2.0s -- still short of 2.5s
    assert d2.state == "walk"
    assert d2.engaged_this_frame is False


# --------------------------------------------------------------------------------------
# Engage transition (walk -> slewing -> parked).
# --------------------------------------------------------------------------------------

def test_engages_once_threshold_crossed_on_level_ground():
    ctl = HoldParkController(_cfg(park_after_sec=2.5, slew_sec=0.7))
    for _ in range(24):
        d = ctl.update(0.1, hold_requested=True, tilt_rad=0.0)  # up to 2.4s
        assert d.state == "walk"
    d = ctl.update(0.1, hold_requested=True, tilt_rad=0.0)  # crosses 2.5s
    assert d.state in ("slewing", "parked")
    assert d.engaged_this_frame is True
    assert d.run_policy is False
    assert d.slew_alpha is not None
    assert 0.0 < d.slew_alpha <= 1.0


def test_slew_progresses_then_reaches_parked():
    ctl = HoldParkController(_cfg(park_after_sec=1.0, slew_sec=0.7))
    for _ in range(9):
        d = ctl.update(0.1, hold_requested=True, tilt_rad=0.0)  # 0.9s total, still walking
        assert d.state == "walk"
    d0 = ctl.update(0.2, hold_requested=True, tilt_rad=0.0)  # crosses 1.0s -> engage
    assert d0.state == "slewing"
    assert d0.engaged_this_frame is True
    alpha0 = d0.slew_alpha
    assert 0.0 < alpha0 < 1.0
    d1 = ctl.update(0.2, hold_requested=True, tilt_rad=0.0)
    assert d1.state == "slewing"
    assert d1.engaged_this_frame is False
    assert d1.slew_alpha > alpha0  # monotonically progressing
    # Finish the slew (well past slew_sec).
    d2 = ctl.update(1.0, hold_requested=True, tilt_rad=0.0)
    assert d2.state == "parked"
    assert d2.slew_alpha == 1.0
    assert d2.run_policy is False


def test_stays_parked_with_no_repeated_engage():
    ctl = HoldParkController(_cfg(park_after_sec=0.5, slew_sec=0.2))
    ctl.update(0.5, hold_requested=True, tilt_rad=0.0)   # engage
    ctl.update(0.2, hold_requested=True, tilt_rad=0.0)   # slew completes
    for _ in range(5):
        d = ctl.update(0.3, hold_requested=True, tilt_rad=0.0)
        assert d.state == "parked"
        assert d.engaged_this_frame is False
        assert d.released_this_frame is False
        assert d.run_policy is False
        assert d.slew_alpha == 1.0


# --------------------------------------------------------------------------------------
# Tilt gate -- entry gate ONLY, checked at the moment the threshold is first satisfied.
# --------------------------------------------------------------------------------------

def test_tilt_above_max_blocks_engagement_at_threshold():
    """A straddle-class tilt (~0.148 rad, incident 8.16 run 6's -8.5 deg) must never engage,
    even once the hold has been sustained long enough."""
    ctl = HoldParkController(_cfg(park_after_sec=1.0, tilt_max_rad=0.14))
    for _ in range(30):
        d = ctl.update(0.1, hold_requested=True, tilt_rad=0.148)  # 3.0s, well past threshold
        assert d.state == "walk", "must never park while tilted like a crest straddle"
        assert d.engaged_this_frame is False


def test_engages_once_tilt_drops_back_under_max_without_losing_accrued_hold():
    """The hold accumulator is NOT reset by a bad-tilt frame (only a non-hold frame resets
    it) -- once tilt recovers to level, the ALREADY-sustained hold engages immediately."""
    ctl = HoldParkController(_cfg(park_after_sec=1.0, tilt_max_rad=0.14, slew_sec=0.5))
    for _ in range(15):
        d = ctl.update(0.1, hold_requested=True, tilt_rad=0.20)  # 1.5s, tilted, never engages
        assert d.state == "walk"
    d = ctl.update(0.01, hold_requested=True, tilt_rad=0.05)  # tiny extra dt, now level
    assert d.state in ("slewing", "parked")
    assert d.engaged_this_frame is True


def test_tilt_boundary_is_strict_less_than():
    """tilt_rad == tilt_max_rad must NOT engage (strict '<' per the implementation)."""
    ctl = HoldParkController(_cfg(park_after_sec=0.1, tilt_max_rad=0.14))
    d = ctl.update(0.2, hold_requested=True, tilt_rad=0.14)
    assert d.state == "walk"
    assert d.engaged_this_frame is False


# --------------------------------------------------------------------------------------
# Release -- instant (no dwell), from any state, the moment hold_requested goes False.
# --------------------------------------------------------------------------------------

def test_release_is_instant_from_parked():
    ctl = HoldParkController(_cfg(park_after_sec=0.2, slew_sec=0.2))
    ctl.update(0.2, hold_requested=True, tilt_rad=0.0)   # engage
    ctl.update(0.5, hold_requested=True, tilt_rad=0.0)   # parked
    d = ctl.update(0.1, hold_requested=False, tilt_rad=0.0)
    assert d.state == "walk"
    assert d.run_policy is True
    assert d.released_this_frame is True
    assert d.hold_elapsed_sec == 0.0


def test_release_is_instant_from_slewing():
    ctl = HoldParkController(_cfg(park_after_sec=0.2, slew_sec=2.0))
    ctl.update(0.2, hold_requested=True, tilt_rad=0.0)   # engage, still slewing (slew_sec=2.0)
    d_mid = ctl.update(0.1, hold_requested=True, tilt_rad=0.0)
    assert d_mid.state == "slewing"
    d = ctl.update(0.1, hold_requested=False, tilt_rad=0.0)
    assert d.state == "walk"
    assert d.released_this_frame is True
    assert d.run_policy is True


def test_release_from_walk_state_is_a_no_op_flagged_false():
    """Releasing while already in 'walk' (never engaged) must not report a spurious
    released_this_frame."""
    ctl = HoldParkController(_cfg(park_after_sec=2.5))
    ctl.update(0.5, hold_requested=True, tilt_rad=0.0)
    d = ctl.update(0.1, hold_requested=False, tilt_rad=0.0)
    assert d.state == "walk"
    assert d.released_this_frame is False


def test_reset_clears_all_state():
    ctl = HoldParkController(_cfg(park_after_sec=0.2, slew_sec=0.2))
    ctl.update(0.2, hold_requested=True, tilt_rad=0.0)
    ctl.update(0.5, hold_requested=True, tilt_rad=0.0)
    assert ctl.state == "parked"
    ctl.reset()
    assert ctl.state == "walk"
    d = ctl.update(0.01, hold_requested=True, tilt_rad=0.0)
    assert d.state == "walk"
    assert abs(d.hold_elapsed_sec - 0.01) < 1e-9


# --------------------------------------------------------------------------------------
# 8.6: dt-accumulation contract -- a smaller dt (slower "FPS") takes proportionally more
# CALLS but the same SIMULATED seconds to engage; the threshold is a duration, not a count.
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Task (2026-07-12, runs 31/32 review): caller-requested IMMEDIATE engage (the stair-base
# approach-squeeze fix, CLAUDE.md 8.15 continuation). ``park_requested`` lets a caller with
# its own FSM context (core/control/stair_policy.base_approach_park_request) bypass
# park_after_sec, but the immediate path must be a STRICT SUBSET of the timed path except
# for the timer -- same tilt gate, no effect once already engaged, no effect on release.
# --------------------------------------------------------------------------------------

def test_park_requested_engages_immediately_well_below_timer_threshold():
    """A single frame with park_requested=True engages on the SPOT (no accrued hold needed
    at all), well before park_after_sec would ever be satisfied."""
    ctl = HoldParkController(_cfg(park_after_sec=2.5, slew_sec=0.7))
    d = ctl.update(0.05, hold_requested=True, tilt_rad=0.0, park_requested=True)
    assert d.state in ("slewing", "parked")
    assert d.engaged_this_frame is True
    assert d.engaged_immediate is True
    assert d.run_policy is False


def test_park_requested_false_does_not_engage_before_timer():
    """Default (park_requested omitted / False) behaves exactly as before -- no immediate
    engage, ordinary timer-gated behavior."""
    ctl = HoldParkController(_cfg(park_after_sec=2.5, slew_sec=0.7))
    d = ctl.update(0.05, hold_requested=True, tilt_rad=0.0)
    assert d.state == "walk"
    assert d.engaged_this_frame is False
    assert d.engaged_immediate is False


def test_park_requested_still_respects_tilt_gate():
    """The immediate path is a STRICT SUBSET of the timed path except for the timer -- the
    SAME tilt_max_rad entry gate still applies. A straddle-class tilt must never engage even
    when explicitly requested (mirrors test_tilt_above_max_blocks_engagement_at_threshold)."""
    ctl = HoldParkController(_cfg(park_after_sec=2.5, tilt_max_rad=0.14))
    for _ in range(10):
        d = ctl.update(0.1, hold_requested=True, tilt_rad=0.148, park_requested=True)
        assert d.state == "walk", "must never park while tilted like a crest straddle"
        assert d.engaged_this_frame is False
        assert d.engaged_immediate is False


def test_park_requested_has_no_effect_once_hold_requested_is_false():
    """park_requested=True with hold_requested=False must NOT engage -- hold_requested is
    checked first (the instant-release branch), matching the task's "AND _motion_hold_
    requested" requirement: the immediate path can never outrun the caller's own hold
    decision."""
    ctl = HoldParkController(_cfg(park_after_sec=2.5))
    d = ctl.update(0.1, hold_requested=False, tilt_rad=0.0, park_requested=True)
    assert d.state == "walk"
    assert d.engaged_this_frame is False
    assert d.engaged_immediate is False
    assert d.run_policy is True


def test_park_requested_has_no_further_effect_once_already_parked():
    """Once already engaged (whether via timer or request), further park_requested=True
    frames are ordinary "already parked" frames -- no re-engage, no engaged_immediate."""
    ctl = HoldParkController(_cfg(park_after_sec=2.5, slew_sec=0.1))
    d0 = ctl.update(0.05, hold_requested=True, tilt_rad=0.0, park_requested=True)
    assert d0.engaged_this_frame is True
    d1 = ctl.update(0.2, hold_requested=True, tilt_rad=0.0, park_requested=True)
    assert d1.state == "parked"
    assert d1.engaged_this_frame is False
    assert d1.engaged_immediate is False


def test_timer_and_request_coincident_reports_as_ordinary_timed_engage():
    """If the timer independently crosses park_after_sec the SAME frame a request is also
    asserted, the engage is reported as an ORDINARY timed engage (engaged_immediate False) --
    the request did not actually bypass anything that frame (see HoldParkDecision.
    engaged_immediate's docstring for the precise "did the request cause it" semantics).
    Uses dt=0.5 (exactly binary-representable) so the accumulator lands on exactly 1.0
    rather than a float-summation artifact just under it."""
    ctl = HoldParkController(_cfg(park_after_sec=1.0, slew_sec=0.7))
    d0 = ctl.update(0.5, hold_requested=True, tilt_rad=0.0)  # 0.5s, still walking
    assert d0.state == "walk"
    d = ctl.update(0.5, hold_requested=True, tilt_rad=0.0, park_requested=True)  # crosses 1.0s exactly
    assert d.engaged_this_frame is True
    assert d.engaged_immediate is False


def test_park_requested_release_semantics_unchanged():
    """Release stays instant and governed solely by hold_requested going False -- a request
    flag has no bearing on release (mirrors test_release_is_instant_from_parked)."""
    ctl = HoldParkController(_cfg(park_after_sec=2.5, slew_sec=0.2))
    ctl.update(0.05, hold_requested=True, tilt_rad=0.0, park_requested=True)  # immediate engage
    ctl.update(0.5, hold_requested=True, tilt_rad=0.0)  # parked
    d = ctl.update(0.1, hold_requested=False, tilt_rad=0.0)
    assert d.state == "walk"
    assert d.run_policy is True
    assert d.released_this_frame is True
    assert d.hold_elapsed_sec == 0.0


def test_threshold_is_a_duration_not_a_frame_count():
    """Different dt/step-count combinations that sum to the same simulated seconds must
    engage at (approximately) the same simulated time -- proving this gates on accumulated
    dt, not calls-to-update (incident 8.6)."""
    cfg = _cfg(park_after_sec=1.0, slew_sec=10.0)  # long slew so state stays "slewing"

    ctl_fast = HoldParkController(cfg)  # e.g. 40 Hz: dt=0.025
    engaged_at_fast = None
    t = 0.0
    for _ in range(80):
        t += 0.025
        d = ctl_fast.update(0.025, hold_requested=True, tilt_rad=0.0)
        if d.engaged_this_frame:
            engaged_at_fast = t
            break

    ctl_slow = HoldParkController(cfg)  # e.g. 4 Hz: dt=0.25
    engaged_at_slow = None
    t = 0.0
    for _ in range(8):
        t += 0.25
        d = ctl_slow.update(0.25, hold_requested=True, tilt_rad=0.0)
        if d.engaged_this_frame:
            engaged_at_slow = t
            break

    assert engaged_at_fast is not None and engaged_at_slow is not None
    assert abs(engaged_at_fast - engaged_at_slow) <= 0.25, (
        f"engage time should track simulated seconds regardless of step rate: "
        f"fast={engaged_at_fast} slow={engaged_at_slow}"
    )
