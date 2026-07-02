"""Host-side tests for the follow-command staleness classifier (no ROS 2 needed).

Pins the graduated fail-safe the 50 Hz control node relies on so a dead vision process
can never leave the dog executing a stale command: fresh -> run, stale -> HOLD (balance,
keep mode), gone -> DAMP; and "never received" is HOLD (stand), never DAMP (collapse).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from real.control.command_gate import classify_command_age, FRESH, HOLD, DAMP

_T, _D = 0.4, 1.0   # timeout_sec, damp_sec


def _cls(age, ever=True):
    return classify_command_age(age_sec=age, ever_received=ever, timeout_sec=_T, damp_sec=_D)


def test_fresh_command_runs():
    assert _cls(0.0) == FRESH
    assert _cls(0.39) == FRESH


def test_stale_past_timeout_holds():
    assert _cls(0.41) == HOLD
    assert _cls(0.99) == HOLD


def test_gone_past_damp_damps():
    assert _cls(1.01) == DAMP
    assert _cls(5.0) == DAMP
    assert _cls(float("inf")) == DAMP


def test_never_received_holds_not_damps():
    # A robot that never heard from vision must STAND (HOLD), not collapse (DAMP),
    # regardless of the (infinite) age -- this is the correct bring-up state.
    assert _cls(float("inf"), ever=False) == HOLD
    assert _cls(0.0, ever=False) == HOLD


def test_boundaries_are_strict_greater_than():
    # Exactly at a threshold is NOT yet the more-severe state (age must EXCEED it).
    assert _cls(_T) == FRESH
    assert _cls(_D) == HOLD


if __name__ == "__main__":
    test_fresh_command_runs()
    test_stale_past_timeout_holds()
    test_gone_past_damp_damps()
    test_never_received_holds_not_damps()
    test_boundaries_are_strict_greater_than()
    print("OK")
