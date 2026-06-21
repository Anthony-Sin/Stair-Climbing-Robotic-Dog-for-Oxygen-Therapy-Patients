"""Host tests for the low-level safety watchdog (no ROS 2 / no robot).

Run: python tests/test_real_safety_watchdog.py  (or via pytest)
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from real.control.safety_watchdog import SafetyWatchdog


def test_fresh_upright_is_ok():
    wd = SafetyWatchdog()
    v = wd.evaluate(now=10.0, last_state_ts=9.95, roll=0.05, pitch=-0.05)
    assert v.ok and v.reason == "ok"


def test_stale_state_trips_and_latches():
    wd = SafetyWatchdog(stale_sec=0.25)
    v = wd.evaluate(now=10.0, last_state_ts=9.0, roll=0.0, pitch=0.0)  # 1.0 s stale
    assert not v.ok and v.reason == "lowstate_stale"
    # latched: even a perfectly healthy reading stays faulted until reset
    v2 = wd.evaluate(now=10.02, last_state_ts=10.0, roll=0.0, pitch=0.0)
    assert not v2.ok and v2.reason == "lowstate_stale"
    wd.reset()
    assert wd.evaluate(now=10.04, last_state_ts=10.02, roll=0.0, pitch=0.0).ok


def test_tilt_trips():
    wd = SafetyWatchdog(max_tilt_rad=0.52)
    v = wd.evaluate(now=1.0, last_state_ts=0.99, roll=0.6, pitch=0.0)
    assert not v.ok and v.reason == "tilt_exceeded"


def test_joint_limit_trips_when_bounds_given():
    lo = [-1.0] * 12
    hi = [1.0] * 12
    wd = SafetyWatchdog(joint_lower=lo, joint_upper=hi)
    bad = [0.0] * 11 + [2.0]  # last joint past +1.0
    v = wd.evaluate(now=1.0, last_state_ts=0.99, roll=0.0, pitch=0.0, targets=bad)
    assert not v.ok and v.reason == "joint_limit"


def test_no_bounds_skips_joint_check():
    wd = SafetyWatchdog()  # no limits supplied
    v = wd.evaluate(now=1.0, last_state_ts=0.99, roll=0.0, pitch=0.0, targets=[9.0] * 12)
    assert v.ok


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
