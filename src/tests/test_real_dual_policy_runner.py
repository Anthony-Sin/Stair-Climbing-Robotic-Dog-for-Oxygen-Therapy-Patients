"""Host tests for the dual-policy runner FSM, using stub policies (no weights/ROS).

Validates the orchestration ported from isaac_env: WALK vs CLIMB selection, the gain
swap (kp40 walk / kp20 climb), the stair-commit forward floor + heading override, the
blind_rl forward floor + reset-on-entry, and the bearing heading-hold.

Run: python tests/test_real_dual_policy_runner.py  (or via pytest)
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from real.control.dual_policy_runner import DualPolicyRunner, RobotState, RunnerOutput
from real.control.follow_command import FollowCommand

_PGTT_TARGETS = np.arange(12, dtype=np.float32)
_BLIND_TARGETS = np.arange(100, 112, dtype=np.float32)


class _StubPgtt:
    def __init__(self):
        self.last_targets_isaac = _PGTT_TARGETS.copy()
        self.last_cmd = None
        self.reset_calls = 0
        self.ext_targets = None

    def reset(self):
        self.reset_calls += 1

    def step(self, art, cmd, dt, hold=False, height_fn=None, **_):
        self.last_cmd = tuple(cmd)
        self.last_hold = hold

    def apply_external_act_targets(self, art, targets_act):
        self.ext_targets = targets_act


class _StubBlind:
    def __init__(self):
        self.last_targets_isaac = _BLIND_TARGETS.copy()
        self.last_cmd = None
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def step(self, art, cmd, dt):
        self.last_cmd = tuple(cmd)


class _StubHandoff:
    def __init__(self, result):
        self.result = result
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def update(self, **kwargs):
        self.seen = kwargs
        return self.result


def _runner(handoff_result):
    return DualPolicyRunner(_StubPgtt(), _StubBlind(), _StubHandoff(handoff_result),
                            climb_backend="blind_rl")


def test_walk_runs_pgtt_with_walk_gains():
    r = _runner({"climb": False, "telemetry": {}})
    out = r.step(object(), FollowCommand(vx=0.5, wz=0.1), depth_106x60=None,
                 height_fn=lambda x, y: 0.0, state=RobotState(), dt=0.02, now=1.0)
    assert out.backend == "walk" and out.kp == 40.0 and out.kd == 0.5
    assert np.allclose(out.targets_isaac, _PGTT_TARGETS)
    assert r.pgtt.last_cmd == (0.5, 0.0, 0.1)


def test_stair_commit_overrides_apply_to_walk():
    r = _runner({"climb": False, "vx_floor": 0.22, "wz_override": 0.3, "telemetry": {}})
    out = r.step(object(), FollowCommand(vx=0.0, wz=0.0), depth_106x60=None,
                 height_fn=lambda x, y: 0.0, state=RobotState(), dt=0.02, now=1.0)
    assert out.backend == "walk"
    assert r.pgtt.last_cmd == (0.22, 0.0, 0.3)   # floored vx + heading override


def test_climb_runs_blind_rl_with_climb_gains_and_floor():
    r = _runner({"climb": True, "use_parkour": True, "telemetry": {}})
    out = r.step(object(), FollowCommand(vx=0.05, wz=0.0, person_detected=False),
                 depth_106x60=None, height_fn=lambda x, y: 0.0,
                 state=RobotState(), dt=0.02, now=1.0)
    assert out.backend == "climb_blind_rl" and out.kp == 20.0 and out.kd == 0.5
    assert np.allclose(out.targets_isaac, _BLIND_TARGETS)
    assert r.blind_rl.reset_calls == 1           # reset on climb entry
    assert r.blind_rl.last_cmd[0] == 0.22         # forward floor applied (vx 0.05 -> 0.22)


def test_climb_heading_hold_uses_bearing_then_holds():
    r = _runner({"climb": True, "use_parkour": True, "telemetry": {}})
    # person visible: wz = yaw_err * bearing_scale(0.9), clipped to rot_max(0.6)
    r.step(object(), FollowCommand(vx=0.3, yaw_err=0.4, person_detected=True),
           depth_106x60=None, height_fn=lambda x, y: 0.0, state=RobotState(), dt=0.02, now=1.0)
    assert abs(r.blind_rl.last_cmd[2] - 0.36) < 1e-5
    # person lost: hold the last bearing, decayed by 0.92
    r.step(object(), FollowCommand(vx=0.3, yaw_err=0.0, person_detected=False),
           depth_106x60=None, height_fn=lambda x, y: 0.0, state=RobotState(), dt=0.02, now=1.02)
    assert abs(r.blind_rl.last_cmd[2] - 0.36) < 1e-5   # held (decay applied to the STORED value)


def test_exit_climb_returns_to_walk_gains():
    # First a climb tick, then handoff says walk -> runner must report walk gains.
    h = _StubHandoff({"climb": True, "use_parkour": True, "telemetry": {}})
    r = DualPolicyRunner(_StubPgtt(), _StubBlind(), h, climb_backend="blind_rl")
    r.step(object(), FollowCommand(vx=0.3, person_detected=True), depth_106x60=None,
           height_fn=lambda x, y: 0.0, state=RobotState(), dt=0.02, now=1.0)
    h.result = {"climb": False, "telemetry": {}}
    out = r.step(object(), FollowCommand(vx=0.3), depth_106x60=None,
                 height_fn=lambda x, y: 0.0, state=RobotState(), dt=0.02, now=1.02)
    assert out.backend == "walk" and out.kp == 40.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
