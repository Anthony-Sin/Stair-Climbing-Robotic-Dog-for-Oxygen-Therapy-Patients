"""Pure-Python tests for the XT16 LiDAR profile wire format, the LiDAR+YOLO
distance fusion, and the RL policy's real per-leg command summary.

No Isaac/OpenCV needed: sim_lidar_xt16 and rl_locomotion_policy are omni-free and
the raycast is injected. Run directly (python tests/test_lidar_fusion.py) or via
pytest.
"""
import math
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("core", os.path.join("sim", "isaac")):
    _p = os.path.join(_REPO, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lidar_fusion as lf
from sim_lidar_xt16 import Xt16Config, cast_scan, profile_from_scan
import rl_locomotion_policy as rlp


def _fake_world(person_range=2.0, wall_range=5.0):
    """Injected raycast: a person dead ahead (world +x) at person_range, walls
    everywhere else."""
    def raycast(origin, direction, max_dist):
        dx, dy = direction[0], direction[1]
        if dx > math.cos(math.radians(6.0)) and abs(dy) < 0.15:
            return person_range
        return wall_range if wall_range <= max_dist else None
    return raycast


def test_profile_roundtrip_and_bearing():
    cfg = Xt16Config(azimuth_step_deg=3.0, max_range_m=50.0)
    scan = cast_scan(cfg, (0.0, 0.0, 0.3), 0.0, _fake_world(2.0, 5.0))
    profile = profile_from_scan(scan, view_range_m=6.0)

    decoded = lf.decode_lidar_profile(profile)
    assert decoded is not None
    assert decoded["ranges_m"].shape[0] == cfg.n_azimuth

    # Person centered in a 1280px frame (cx=640) -> bearing 0 -> nearest ~2 m.
    bearing = lf.person_bearing_rad(640, 640, 900)
    assert abs(bearing) < 1e-6
    rng = lf.lidar_range_at_bearing(decoded, bearing, window_deg=4.0)
    assert rng is not None and abs(rng - 2.0) < 0.05

    # Image-right target -> robot-right -> negative bearing; image-left -> positive.
    assert lf.person_bearing_rad(900, 640, 900) < 0
    assert lf.person_bearing_rad(300, 640, 900) > 0


def test_decode_handles_empty():
    assert lf.decode_lidar_profile(None) is None
    assert lf.decode_lidar_profile({}) is None
    assert lf.decode_lidar_profile({"ranges_mm": ""}) is None


def test_fuse_agreement_weighted():
    # Agree -> blend, no flag, confidence >= 0.5.
    a = lf.fuse_distance(2.1, 2.0, agree_tol_m=0.25, lidar_weight=0.6)
    assert a["source"] == "fused" and not a["disagreement"]
    assert 2.0 <= a["fused_m"] <= 2.1 and a["confidence"] >= 0.5

    # Disagree -> keep depth, flag, low confidence.
    z = lf.fuse_distance(2.1, 5.0, agree_tol_m=0.25)
    assert z["disagreement"] and z["source"] == "depth_disagree" and z["fused_m"] == 2.1

    # Single-source and empty.
    assert lf.fuse_distance(None, 2.0)["fused_m"] == 2.0
    assert lf.fuse_distance(2.0, None)["fused_m"] == 2.0
    assert lf.fuse_distance(None, None)["fused_m"] is None


def _fake_policy():
    pol = object.__new__(rlp.RLLocomotionPolicy)  # bypass __init__ (no model needed)
    pol.default_pos_policy = np.array(
        [rlp.POLICY_DEFAULT_BY_JOINT[j] for (_l, j) in rlp.POLICY_JOINT_ORDER], np.float32
    )
    pol.action_scale_policy = np.array(
        [rlp.POLICY_ACTION_SCALE_BY_JOINT[j] for (_l, j) in rlp.POLICY_JOINT_ORDER], np.float32
    )
    pol._last_action = np.zeros(12, np.float32)
    pol._last_target_policy = pol.default_pos_policy.copy()
    return pol


def test_leg_summary_default_is_all_stance():
    s = _fake_policy().leg_command_summary()
    assert s["swing_legs"] == []
    assert set(s["leg_commands"]) == {"FL", "FR", "RL", "RR"}
    assert all(c["state"] == "stance" for c in s["leg_commands"].values())


def test_leg_summary_knee_flexion_is_swing():
    pol = _fake_policy()
    tp = pol.default_pos_policy.copy()
    fr_calf = [i for i, (l, j) in enumerate(rlp.POLICY_JOINT_ORDER) if l == "fr" and j == "calf"][0]
    tp[fr_calf] = -2.1  # bend the knee past the -1.5 default -> leg retracts -> swing
    pol._last_target_policy = tp
    s = pol.leg_command_summary()
    assert s["swing_legs"] == ["FR"]
    assert s["leg_commands"]["FR"]["state"] == "swing"
    assert s["leg_commands"]["FR"]["foot_lift_m"] > 0.02
    assert s["leg_commands"]["FL"]["state"] == "stance"


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for _fn in _tests:
        _fn()
        print("PASS", _fn.__name__)
    print(f"ALL {len(_tests)} TESTS PASSED")
