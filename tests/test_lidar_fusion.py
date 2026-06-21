"""Pure-Python tests for the XT16 LiDAR profile wire format and the LiDAR+YOLO
distance fusion.

No Isaac/OpenCV/torch needed: sim_lidar_xt16 is omni-free and the raycast is
injected. Run directly (python tests/test_lidar_fusion.py) or via pytest.
(The locomotion policy's per-leg command summary is tested in
tests/test_parkour_contract.py, where the torch-backed policy lives.)
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
from perception.sim_lidar_xt16 import Xt16Config, cast_scan, profile_from_scan


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


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for _fn in _tests:
        _fn()
        print("PASS", _fn.__name__)
    print(f"ALL {len(_tests)} TESTS PASSED")
