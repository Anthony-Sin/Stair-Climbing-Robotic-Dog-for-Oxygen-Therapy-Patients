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
# Repo root enables `core.*` imports; sim/isaac enables the omni-free perception
# modules. core/ itself is intentionally NOT on sys.path (core uses fully-qualified
# `core.vision` imports), so bare `perception` always resolves to sim/isaac's.
for _p in (os.path.join(_REPO, "sim", "isaac"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.vision import lidar_fusion as lf
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


def _world_floor_and_person(sensor_z=0.4, person_range=2.0, person_half_deg=6.0):
    """Injected raycast modelling the P0-2 failure mode.

    The downward XT16 channels ring the FLOOR: a ray with a negative vertical
    component (direction[2] < 0) hits the ground at range sensor_z/|sin(elev)|
    (~1.545 m for the -15 deg channel at a 0.4 m mount). A near-horizontal ray dead
    ahead (world +x) instead hits the PERSON at person_range (~2.0 m). Everything
    else misses (open scene). This is the exact geometry where the pre-filter
    nearest-per-azimuth collapse reported the 1.545 m floor as the "person range".
    """
    def raycast(origin, direction, max_dist):
        dx, dy, dz = direction[0], direction[1], direction[2]
        # Downward ray -> floor hit (the ring that masks the person).
        if dz < -1e-6:
            r = float(sensor_z) / abs(dz)  # plane z=0 at height sensor_z below origin
            return r if r <= max_dist else None
        # Near-horizontal ray dead ahead -> the person at person_range.
        horiz = math.hypot(dx, dy)
        if horiz > 1e-6:
            fwd_cos = dx / horiz
            if fwd_cos > math.cos(math.radians(person_half_deg)) and abs(dy) < 0.2:
                return person_range if person_range <= max_dist else None
        return None
    return raycast


def test_ground_filter_recovers_person_behind_floor_ring():
    cfg = Xt16Config(azimuth_step_deg=3.0, max_range_m=50.0, mount_z_m=0.0)
    # Sensor origin z=0.4 so the -15 deg channel rings the floor at ~1.545 m; person
    # dead ahead at 2.0 m. The downward floor ring is NEARER than the person.
    scan = cast_scan(cfg, (0.0, 0.0, 0.4), 0.0, _world_floor_and_person(0.4, 2.0))

    # WITHOUT the ground filter, the nearest-per-azimuth collapse reports the floor.
    prof_raw = profile_from_scan(scan, view_range_m=6.0, ground_clip_below_sensor_m=0.0)
    decoded_raw = lf.decode_lidar_profile(prof_raw)
    rng_raw = lf.lidar_range_at_bearing(decoded_raw, 0.0, window_deg=4.0)
    assert rng_raw is not None and abs(rng_raw - 1.545) < 0.1, (
        f"expected the floor ring (~1.545 m) without the filter, got {rng_raw}"
    )

    # WITH the ground filter (default enabled) the floor cells are dropped and the
    # person's true 2.0 m range survives at bearing 0.
    prof = profile_from_scan(scan, view_range_m=6.0)  # default clip from config
    decoded = lf.decode_lidar_profile(prof)
    rng = lf.lidar_range_at_bearing(decoded, 0.0, window_deg=4.0)
    assert rng is not None and abs(rng - 2.0) < 0.05, (
        f"expected the person (~2.0 m) after the ground filter, got {rng}"
    )


def _world_person_at(az_deg, rng=1.5, wall=5.0):
    """Injected raycast: a single near return at azimuth az_deg (CCW/+left), walls elsewhere."""
    target = math.radians(az_deg)

    def raycast(origin, direction, max_dist):
        a = math.atan2(direction[1], direction[0])
        da = abs((a - target + math.pi) % (2.0 * math.pi) - math.pi)
        if da < math.radians(4.0):
            return rng
        return wall if wall <= max_dist else None
    return raycast


def test_person_bearing_from_profile_recovers_offaxis_side():
    cfg = Xt16Config(azimuth_step_deg=3.0, max_range_m=50.0)
    # Patient ~20 deg to the LEFT (+ in CCW convention) at 1.5 m; walls beyond.
    scan = cast_scan(cfg, (0.0, 0.0, 0.3), 0.0, _world_person_at(20.0, 1.5, 5.0))
    decoded = lf.decode_lidar_profile(profile_from_scan(scan, view_range_m=6.0))

    # Prior near centre, prior range ~1.5 -> recover the +20 deg (left) foreground return.
    b = lf.person_bearing_from_profile(decoded, prior_bearing_rad=0.0, prior_range_m=1.5)
    assert b is not None and abs(math.degrees(b) - 20.0) <= 4.0

    # Patient to the RIGHT (negative bearing in the +left convention).
    scan_r = cast_scan(cfg, (0.0, 0.0, 0.3), 0.0, _world_person_at(-20.0, 1.5, 5.0))
    decoded_r = lf.decode_lidar_profile(profile_from_scan(scan_r, view_range_m=6.0))
    br = lf.person_bearing_from_profile(decoded_r, prior_bearing_rad=0.0, prior_range_m=1.5)
    assert br is not None and math.degrees(br) < 0.0


def test_person_bearing_from_profile_rejects_far_and_out_of_window():
    cfg = Xt16Config(azimuth_step_deg=3.0, max_range_m=50.0)
    scan = cast_scan(cfg, (0.0, 0.0, 0.3), 0.0, _world_person_at(20.0, 1.5, 5.0))
    decoded = lf.decode_lidar_profile(profile_from_scan(scan, view_range_m=6.0))

    # The patient (20 deg) is outside a window centred on 120 deg -> only far walls there -> None.
    assert lf.person_bearing_from_profile(
        decoded, prior_bearing_rad=math.radians(120.0), prior_range_m=1.0, window_deg=20.0
    ) is None
    # A tight range gate rejects the 5 m walls when no near return sits near the prior bearing.
    scan_walls = cast_scan(cfg, (0.0, 0.0, 0.3), 0.0, _world_person_at(20.0, 5.0, 5.0))
    decoded_walls = lf.decode_lidar_profile(profile_from_scan(scan_walls, view_range_m=6.0))
    assert lf.person_bearing_from_profile(
        decoded_walls, prior_bearing_rad=0.0, prior_range_m=1.0
    ) is None


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

    # Absurd-far depth vs a confident near LiDAR -> REJECT the depth, use LiDAR.
    # (Evidence: a frame accepted fused=36.87 m while the LiDAR said 0.59 m.)
    r = lf.fuse_distance(36.87, 0.59)
    assert r["source"] == "lidar_reject_far_depth" and r["fused_m"] == 0.59
    assert r["disagreement"] and r["confidence"] >= 0.5

    # A FAR LiDAR return is not "confident near", so an equally far depth is NOT
    # force-rejected (falls through to the ordinary disagreement fallback / blend).
    far = lf.fuse_distance(12.0, 5.0)
    assert far["source"] != "lidar_reject_far_depth"

    # The guard is opt-outable and does not fire when depth is only modestly farther.
    off = lf.fuse_distance(36.87, 0.59, reject_far_depth=False)
    assert off["source"] != "lidar_reject_far_depth"

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
