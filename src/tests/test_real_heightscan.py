"""Host tests for the LiDAR heightscan provider (no ROS 2 / no robot).

Pins: flat mode -> zeros; lidar mode builds a body-frame elevation map matching PGTT's
99-cell grid; height_fn interpolation; and the grid<->height_fn round-trip the topic uses.

Run: python tests/test_real_heightscan.py  (or via pytest)
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from real.perception.heightscan_provider import HeightscanProvider, height_fn_from_grid
from go2_locomotion.pgtt_heightmap import PGTT_N_POINTS, PGTT_N_ROWS, PGTT_N_COLS, PGTT_DIST_X, PGTT_DIST_Y


def _step_points():
    """Flat ground z=0, plus a 0.15 m step for the front of the footprint (x > 0.3 m)."""
    xs = np.linspace(-0.5, 0.5, 21)
    ys = np.linspace(-0.4, 0.4, 17)
    pts = []
    for x in xs:
        for y in ys:
            z = 0.15 if x > 0.3 else 0.0
            pts.append((x, y, z))
    return np.asarray(pts, dtype=np.float32)


def test_flat_mode_is_zero():
    p = HeightscanProvider(mode="flat")
    hs = p.heightscan_99()
    assert hs.shape == (PGTT_N_POINTS,) and np.allclose(hs, 0.0)
    assert np.allclose(p.grid_raw(), 0.0)
    assert p.height_fn()(0.3, 0.0) == 0.0


def test_flat_mode_ignores_points():
    p = HeightscanProvider(mode="flat")
    p.update_from_points(_step_points())   # no-op in flat mode
    assert np.allclose(p.heightscan_99(), 0.0)


def test_lidar_mode_builds_front_step():
    p = HeightscanProvider(mode="lidar")
    p.update_from_points(_step_points())
    assert p.grid_raw().shape == (PGTT_N_POINTS,)
    fn = p.height_fn()
    front = fn(0.4, 0.0)   # forward 0.4 m -> on the step
    rear = fn(-0.4, 0.0)   # behind the robot -> ground
    assert front > 0.12 and abs(rear) < 0.03
    assert front - rear > 0.1


def test_height_fn_from_grid_roundtrips_cell_centers():
    # grid[i,j] = i*0.01; the cell center is at forward=(5-i)*dist, lateral=(4-j)*dist.
    grid = np.zeros((PGTT_N_ROWS, PGTT_N_COLS), dtype=np.float32)
    for i in range(PGTT_N_ROWS):
        grid[i, :] = i * 0.01
    fn = height_fn_from_grid(grid.ravel())
    c_h = (PGTT_N_ROWS - 1) / 2.0
    c_w = (PGTT_N_COLS - 1) / 2.0
    for i in (0, 5, 10):
        fwd = (c_h - i) * PGTT_DIST_X
        lat = (c_w - 4) * PGTT_DIST_Y  # j=4 (center col)
        assert abs(fn(fwd, lat) - grid[i, 4]) < 1e-4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK")
