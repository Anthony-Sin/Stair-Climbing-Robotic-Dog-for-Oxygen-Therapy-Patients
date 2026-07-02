"""Synthesize PGTT's 99-value heightscan from a real LiDAR point cloud.

PGTT was trained on a ground-truth terrain oracle (``get_terrain_height``); the real
robot has no such oracle, so this builds a BODY-FRAME 2.5D elevation map from the
LiDAR points and exposes it the two ways PGTT needs:
  * ``height_fn(x, y)`` -- absolute terrain Z at a body-frame point, which PGTT's
    ``build_heightscan`` samples over its 11x9 grid (the policy keeps its own
    subtract-min normalization, so feeding a height_fn is the faithful path).
  * ``heightscan_99()`` -- the same grid pre-raveled, for the /go2/heightscan topic
    and the diagnostic.

The grid matches ``go2_locomotion.pgtt_heightmap`` exactly: 11 rows x 9 cols, 0.1 m
spacing, row i -> forward (5-i)*0.1 m, col j -> lateral (4-j)*0.1 m.

MODES (config ``heightscan_mode``):
  * "flat"  (DEFAULT, bring-up): height_fn == 0 -> heightscan all zeros -> PGTT walks
    a competent blind-flat trot. Climbing is the proprioceptive blind_rl path, which
    needs no heightscan. Safe even with a dead/garbage LiDAR.
  * "lidar" (opt-in, HIL-validated): the synthesized map. This is the single biggest
    sim->real distribution shift; gate it behind the ``pgtt_heightscan`` diagnostic
    (hs_max must rise approaching a known riser) before trusting it for control.

HIL NOTE (lidar mode): PGTT samples height_fn at grid points rotated by the base
yaw (the adapter reports the true IMU yaw so gravity/gyro stay consistent). Climbing
holds yaw~0 (heading-hold up the +x stairs), so the rotation is small; validate the
heightscan against a known staircase before relying on it off-axis.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from go2_locomotion.pgtt_heightmap import (
    PGTT_DIST_X, PGTT_DIST_Y, PGTT_N_ROWS, PGTT_N_COLS, PGTT_N_POINTS,
)

# Half-extents of the grid footprint (m) used to reject points outside it.
_HALF_FWD = (PGTT_N_ROWS - 1) / 2.0 * PGTT_DIST_X + PGTT_DIST_X / 2.0   # ~0.55
_HALF_LAT = (PGTT_N_COLS - 1) / 2.0 * PGTT_DIST_Y + PGTT_DIST_Y / 2.0   # ~0.45


class HeightscanProvider:
    def __init__(
        self,
        mode: str = "flat",
        *,
        cell_percentile: float = 80.0,
        ewma: float = 0.5,
        scale: float = 1.0,
    ) -> None:
        self.mode = str(mode)
        self.cell_percentile = float(cell_percentile)
        self.ewma = float(np.clip(ewma, 0.0, 1.0))
        self.scale = float(scale)
        # Body-frame elevation grid (rows front->rear, cols left->right). NaN = no data.
        self._grid = np.zeros((PGTT_N_ROWS, PGTT_N_COLS), dtype=np.float32)
        self._have_data = False

    # ------------------------------------------------------------------ ingest
    def update_from_points(self, points_body: np.ndarray) -> None:
        """Bin a body-frame ``(N,3)`` cloud into per-cell surface heights (lidar mode)."""
        if self.mode != "lidar":
            return
        pts = np.asarray(points_body, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
            return
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        keep = (np.abs(x) <= _HALF_FWD) & (np.abs(y) <= _HALF_LAT) & np.isfinite(z)
        x, y, z = x[keep], y[keep], z[keep]
        if x.size == 0:
            return
        # body forward x -> row i = 5 - x/dist_x ; body lateral y -> col j = 4 - y/dist_y
        i = np.clip(np.round((PGTT_N_ROWS - 1) / 2.0 - x / PGTT_DIST_X), 0, PGTT_N_ROWS - 1).astype(int)
        j = np.clip(np.round((PGTT_N_COLS - 1) / 2.0 - y / PGTT_DIST_Y), 0, PGTT_N_COLS - 1).astype(int)
        new = np.full((PGTT_N_ROWS, PGTT_N_COLS), np.nan, dtype=np.float32)
        flat = i * PGTT_N_COLS + j
        for cell in np.unique(flat):
            zc = z[flat == cell]
            new.flat[cell] = float(np.percentile(zc, self.cell_percentile))
        filled = self._fill_gaps(new)
        if not self._have_data:
            self._grid = filled
            self._have_data = True
        else:
            a = self.ewma
            self._grid = (a * filled + (1.0 - a) * self._grid).astype(np.float32)

    @staticmethod
    def _fill_gaps(grid: np.ndarray) -> np.ndarray:
        """Fill empty cells from the median of valid cells (ground plane proxy)."""
        out = grid.copy()
        mask = np.isnan(out)
        if mask.all():
            return np.zeros_like(out)
        out[mask] = float(np.nanmedian(out))
        return out

    # ------------------------------------------------------------------ outputs
    def grid_raw(self) -> np.ndarray:
        """Absolute body-frame elevations raveled (99,) -- what the topic carries so the
        low-level node can rebuild height_fn. Zeros until lidar data has arrived."""
        if self.mode != "lidar" or not self._have_data:
            return np.zeros(PGTT_N_POINTS, dtype=np.float32)
        return self._grid.ravel().astype(np.float32)

    def heightscan_99(self) -> np.ndarray:
        """The 99-value heightscan (subtract-min, scaled) for the diagnostic/HUD."""
        if self.mode != "lidar" or not self._have_data:
            return np.zeros(PGTT_N_POINTS, dtype=np.float32)
        z = self._grid.ravel().astype(np.float32)
        return ((z - z.min()) * self.scale).astype(np.float32)

    def height_fn(self) -> Callable[[float, float], float]:
        """Return a ``height_fn(x, y)`` for PGTT to sample (body-frame elevation)."""
        if self.mode != "lidar" or not self._have_data:
            return lambda x, y: 0.0
        return height_fn_from_grid(self._grid.ravel())


def height_fn_from_grid(grid_flat99) -> Callable[[float, float], float]:
    """Build a bilinear body-frame ``height_fn(x, y)`` from a raveled 99-cell grid.

    Used by the low-level control node to reconstruct PGTT's height_fn from the
    ``/go2/heightscan`` topic the LiDAR node publishes (the closure can't cross the
    process boundary, but the grid can).
    """
    grid = np.asarray(grid_flat99, dtype=np.float32).reshape(PGTT_N_ROWS, PGTT_N_COLS)

    def _fn(x: float, y: float) -> float:
        # Inverse of the binning, with bilinear interpolation; clamp outside the grid.
        fi = (PGTT_N_ROWS - 1) / 2.0 - float(x) / PGTT_DIST_X
        fj = (PGTT_N_COLS - 1) / 2.0 - float(y) / PGTT_DIST_Y
        fi = min(max(fi, 0.0), PGTT_N_ROWS - 1.0)
        fj = min(max(fj, 0.0), PGTT_N_COLS - 1.0)
        i0, j0 = int(np.floor(fi)), int(np.floor(fj))
        i1 = min(i0 + 1, PGTT_N_ROWS - 1)
        j1 = min(j0 + 1, PGTT_N_COLS - 1)
        di, dj = fi - i0, fj - j0
        top = grid[i0, j0] * (1 - dj) + grid[i0, j1] * dj
        bot = grid[i1, j0] * (1 - dj) + grid[i1, j1] * dj
        return float(top * (1 - di) + bot * di)

    return _fn
