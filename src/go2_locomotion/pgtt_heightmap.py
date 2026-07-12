"""Body-aligned heightmap sensor for the PGTT policy (numpy only).

Direct port of ``deploy/cpu_heightmap/heightmap.py:create_sensor_matrix`` from
github.com/NtagkasAlex/phase_guided_terrain_traversal, with the one MuJoCo
``mj_ray`` call replaced by an injected ``height_fn(x, y) -> world_z``. Everything
else -- the grid layout, the body->world yaw rotation, the C-order ravel and the
subtract-min normalization -- is reproduced EXACTLY so the 99-value observation
matches the distribution the policy was trained/deployed on.

Grid (Go2 defaults from ``go2/robot_config.py``): 11 rows x 9 cols, 0.1 m spacing,
centered on the robot base, rotated to body frame by the base yaw.
  - row index ``idx_h`` 0..10 maps to offset ``p = (5 - idx_h) * dist_x``:
    ``idx_h=0`` is +0.5 m FORWARD, ``idx_h=10`` is -0.5 m REAR.
  - col index ``idx_w`` 0..8 maps to offset ``k = (4 - idx_w) * dist_y``:
    ``idx_w=0`` is +0.4 m LEFT, ``idx_w=8`` is -0.4 m RIGHT.
The observation is ``z.ravel()`` (C-order: row-major, front->rear then left->right)
minus its own minimum, optionally scaled.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# Go2 PGTT heightmap geometry (go2/robot_config.py: dist_x/y, num_height/widthscans).
PGTT_DIST_X = 0.1
PGTT_DIST_Y = 0.1
PGTT_N_ROWS = 11   # num_heightscans (forward/back)
PGTT_N_COLS = 9    # num_widthscans (left/right)
PGTT_N_POINTS = PGTT_N_ROWS * PGTT_N_COLS  # 99


def build_grid_xy(
    center_xy: Tuple[float, float],
    yaw: float,
    *,
    dist_x: float = PGTT_DIST_X,
    dist_y: float = PGTT_DIST_Y,
    n_rows: int = PGTT_N_ROWS,
    n_cols: int = PGTT_N_COLS,
) -> np.ndarray:
    """World (x, y) of every grid cell, shape ``(n_rows, n_cols, 2)``.

    Exact reproduction of the upstream meshgrid + ``offsets @ R_W2H`` rotation.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    # R_W2H rotates the body-frame offsets into the world frame (matches source).
    R_W2H = np.array([[cy, sy], [-sy, cy]], dtype=np.float64)

    c_h = (n_rows - 1) / 2.0
    c_w = (n_cols - 1) / 2.0
    idx_h = np.arange(n_rows)
    idx_w = np.arange(n_cols)
    p, k = np.meshgrid(c_h - idx_h, c_w - idx_w, indexing="ij")
    offsets = np.stack([p * dist_x, k * dist_y], axis=-1)  # (n_rows, n_cols, 2)
    offsets = offsets @ R_W2H

    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    xy = center + offsets
    # Source pins the center cell exactly to the base XY; the offset there is
    # already (0, 0) so this is a no-op kept for fidelity.
    cr, cc = int(round(c_h)), int(round(c_w))
    xy[cr, cc] = center
    return xy


def build_heightscan(
    center_xy: Tuple[float, float],
    yaw: float,
    height_fn: Callable[[float, float], float],
    *,
    dist_x: float = PGTT_DIST_X,
    dist_y: float = PGTT_DIST_Y,
    n_rows: int = PGTT_N_ROWS,
    n_cols: int = PGTT_N_COLS,
    scale: float = 1.0,
    drop_cap: Optional[float] = None,
    out_stats: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Return the PGTT heightscan observation: ``(n_rows*n_cols,)`` float32.

    ``height_fn(x, y)`` must return the absolute world Z of the terrain top at
    ``(x, y)`` -- i.e. where a ray cast straight down would hit (this is exactly
    what the sim ``get_terrain_height`` returns, and what a real elevation map
    provides). The result is ``z.ravel(C) - min(z)`` then ``* scale``.

    ``drop_cap`` (incident 8.15/8.16 F3, run 11 evidence -- isaac_env.jsonl
    ``pgtt_heightscan`` events jumped ``hs_max`` 0.0 -> 2.1 and ``action_norm`` 1.1 -> 3.15
    once the +/-0.5x+/-0.4 m scan footprint crossed the 2.1 m top-landing edge): when not
    None, clamp every cell's raw ``z`` from BELOW to ``(center_z - drop_cap)``, where
    ``center_z`` is the grid's OWN center cell (``build_grid_xy``'s ``(cr, cc)`` -- the
    terrain directly under the robot base), BEFORE the ``z.min()`` normalization below. One
    over-the-edge cell otherwise pulls the min down by the full drop, and since every OTHER
    cell is normalized against that same min, the whole 99-cell grid shifts up by the drop
    (here 2.1 m) -- wildly out of the training distribution. The default caller value (0.6 m,
    see ``isaac_args.py --pgtt-heightscan-drop-cap-m``) stays deep enough to leave legitimate
    descending-stair reads intact at the crest straddle (rear cells within the +/-0.5 m
    footprint legitimately read up to ~3 x 0.15-0.175 m of riser below the base).

    ``out_stats``, if given a dict, is updated in place with ``clamp_engaged`` (bool) and
    ``clamp_cells`` (int, cells actually pulled up) so the caller can log a boot-time
    diagnostic when the guard actually engages (incident 8.8) without recomputing anything.
    """
    xy = build_grid_xy(
        center_xy, yaw, dist_x=dist_x, dist_y=dist_y, n_rows=n_rows, n_cols=n_cols
    )
    z = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            z[i, j] = float(height_fn(float(xy[i, j, 0]), float(xy[i, j, 1])))
    if drop_cap is not None:
        cr = int(round((n_rows - 1) / 2.0))
        cc = int(round((n_cols - 1) / 2.0))
        z_floor = z[cr, cc] - float(drop_cap)
        clamped_mask = z < z_floor
        n_clamped = int(np.count_nonzero(clamped_mask))
        if n_clamped:
            z = np.where(clamped_mask, z_floor, z)
        if out_stats is not None:
            out_stats["clamp_engaged"] = bool(n_clamped > 0)
            out_stats["clamp_cells"] = n_clamped
    z_vec = z.ravel()  # C-order: rows front->rear, within each row cols left->right
    z_rel = z_vec - z_vec.min()
    return (z_rel * float(scale)).astype(np.float32)
