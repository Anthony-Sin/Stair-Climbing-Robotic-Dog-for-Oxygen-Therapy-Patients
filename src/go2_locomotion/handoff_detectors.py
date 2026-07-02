"""Task-2 perception for the dual-policy handoff: stall + depth-stair detectors.

Two independent detectors the ``HandoffController`` owns and polls each step:
  * ``StallDetector`` (2a) -- "commanded forward yet going nowhere", from live
    robot state (no distance threshold).
  * ``DepthStairDetector`` (2b) -- detect + COUNT stairs from the body-mounted
    parkour depth image (not a flat ground-truth heightmap).

Split out of ``pgtt_stair_handoff`` (which now re-exports these) so the
perception helpers are a single-responsibility module the FSM imports.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from go2_locomotion.handoff_config import HandoffConfig


class StallDetector:
    """Task 2a: detect that the walking policy is stalled, from live robot state.

    The trigger is dynamic (no distance threshold): the walker is stalled when it is
    being commanded forward yet the measured body speed stays near zero and the
    commanded-vs-actual divergence is large, sustained for ``stall_consec_sec``.
    """

    def __init__(self, cfg: HandoffConfig) -> None:
        self.cfg = cfg
        self._win: list = []      # rolling [(dt, fwd_displacement)] over the last consec sec
        self._win_sec = 0.0
        self._cmd_sec = 0.0       # continuous time spent commanding forward
        self._stalled = False
        self._last_disp = 0.0

    def reset(self) -> None:
        self._win = []
        self._win_sec = 0.0
        self._cmd_sec = 0.0
        self._stalled = False
        self._last_disp = 0.0

    def update(self, dt: float, cmd_vx: float, body_fwd: Optional[float]) -> bool:
        if body_fwd is None:
            # No velocity odometry: the detector CANNOT measure movement, so it must NOT
            # declare a stall. Coercing None->0 would read "wedged" every step and fire a
            # false climb hand-off / egress-hold on a robot that is walking fine (the real
            # port leaves body_fwd/body_speed None until SportModeState velocity is wired).
            self.reset()
            return False
        cmd = max(0.0, float(cmd_vx))
        fwd = float(body_fwd)
        dt = max(0.0, float(dt))
        # Only a STALL while we are actually commanding forward (a deliberate hold
        # commands ~0 and must not register). Resetting here also means a stall must
        # be sustained UNDER a continuous forward command.
        if cmd < self.cfg.stall_cmd_min_mps:
            self._win = []
            self._win_sec = 0.0
            self._cmd_sec = 0.0
            self._stalled = False
            self._last_disp = 0.0
            return False
        self._cmd_sec += dt
        self._win.append((dt, fwd * dt))
        self._win_sec += dt
        while self._win_sec > self.cfg.stall_consec_sec and len(self._win) > 1:
            d0, _ = self._win.pop(0)
            self._win_sec -= d0
        win_disp = sum(s for _, s in self._win)   # net forward travel over the window
        self._last_disp = win_disp
        # Stalled: commanded forward for at least the window, yet net forward travel
        # over that window is below the progress floor (wedged / going nowhere).
        self._stalled = (
            self._cmd_sec >= self.cfg.stall_consec_sec
            and win_disp < self.cfg.stall_min_progress_m
        )
        return self._stalled

    @property
    def stalled(self) -> bool:
        return self._stalled

    def telemetry(self) -> Dict[str, Any]:
        return {
            "stall_cmd_sec": round(self._cmd_sec, 3),
            "stall_win_disp_m": round(self._last_disp, 4),
            "stalled": bool(self._stalled),
        }


class DepthStairDetector:
    """Task 2b: detect + COUNT stairs from the depth image (not a flat heightmap).

    Builds a central-column depth profile from the body-mounted parkour depth
    camera, back-projects each row to a (forward_distance, world_height) point using
    the camera geometry, then clusters the points into discrete tread LEVELS. Each
    level that sits at least one ``stair_min_riser_m`` above the ground (i.e. a riser
    taller than the robot's leg clearance) and is separated from its neighbours by a
    riser counts as one stair.

    Output: {stair_detected, stair_count, leading_edge_distance, ground_level_m, ...}
    """

    def __init__(self, cfg: HandoffConfig) -> None:
        self.cfg = cfg
        self._empty = {
            "stair_detected": False,
            "stair_count": 0,
            "leading_edge_distance": None,
            "ground_level_m": None,
            "level_heights_m": [],
            "valid_rows": 0,
        }
        self._last = dict(self._empty)

    def reset(self) -> None:
        self._last = dict(self._empty)

    def detect(self, depth_hw: Any) -> Dict[str, Any]:
        D = np.asarray(depth_hw, dtype=np.float32) if depth_hw is not None else None
        if D is None or D.ndim != 2 or D.size == 0:
            self._last = dict(self._empty)
            return self._last
        H, W = D.shape
        band = max(0.05, min(1.0, float(self.cfg.stair_band_frac)))
        c0 = int(round(W * (0.5 - band / 2.0)))
        c1 = int(round(W * (0.5 + band / 2.0)))
        c0 = max(0, c0)
        c1 = min(W, max(c0 + 1, c1))
        sub = D[:, c0:c1]

        max_r = float(self.cfg.stair_max_range_m)
        cy = (H - 1) / 2.0
        vfov = math.radians(float(self.cfg.stair_cam_vfov_deg))
        pitch = math.radians(float(self.cfg.stair_cam_pitch_deg))
        cam_h = float(self.cfg.stair_cam_height_m)

        xs = []  # forward distance (m)
        zs = []  # world height of the surface point (m, ground ~ 0)
        valid_rows = 0
        for r in range(H):
            row = sub[r]
            row = row[np.isfinite(row) & (row > 0.06) & (row < max_r + 0.6)]
            if row.shape[0] < int(self.cfg.stair_min_valid_px):
                continue
            d = float(np.median(row))
            valid_rows += 1
            # Row r increases downward in the image -> the ray points further BELOW the
            # optical axis. theta_v = this row's vertical angle off the optical axis;
            # ang = total downward angle from horizontal (axis pitched down `pitch`).
            # Isaac reports distance_to_image_plane (PERPENDICULAR depth), so the
            # Euclidean range along the ray is d / cos(theta_v).
            theta_v = ((r - cy) / float(H)) * vfov
            ang = pitch + theta_v
            rng = d / max(0.2, math.cos(theta_v))
            x_fwd = rng * math.cos(ang)
            z_h = cam_h - rng * math.sin(ang)
            if 0.10 <= x_fwd <= max_r:
                xs.append(x_fwd)
                zs.append(z_h)

        if len(zs) < 4:
            self._last = {**self._empty, "valid_rows": int(valid_rows)}
            return self._last

        xs_a = np.asarray(xs, dtype=np.float32)
        zs_a = np.asarray(zs, dtype=np.float32)
        min_riser = max(0.02, float(self.cfg.stair_min_riser_m))

        # Cluster ALL profile points into discrete height LEVELS (treads): sort by
        # height and split a new level wherever the height jumps by >= one riser. The
        # number of RISERS (= levels - 1, each gap a step taller than the leg-clearance
        # threshold) is the stair count. Counting risers rather than "levels above the
        # floor" is robust to the near-horizontal parkour cam NOT seeing the base floor
        # (it commonly sees only the tread faces), which would otherwise undercount.
        pts = sorted(((float(x), float(z)) for x, z in zip(xs_a, zs_a)), key=lambda p: p[1])
        level_base = [pts[0][1]]      # lower edge (height) of each level cluster
        level_min_x = [pts[0][0]]     # nearest forward distance seen in each level
        for x, z in pts[1:]:
            if z - level_base[-1] >= min_riser:
                level_base.append(z)
                level_min_x.append(x)
            else:
                level_min_x[-1] = min(level_min_x[-1], x)
        n_levels = len(level_base)
        stair_count = max(0, n_levels - 1)   # risers between consecutive treads
        # Leading edge = nearest forward distance among the elevated levels (the first
        # riser the dog faces); the lowest level is the surface it stands on.
        leading_edge = round(float(min(level_min_x[1:])), 3) if n_levels >= 2 else None
        ground = float(level_base[0])
        level_heights = [round(float(b - ground), 3) for b in level_base]

        self._last = {
            "stair_detected": bool(stair_count >= 1 and leading_edge is not None),
            "stair_count": int(stair_count),
            "leading_edge_distance": leading_edge,
            "ground_level_m": round(ground, 3),
            "level_heights_m": level_heights,
            "valid_rows": int(valid_rows),
        }
        return self._last

    @property
    def last(self) -> Dict[str, Any]:
        return dict(self._last)
