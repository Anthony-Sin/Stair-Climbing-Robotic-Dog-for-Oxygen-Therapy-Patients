"""Simulated Hesai XT16 LiDAR for the Isaac Go2 environment.

This casts *real* rays against the scene's collision geometry in the XT16 scan
pattern (16 channels, -15 deg .. +15 deg vertical, 360 deg horizontal) and turns
the returns into OpenCV images so the demo preview can show what the robot's
actual LiDAR would see -- a bird's-eye-view (BEV) scatter and an unrolled range
image.

Unlike the synthetic ``_get_analytical_terrain_height`` probe in
``sim_go2_locomotion.py`` (which just reads hard-coded stair geometry), this
module knows nothing about the scene: it only fires rays through an injected
``raycast_fn`` and plots whatever comes back. The raycast itself is injected so
this file stays free of ``omni`` imports and is testable on its own.

Frames:
  * world frame  -- (x, y, z), z up, used for the raycast origin/direction.
  * sensor frame -- x forward (robot heading), y left, z up; used for plotting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

# XT16 hardware geometry (Hesai XT16 datasheet / ros2_ws hesai_xt16 config):
#   16 channels, vertical FOV -15 deg .. +15 deg (2 deg spacing), 360 deg azimuth.
XT16_CHANNELS = 16
XT16_VERT_MIN_DEG = -15.0
XT16_VERT_MAX_DEG = 15.0
# Datasheet range is 0.05 m .. 120 m; capped here for the indoor stair scene so
# the BEV/range image colour scales stay useful.
XT16_MAX_RANGE_M = 50.0
XT16_MIN_RANGE_M = 0.05

# raycast_fn(origin_xyz, direction_xyz, max_dist_m) -> hit distance in metres,
# or None when the ray hits nothing. The direction is expected unit-length.
RaycastFn = Callable[
    [Tuple[float, float, float], Tuple[float, float, float], float], Optional[float]
]


@dataclass
class Xt16Config:
    channels: int = XT16_CHANNELS
    vert_min_deg: float = XT16_VERT_MIN_DEG
    vert_max_deg: float = XT16_VERT_MAX_DEG
    azimuth_step_deg: float = 3.0
    max_range_m: float = XT16_MAX_RANGE_M
    min_range_m: float = XT16_MIN_RANGE_M
    # Sensor mount in the robot base frame (the XT16 sits on the dog's back).
    mount_x_m: float = 0.0
    mount_y_m: float = 0.0
    mount_z_m: float = 0.10

    @property
    def n_azimuth(self) -> int:
        return max(1, int(round(360.0 / max(1e-3, self.azimuth_step_deg))))

    def vertical_angles_deg(self) -> np.ndarray:
        return np.linspace(self.vert_min_deg, self.vert_max_deg, self.channels)

    def azimuth_angles_deg(self) -> np.ndarray:
        return np.arange(self.n_azimuth) * (360.0 / self.n_azimuth)


@dataclass
class Xt16Scan:
    config: Xt16Config
    ranges: np.ndarray          # (channels, n_azimuth) metres, NaN where no return
    points_sensor: np.ndarray   # (N, 3) hits in sensor frame (x fwd, y left, z up)
    origin_world: Tuple[float, float, float]
    yaw_rad: float
    n_rays: int
    n_hits: int

    @property
    def hit_ratio(self) -> float:
        return (self.n_hits / self.n_rays) if self.n_rays else 0.0

    @property
    def min_range_m(self) -> Optional[float]:
        if self.n_hits == 0:
            return None
        return float(np.nanmin(self.ranges))


def cast_scan(
    config: Xt16Config,
    origin_world: Tuple[float, float, float],
    yaw_rad: float,
    raycast_fn: RaycastFn,
) -> Xt16Scan:
    """Fire the full XT16 ray pattern from the sensor mount and collect returns."""
    ox, oy, oz = origin_world
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    # Mount offset expressed in the robot base frame, rotated into world.
    sx = ox + cy * config.mount_x_m - sy * config.mount_y_m
    syw = oy + sy * config.mount_x_m + cy * config.mount_y_m
    sz = oz + config.mount_z_m
    origin = (sx, syw, sz)

    vert = np.radians(config.vertical_angles_deg())
    az = np.radians(config.azimuth_angles_deg())
    n_ch, n_az = config.channels, config.n_azimuth

    ranges = np.full((n_ch, n_az), np.nan, dtype=np.float32)
    pts = []
    n_rays = 0
    n_hits = 0
    max_r = float(config.max_range_m)
    min_r = float(config.min_range_m)

    for ci in range(n_ch):
        elev = float(vert[ci])
        cos_e = math.cos(elev)
        sin_e = math.sin(elev)
        for ai in range(n_az):
            sa = float(az[ai])              # azimuth relative to robot heading
            world_az = yaw_rad + sa
            dx = math.cos(world_az) * cos_e
            dy = math.sin(world_az) * cos_e
            dz = sin_e
            n_rays += 1
            dist = raycast_fn(origin, (dx, dy, dz), max_r)
            if dist is None or dist < min_r or dist > max_r:
                continue
            ranges[ci, ai] = dist
            n_hits += 1
            pts.append(
                (
                    math.cos(sa) * cos_e * dist,   # x forward
                    math.sin(sa) * cos_e * dist,   # y left
                    sin_e * dist,                  # z up
                )
            )

    points = np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 3), np.float32)
    return Xt16Scan(config, ranges, points, origin, float(yaw_rad), n_rays, n_hits)


# ---------------------------------------------------------------------------
# OpenCV rendering
# ---------------------------------------------------------------------------
_PANEL_W = 480
_BEV_H = 480
_HEADER_H = 40
_RANGE_LABEL_H = 24
_RANGE_H = 160


def _height_colors(z: np.ndarray, z_min: float = -0.4, z_max: float = 1.2) -> np.ndarray:
    """Map per-point height to a BGR colour via a JET colormap."""
    import cv2

    t = np.clip((z - z_min) / max(1e-3, (z_max - z_min)), 0.0, 1.0)
    gray = (t * 255.0).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET).reshape(-1, 3)
    return bgr


def render_bev(scan: Xt16Scan, view_range_m: float = 6.0) -> np.ndarray:
    """Top-down bird's-eye-view scatter of the returns, coloured by height."""
    import cv2

    img = np.full((_BEV_H, _PANEL_W, 3), 12, dtype=np.uint8)
    cx = _PANEL_W // 2
    cy = _BEV_H // 2
    scale = (min(_PANEL_W, _BEV_H) * 0.5) / max(0.5, view_range_m)  # px per metre

    # Range rings + labels.
    for r in range(1, int(view_range_m) + 1):
        cv2.circle(img, (cx, cy), int(r * scale), (45, 45, 45), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r}m", (cx + int(r * scale) - 18, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, _BEV_H), (45, 45, 45), 1)
    cv2.line(img, (0, cy), (_PANEL_W, cy), (45, 45, 45), 1)

    pts = scan.points_sensor
    if pts.shape[0]:
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
        # Sensor x (forward) -> screen up; sensor y (left) -> screen left.
        u = (cx - py * scale).astype(np.int32)
        v = (cy - px * scale).astype(np.int32)
        inb = (u >= 0) & (u < _PANEL_W) & (v >= 0) & (v < _BEV_H)
        if np.any(inb):
            colors = _height_colors(pz[inb])
            uu, vv = u[inb], v[inb]
            img[vv, uu] = colors
            # Thicken to 2x2 so sparse returns stay visible.
            for du, dv in ((1, 0), (0, 1), (1, 1)):
                mu, mv = uu + du, vv + dv
                ok = (mu < _PANEL_W) & (mv < _BEV_H)
                img[mv[ok], mu[ok]] = colors[ok]

    # Robot at centre + heading arrow (pointing screen-up = forward).
    cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.arrowedLine(img, (cx, cy), (cx, cy - int(0.7 * scale)),
                    (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
    return img


def render_range_image(scan: Xt16Scan, view_range_m: float = 6.0) -> np.ndarray:
    """Unrolled 16-channel range panorama (near = warm, no-return = dark)."""
    import cv2

    r = scan.ranges
    valid = ~np.isnan(r)
    norm = np.zeros(r.shape, dtype=np.uint8)
    rr = np.clip(r / max(0.5, view_range_m), 0.0, 1.0)
    # Near returns bright, far returns dark.
    norm[valid] = (255.0 * (1.0 - rr[valid])).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[~valid] = (25, 25, 25)
    # Channel 0 = -15 deg (bottom). Flip so +15 deg sits on the top row.
    color = np.flipud(color)
    return cv2.resize(color, (_PANEL_W, _RANGE_H), interpolation=cv2.INTER_NEAREST)


def render_preview(scan: Xt16Scan, view_range_m: float = 6.0) -> np.ndarray:
    """Stack a header, the BEV, and the range image into one preview frame."""
    import cv2

    bev = render_bev(scan, view_range_m)
    rng = render_range_image(scan, view_range_m)

    header = np.full((_HEADER_H, _PANEL_W, 3), 20, dtype=np.uint8)
    near = scan.min_range_m
    near_txt = f"{near:.2f}m" if near is not None else "--"
    cv2.putText(header, "HESAI XT16 (SIM RAYCAST)", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(header,
                f"{scan.n_hits}/{scan.n_rays} hits  {scan.hit_ratio * 100:.0f}%  "
                f"near {near_txt}  {scan.config.channels}ch",
                (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    rng_label = np.full((_RANGE_LABEL_H, _PANEL_W, 3), 20, dtype=np.uint8)
    cv2.putText(rng_label, "RANGE IMAGE  (+15 top / -15 bottom, 360 deg)", (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1, cv2.LINE_AA)

    return np.vstack([header, bev, rng_label, rng])
