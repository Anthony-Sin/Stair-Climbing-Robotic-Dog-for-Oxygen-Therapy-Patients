"""OpenCV preview rendering for the simulated XT16 LiDAR.

Split out of ``sim_lidar_xt16`` (which stays the pure sensor model). Every
pixel is plotted from the *real* ray returns in the scan (PhysX raycast against
the scene's collision geometry); nothing here is fabricated. The BEV adds a
forward perception cone and an obstacle contour so the preview reads as "what
the LiDAR actually sees ahead of the robot".

``cv2`` is imported lazily inside each function so this module stays cheap to
import on hosts without OpenCV.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle; annotations stay strings
    from perception.sim_lidar_xt16 import Xt16Scan


_PANEL_W = 480
_BEV_H = 480
_HEADER_H = 46
_RANGE_LABEL_H = 24
_RANGE_H = 150
_LEGEND_H = 30

# Forward perception cone half-angle drawn on the BEV (front RealSense D435
# horizontal FOV ~86 deg). Purely a visual highlight of the "ahead" sector.
_FRONT_FOV_DEG = 86.0

# Palette (BGR).
_BG = (20, 18, 16)            # panel background (near-black slate)
_PANE = (26, 23, 20)          # header/label band
_GRID = (54, 50, 46)          # range rings / spokes
_FREE = (46, 52, 34)          # shaded "scanned free space" inside the contour
_FOV_FILL = (78, 60, 24)      # forward cone tint
_CONTOUR = (150, 230, 255)    # obstacle boundary (warm cyan)
_CONTOUR_FRONT = (90, 255, 255)  # boundary inside the forward cone (brighter)
_ROBOT = (0, 230, 255)        # robot body / heading
_TXT = (210, 210, 210)
_TXT_DIM = (120, 120, 120)
_ACCENT = (0, 220, 255)
_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


def _height_colors(z: np.ndarray, z_min: float = -0.4, z_max: float = 1.2) -> np.ndarray:
    """Map per-point height to a BGR colour via a JET colormap."""
    import cv2

    t = np.clip((z - z_min) / max(1e-3, (z_max - z_min)), 0.0, 1.0)
    gray = (t * 255.0).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET).reshape(-1, 3)
    return bgr


def _nearest_per_azimuth(scan: Xt16Scan) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse the (channel, azimuth) grid to the nearest return per azimuth.

    Returns ``(azimuth_rad, nearest_m)`` where azimuth 0 is straight ahead and
    increases toward the robot's left (+y). ``nearest_m == 0`` marks an azimuth
    column with no return at all. This is the same nearest-per-column reduction
    the controller profile uses, reused here to draw the obstacle contour.
    """
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    if ranges.size == 0 or ranges.shape[1] == 0:
        return np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    finite = np.where(np.isnan(ranges), np.inf, ranges)
    nearest = finite.min(axis=0)
    nearest = np.where(np.isfinite(nearest), nearest, 0.0).astype(np.float32)
    n = int(nearest.shape[0])
    az = np.arange(n, dtype=np.float32) * (2.0 * math.pi / max(1, n))
    return az, nearest


def _forward_min_range(az_rad: np.ndarray, nearest_m: np.ndarray,
                       fov_deg: float) -> Optional[float]:
    """Nearest real return within +/- fov_deg/2 of straight ahead, or None."""
    if az_rad.size == 0:
        return None
    half = math.radians(fov_deg * 0.5)
    a = (az_rad + math.pi) % (2.0 * math.pi) - math.pi  # wrap to [-pi, pi]
    mask = (np.abs(a) <= half) & (nearest_m > 0.0)
    if not np.any(mask):
        return None
    return float(np.min(nearest_m[mask]))


def _ahead_color(dist_m: Optional[float]) -> Tuple[int, int, int]:
    """Green (clear) -> amber -> red (close) for the ahead-distance callout."""
    if dist_m is None:
        return _TXT_DIM
    if dist_m < 1.0:
        return (40, 40, 235)    # red
    if dist_m < 2.0:
        return (40, 170, 240)   # amber
    return (90, 220, 120)       # green


def render_bev(scan: Xt16Scan, view_range_m: float = 6.0) -> np.ndarray:
    """Top-down bird's-eye view: forward cone, scanned free space, the obstacle
    contour the LiDAR sees, and the height-coloured returns."""
    import cv2

    img = np.full((_BEV_H, _PANEL_W, 3), _BG, dtype=np.uint8)
    cx = _PANEL_W // 2
    cy = _BEV_H // 2
    radius_px = min(_PANEL_W, _BEV_H) * 0.5 - 6.0
    scale = radius_px / max(0.5, view_range_m)  # px per metre

    az, nearest = _nearest_per_azimuth(scan)
    half_fov = math.radians(_FRONT_FOV_DEG * 0.5)

    # --- forward perception cone (translucent wedge pointing screen-up) -------
    overlay = img.copy()
    cone = [(cx, cy)]
    for d in np.linspace(-half_fov, half_fov, 28):
        cone.append((int(round(cx - math.sin(d) * radius_px)),
                     int(round(cy - math.cos(d) * radius_px))))
    cv2.fillPoly(overlay, [np.array(cone, np.int32)], _FOV_FILL)
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0.0, img)

    # --- scanned free space: fill inside the nearest-return contour -----------
    if az.size:
        disp = np.where(nearest > 0.0, np.minimum(nearest, view_range_m), view_range_m)
        u = (cx - np.sin(az) * disp * scale)
        v = (cy - np.cos(az) * disp * scale)
        poly = np.stack([u, v], axis=1).astype(np.int32)
        free = img.copy()
        cv2.fillPoly(free, [poly], _FREE)
        cv2.addWeighted(free, 0.45, img, 0.55, 0.0, img)

    # --- range rings + 45 deg spokes (over the fills so they stay readable) ---
    for r in range(1, int(view_range_m) + 1):
        rp = int(round(r * scale))
        if rp < radius_px:
            cv2.circle(img, (cx, cy), rp, _GRID, 1, cv2.LINE_AA)
            cv2.putText(img, f"{r}m", (cx + rp - 18, cy - 4),
                        _FONT, 0.3, _TXT_DIM, 1, cv2.LINE_AA)
    for deg in range(0, 360, 45):
        a = math.radians(deg)
        cv2.line(img, (cx, cy),
                 (int(round(cx - math.sin(a) * radius_px)),
                  int(round(cy - math.cos(a) * radius_px))),
                 _GRID, 1, cv2.LINE_AA)

    # --- obstacle contour: connect adjacent real returns (brighter ahead) -----
    if az.size:
        hit = nearest > 0.0
        a_wrap = (az + math.pi) % (2.0 * math.pi) - math.pi
        front = np.abs(a_wrap) <= half_fov
        n = poly.shape[0]
        for i in range(n):
            j = (i + 1) % n
            if hit[i] and hit[j]:
                col = _CONTOUR_FRONT if (front[i] and front[j]) else _CONTOUR
                thick = 2 if (front[i] and front[j]) else 1
                cv2.line(img, tuple(poly[i]), tuple(poly[j]), col, thick, cv2.LINE_AA)

    # --- height-coloured returns (every channel) on top of the contour --------
    pts = scan.points_sensor
    if pts.shape[0]:
        px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
        u = (cx - py * scale).astype(np.int32)
        v = (cy - px * scale).astype(np.int32)
        inb = (u >= 0) & (u < _PANEL_W) & (v >= 0) & (v < _BEV_H)
        if np.any(inb):
            colors = _height_colors(pz[inb])
            uu, vv = u[inb], v[inb]
            img[vv, uu] = colors
            for du, dv in ((1, 0), (0, 1), (1, 1)):
                mu, mv = uu + du, vv + dv
                ok = (mu < _PANEL_W) & (mv < _BEV_H)
                img[mv[ok], mu[ok]] = colors[ok]

    # --- nearest-obstacle-ahead callout ---------------------------------------
    fwd = _forward_min_range(az, nearest, _FRONT_FOV_DEG)
    if fwd is not None:
        col = _ahead_color(fwd)
        ry = int(round(cy - min(fwd, view_range_m) * scale))
        cv2.line(img, (cx - 9, ry), (cx + 9, ry), col, 2, cv2.LINE_AA)
        cv2.putText(img, f"{fwd:.2f}m", (cx + 12, ry + 4),
                    _FONT, 0.4, col, 1, cv2.LINE_AA)

    # --- robot footprint (~0.7 x 0.32 m) + heading nose, pointing up = fwd ----
    hl = max(7, int(round(0.35 * scale)))
    hw = max(4, int(round(0.16 * scale)))
    cv2.rectangle(img, (cx - hw, cy - hl), (cx + hw, cy + hl), (36, 34, 30), -1)
    cv2.rectangle(img, (cx - hw, cy - hl), (cx + hw, cy + hl), _ROBOT, 1, cv2.LINE_AA)
    nose = np.array([[cx, cy - hl - 8], [cx - 6, cy - hl + 1], [cx + 6, cy - hl + 1]],
                    dtype=np.int32)
    cv2.fillPoly(img, [nose], _ROBOT)

    # --- compass labels --------------------------------------------------------
    for txt, (lx, ly) in (("FWD", (cx - 14, 16)), ("BACK", (cx - 16, _BEV_H - 6)),
                          ("L", (6, cy + 4)), ("R", (_PANEL_W - 16, cy + 4))):
        cv2.putText(img, txt, (lx, ly), _FONT, 0.36, _ACCENT, 1, cv2.LINE_AA)
    return img


def render_range_image(scan: Xt16Scan, view_range_m: float = 6.0) -> np.ndarray:
    """Unrolled 16-channel range panorama (near = warm, no-return = dark), with a
    0 deg-elevation horizon line."""
    import cv2

    r = scan.ranges
    valid = ~np.isnan(r)
    norm = np.zeros(r.shape, dtype=np.uint8)
    rr = np.clip(r / max(0.5, view_range_m), 0.0, 1.0)
    # Near returns bright, far returns dark.
    norm[valid] = (255.0 * (1.0 - rr[valid])).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[~valid] = (28, 26, 24)
    # Channel 0 = -15 deg (bottom). Flip so +15 deg sits on the top row.
    color = np.flipud(color)
    out = cv2.resize(color, (_PANEL_W, _RANGE_H), interpolation=cv2.INTER_NEAREST)
    # Horizon (0 deg elevation) marker.
    cv2.line(out, (0, _RANGE_H // 2), (_PANEL_W, _RANGE_H // 2), (90, 90, 90), 1, cv2.LINE_AA)
    return out


def render_preview(scan: Xt16Scan, view_range_m: float = 6.0) -> np.ndarray:
    """Stack header, BEV, an azimuth-ticked range image, and a height legend."""
    import cv2

    bev = render_bev(scan, view_range_m)
    rng = render_range_image(scan, view_range_m)

    # --- header: title + live scan stats + nearest-ahead ----------------------
    header = np.full((_HEADER_H, _PANEL_W, 3), _PANE, dtype=np.uint8)
    near = scan.min_range_m
    near_txt = f"{near:.2f}m" if near is not None else "--"
    az, nearest = _nearest_per_azimuth(scan)
    fwd = _forward_min_range(az, nearest, _FRONT_FOV_DEG)
    fwd_txt = f"{fwd:.2f}m" if fwd is not None else "clear"
    cv2.putText(header, "HESAI XT16  (SIM PHYSX RAYCAST)", (8, 17),
                _FONT, 0.42, _ACCENT, 1, cv2.LINE_AA)
    cv2.putText(header,
                f"{scan.n_hits}/{scan.n_rays} hits  {scan.hit_ratio * 100:.0f}%  "
                f"near {near_txt}  ahead {fwd_txt}  {scan.config.channels}ch",
                (8, 35), _FONT, 0.36, _TXT, 1, cv2.LINE_AA)

    # --- range-strip label with azimuth ticks (F / L / B / R) -----------------
    rng_label = np.full((_RANGE_LABEL_H, _PANEL_W, 3), _PANE, dtype=np.uint8)
    cv2.putText(rng_label, "RANGE  +15/-15deg", (8, 16),
                _FONT, 0.34, _TXT_DIM, 1, cv2.LINE_AA)
    for frac, name in ((0.0, "F"), (0.25, "L"), (0.5, "B"), (0.75, "R"), (1.0, "F")):
        tx = int(round(frac * (_PANEL_W - 1)))
        tx = min(max(tx, 4), _PANEL_W - 10)
        cv2.putText(rng_label, name, (tx, 16), _FONT, 0.34, _ACCENT, 1, cv2.LINE_AA)

    # --- legend: height colormap gradient + view range ------------------------
    legend = np.full((_LEGEND_H, _PANEL_W, 3), _PANE, dtype=np.uint8)
    bar_x0, bar_x1, bar_y0, bar_y1 = 60, 220, 9, 21
    grad = np.linspace(0, 255, bar_x1 - bar_x0).astype(np.uint8).reshape(1, -1)
    grad = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    legend[bar_y0:bar_y1, bar_x0:bar_x1] = np.repeat(grad, bar_y1 - bar_y0, axis=0)
    cv2.rectangle(legend, (bar_x0, bar_y0), (bar_x1 - 1, bar_y1 - 1), _GRID, 1, cv2.LINE_AA)
    cv2.putText(legend, "HEIGHT", (8, 20), _FONT, 0.34, _TXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(legend, "low", (bar_x0 - 1, bar_y1 + 8), _FONT, 0.28, _TXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(legend, "high", (bar_x1 - 18, bar_y1 + 8), _FONT, 0.28, _TXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(legend, f"view {view_range_m:.0f}m  FOV {int(_FRONT_FOV_DEG)}deg",
                (bar_x1 + 14, 20), _FONT, 0.34, _TXT_DIM, 1, cv2.LINE_AA)

    return np.vstack([header, bev, rng_label, rng, legend])
