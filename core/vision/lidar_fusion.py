"""LiDAR (Hesai XT16) <-> YOLO fusion helpers for the sim vision controller.

In the sim, the Isaac process raycasts the robot's XT16 and sends a compact polar
profile (nearest return per azimuth bin) alongside each camera frame. On the real
robot the same profile would come from the Hesai point cloud. These helpers:

  * decode the polar profile into a metres array,
  * map a YOLO bbox to a sensor-frame bearing and sample the LiDAR range there,
  * fuse that LiDAR range with the depth-camera estimate (agreement-weighted).

Bearing convention matches sim_lidar_xt16.cast_scan: azimuth is measured CCW from
straight ahead (the robot/XT16 forward axis), +left, in radians. 0 == ahead.
All pure functions -- no OpenCV/Isaac deps -- so they are unit-testable.
"""
from __future__ import annotations

import base64
import math
import zlib
from typing import Any, Dict, Optional

import numpy as np


def decode_lidar_profile(profile: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Decode the UDP polar profile into {ranges_m, azimuth metadata}.

    Returns None when no usable profile is present (e.g. before the first scan).
    ranges_m holds the nearest XT16 return per azimuth bin in metres; 0.0 means
    "no return" for that bin.
    """
    if not isinstance(profile, dict):
        return None
    blob = profile.get("ranges_mm")
    if not blob:
        return None
    try:
        raw = zlib.decompress(base64.b64decode(blob))
        ranges_mm = np.frombuffer(raw, dtype=np.uint16)
    except Exception:
        return None
    if ranges_mm.size == 0:
        return None
    ranges_m = ranges_mm.astype(np.float32) / 1000.0
    n = int(profile.get("n_azimuth", ranges_m.size)) or ranges_m.size
    return {
        "ranges_m": ranges_m,
        "n_azimuth": n,
        "azimuth_step_deg": float(profile.get("azimuth_step_deg", 360.0 / max(1, n))),
        "view_range_m": float(profile.get("view_range_m", 6.0)),
        "min_range_m": profile.get("min_range_m"),
        "hit_count": int(profile.get("hit_count", 0)),
        "ray_count": int(profile.get("ray_count", 0)),
    }


def person_bearing_rad(
    bbox_center_x: float,
    camera_cx: float,
    camera_fx: float,
    yaw_offset_rad: float = 0.0,
) -> Optional[float]:
    """Sensor-frame bearing (CCW from forward, +left) of a pixel column.

    A target to the image-right (bbox_center_x > cx) is to the robot's right, i.e.
    clockwise, i.e. a negative bearing in the LiDAR's CCW/+left convention.
    """
    if camera_fx <= 0.0:
        return None
    theta = math.atan2(float(bbox_center_x) - float(camera_cx), float(camera_fx))
    return -theta + float(yaw_offset_rad)


def lidar_range_at_bearing(
    decoded: Dict[str, Any],
    bearing_rad: float,
    window_deg: float = 4.0,
) -> Optional[float]:
    """Nearest valid LiDAR return within +/-window_deg of a bearing, or None.

    The followed person is the closest object around their bearing (the
    background -- walls, stairs -- sits farther out), so the nearest return is the
    person's range. This mirrors the depth-camera "foreground" choice and avoids a
    median being dragged to the background when the person spans only a bin or two.
    """
    ranges = decoded.get("ranges_m")
    if ranges is None or ranges.shape[0] == 0:
        return None
    n = ranges.shape[0]
    bearing_deg = math.degrees(bearing_rad) % 360.0
    half = max(float(decoded.get("azimuth_step_deg", 360.0 / n)), float(window_deg))
    bin_deg = np.arange(n) * (360.0 / n)
    # Circular angular distance from each bin to the requested bearing.
    delta = np.abs((bin_deg - bearing_deg + 180.0) % 360.0 - 180.0)
    sel = (delta <= half) & (ranges > 0.0)
    if not np.any(sel):
        return None
    return float(np.min(ranges[sel]))


def person_bearing_from_profile(
    decoded: Dict[str, Any],
    prior_bearing_rad: float,
    prior_range_m: Optional[float] = None,
    *,
    window_deg: float = 60.0,
    range_margin_m: float = 1.0,
    min_range_m: float = 0.2,
) -> Optional[float]:
    """Recover the followed person's bearing from the LiDAR profile when YOLO has no box.

    The 69 deg RGB camera (YOLO) is narrower than the 360 deg LiDAR, so a patient that
    turns hard at a close standoff leaves the RGB frame while still being the nearest
    object on the LiDAR. When the bbox is gone we still know roughly WHERE the patient was
    (``prior_bearing_rad``) and roughly HOW FAR (``prior_range_m``). Within +/-window_deg of
    that prior bearing, pick the nearest FOREGROUND return (closest range, optionally capped
    at ``prior_range + range_margin`` so a far wall/stair is rejected) and return its bearing.

    Returns the bearing in radians (CCW from forward, +left -- the same convention as
    ``person_bearing_rad`` / ``cast_scan``), or None when no consistent foreground return
    exists in the window. Pure function -- unit-testable, no Isaac/OpenCV deps.
    """
    ranges = decoded.get("ranges_m")
    if ranges is None or ranges.shape[0] == 0:
        return None
    n = ranges.shape[0]
    bin_deg = np.arange(n) * (360.0 / n)
    prior_deg = math.degrees(float(prior_bearing_rad)) % 360.0
    # Circular angular distance from each bin to the prior bearing.
    delta = np.abs((bin_deg - prior_deg + 180.0) % 360.0 - 180.0)
    sel = (delta <= float(window_deg)) & (ranges > float(min_range_m))
    if prior_range_m is not None and float(prior_range_m) > 0.0:
        sel = sel & (ranges <= float(prior_range_m) + float(range_margin_m))
    if not np.any(sel):
        return None
    idx = np.where(sel)[0]
    j = int(idx[int(np.argmin(ranges[idx]))])
    # Signed bearing in (-180, 180], CCW/+left.
    signed_deg = ((float(bin_deg[j]) + 180.0) % 360.0) - 180.0
    return math.radians(signed_deg)


def fuse_distance(
    depth_m: Optional[float],
    lidar_m: Optional[float],
    *,
    agree_tol_m: float = 0.25,
    rel_tol: float = 0.15,
    lidar_weight: float = 0.6,
    reject_far_depth: bool = True,
    far_depth_ratio: float = 2.0,
    confident_near_lidar_m: float = 3.0,
) -> Dict[str, Any]:
    """Agreement-weighted blend of the depth-camera and LiDAR distances.

    - both valid and agree (|d-l| <= tol) -> weighted blend, high confidence
    - both valid, depth ABSURDLY FAR vs a confident near LiDAR -> use LiDAR
    - both valid and disagree              -> fall back to depth, flag, low conf
    - only one valid                       -> use it
    - neither valid                        -> fused_m None
    Returns {fused_m, lidar_m, depth_m, confidence, disagreement, source}.

    Plausibility guard (``reject_far_depth``, default on): the bimodal/foreground
    depth occasionally latches onto the far background and reports an absurd range
    (a frame accepted fused=36.87 m while the LiDAR said 0.59 m). When BOTH sources
    are valid, the LiDAR return is a confident NEAR one (<= ``confident_near_lidar_m``),
    and depth exceeds it by more than ``far_depth_ratio`` x, the depth is REJECTED and
    the LiDAR range is used instead of accepting the impossible depth.
    """
    d = float(depth_m) if (depth_m is not None and depth_m > 0.0) else None
    l = float(lidar_m) if (lidar_m is not None and lidar_m > 0.0) else None

    out: Dict[str, Any] = {"depth_m": d, "lidar_m": l}
    if d is None and l is None:
        out.update(fused_m=None, confidence=0.0, disagreement=False, source="none")
        return out
    if d is None:
        out.update(fused_m=l, confidence=0.6, disagreement=False, source="lidar_only")
        return out
    if l is None:
        out.update(fused_m=d, confidence=0.6, disagreement=False, source="depth_only")
        return out

    # Absurd-far-depth guard: a confident near LiDAR return beats a depth that reads
    # more than far_depth_ratio x farther (the background-latch failure). Take the
    # LiDAR range; flag the disagreement but keep moderate confidence (the near LiDAR
    # is the trustworthy source here, unlike the ambiguous depth-disagree fallback).
    if (
        reject_far_depth
        and l <= float(confident_near_lidar_m)
        and d > float(far_depth_ratio) * l
    ):
        out.update(
            fused_m=l, confidence=0.5, disagreement=True, source="lidar_reject_far_depth"
        )
        return out

    tol = max(float(agree_tol_m), float(rel_tol) * min(d, l))
    if abs(d - l) <= tol:
        w_l = float(np.clip(lidar_weight, 0.0, 1.0))
        fused = w_l * l + (1.0 - w_l) * d
        # Confidence rises the closer the two sources agree (0.5..1.0).
        conf = 0.5 + 0.5 * float(np.clip(1.0 - abs(d - l) / max(1e-6, tol), 0.0, 1.0))
        out.update(fused_m=fused, confidence=round(conf, 3), disagreement=False, source="fused")
        return out

    # Disagreement: keep the proven depth estimate but flag low confidence.
    out.update(fused_m=d, confidence=0.3, disagreement=True, source="depth_disagree")
    return out
