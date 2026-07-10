"""Reactive obstacle avoidance for the person-follow controller.

The follow controller steers the dog straight at the person and regulates the gap.
It has NO notion of obstacles, so in a furnished room it drives into whatever sits
between it and the person (a couch, a coffee table) and wedges -- and that same
furniture occludes the person from its camera, so it loses the follow entirely.

This module adds a REACTIVE steer-around: given furniture obstacles detected by the
open-vocabulary YOLO-World head (the staircase is a SEPARATE class and is never in
this list, so the dog still climbs stairs), each with a depth-measured range, it
picks the most threatening one in the forward driving cone and returns:

  * ``yaw_target_rad`` -- a heading (CCW/+left, matching
    ``vision.lidar_fusion.person_bearing_rad``) that skirts the obstacle's edge on
    the side the PERSON is on (so the detour still heads toward the goal, not away),
  * ``weight`` in [0,1] -- how strongly to blend that heading over the follow heading
    (ramps up as the obstacle gets closer), and
  * ``speed_factor`` in [0,1] -- a forward-speed cut when something close is nearly
    dead-ahead (so the dog slows to turn instead of ramming).

The caller blends ``yaw_target_rad`` into BOTH heading channels (the ``yaw_err`` hint
the hybrid parkour policy self-steers from, and the ``rotation_cmd`` twist) and scales
``trans_x_cmd`` by ``speed_factor``. Commands are yaw + forward only (no lateral vy),
so "avoid" is necessarily "turn toward free space and slow", not "strafe".

Pure/stateless (math + plain dicts, no cv/torch), so the geometry is host-unit-testable
without the perception stack -- see ``src/tests`` / the ``__main__`` self-check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def bearing_rad(pixel_x: float, cam_cx: float, cam_fx: float) -> float:
    """Sensor-frame bearing (CCW from forward, +left) of an image column.

    Sign matches ``vision.lidar_fusion.person_bearing_rad``: a column to the
    image-right (pixel_x > cx) is to the robot's right -> NEGATIVE bearing.
    """
    return -math.atan2(float(pixel_x) - float(cam_cx), float(cam_fx))


@dataclass
class AvoidanceConfig:
    range_m: float = 2.0          # ignore obstacles farther than this
    engage_m: float = 1.3         # weight saturates to 1.0 at/below this range
    cone_deg: float = 24.0        # forward driving cone; only obstacles overlapping it count
    margin_deg: float = 12.0      # steer this far PAST the skirted edge (clearance)
    slow_range_m: float = 1.1     # start cutting forward speed below this (if ~dead-ahead)
    min_speed_factor: float = 0.35
    max_yaw_rad: float = 0.6      # clamp the avoidance heading magnitude
    yaw_gain: float = 1.0         # yaw_target -> rotation_cmd twist gain


def compute_obstacle_avoidance(
    obstacles: Optional[List[Dict[str, Any]]],
    person_bearing: Optional[float],
    cam_cx: float,
    cam_fx: float,
    cfg: AvoidanceConfig,
) -> Dict[str, Any]:
    """Return the avoidance decision for this frame.

    ``obstacles``: list of ``{"bbox": [x1,y1,x2,y2], "range_m": float|None, "label": str}``
    (pixel bbox in the same frame the person bbox is in; ``range_m`` from depth).
    ``person_bearing``: the person's bearing (rad, +left) or None.
    """
    out: Dict[str, Any] = {
        "active": False,
        "yaw_target_rad": None,
        "weight": 0.0,
        "speed_factor": 1.0,
        "threat_range_m": None,
        "threat_bearing_deg": None,
        "pass_side": None,
        "n_considered": 0,
    }
    if not obstacles or cam_fx <= 0.0:
        return out

    cone = math.radians(cfg.cone_deg)
    threats = []
    for ob in obstacles:
        bb = ob.get("bbox")
        rng = ob.get("range_m")
        if not bb or len(bb) < 4 or rng is None:
            continue
        rng = float(rng)
        if rng <= 0.0 or rng > cfg.range_m:
            continue
        x1, x2 = float(bb[0]), float(bb[2])
        b_left = bearing_rad(min(x1, x2), cam_cx, cam_fx)   # smaller pixel_x -> more +left
        b_right = bearing_rad(max(x1, x2), cam_cx, cam_fx)
        # obstacle interval [b_right, b_left] overlaps the forward cone [-cone, +cone]?
        if b_left < -cone or b_right > cone:
            continue
        b_center = 0.5 * (b_left + b_right)
        threats.append((rng, b_left, b_right, b_center, ob))

    out["n_considered"] = len(threats)
    if not threats:
        return out

    threats.sort(key=lambda t: t[0])  # closest is the threat
    rng, b_left, b_right, b_center, ob = threats[0]

    margin = math.radians(cfg.margin_deg)
    # Pass on the side the person is on, so the detour still heads toward the goal.
    # With no person bearing, take the side that needs the smaller turn.
    if person_bearing is not None:
        pass_left = person_bearing >= b_center
    else:
        pass_left = abs(b_left + margin) <= abs(b_right - margin)

    if pass_left:
        yaw_target = b_left + margin
        side = "left"
    else:
        yaw_target = b_right - margin
        side = "right"
    yaw_target = max(-cfg.max_yaw_rad, min(cfg.max_yaw_rad, yaw_target))

    denom = max(1e-3, cfg.range_m - cfg.engage_m)
    weight = max(0.0, min(1.0, (cfg.range_m - rng) / denom))

    speed_factor = 1.0
    if rng < cfg.slow_range_m and abs(b_center) < cone:
        speed_factor = max(cfg.min_speed_factor, min(1.0, rng / max(1e-3, cfg.slow_range_m)))

    out.update(
        active=weight > 0.0,
        yaw_target_rad=float(yaw_target),
        weight=float(weight),
        speed_factor=float(speed_factor),
        threat_range_m=round(rng, 3),
        threat_bearing_deg=round(math.degrees(b_center), 1),
        pass_side=side,
    )
    return out


if __name__ == "__main__":
    # Host self-check: sign + skirt-direction geometry, no perception deps.
    cx, fx = 640.0, 924.4  # 1280-wide D435 RGB (fx = width*26/36)
    cfg = AvoidanceConfig()

    # A couch dead-ahead at 1.0 m (< slow_range), person slightly LEFT -> steer LEFT (+yaw), slow.
    obs = [{"bbox": [520, 300, 760, 520], "range_m": 1.0, "label": "couch"}]
    person_left = bearing_rad(560, cx, fx)  # person column left of image centre -> +left
    r = compute_obstacle_avoidance(obs, person_left, cx, fx, cfg)
    assert r["active"] and r["pass_side"] == "left" and r["yaw_target_rad"] > 0, r
    assert r["speed_factor"] < 1.0, r
    print("dead-ahead, person-left  ->", r)

    # Same couch, person to the RIGHT -> steer RIGHT (-yaw).
    person_right = bearing_rad(740, cx, fx)
    r2 = compute_obstacle_avoidance(obs, person_right, cx, fx, cfg)
    assert r2["pass_side"] == "right" and r2["yaw_target_rad"] < 0, r2
    print("dead-ahead, person-right ->", r2)

    # Obstacle far off to the left edge (out of the forward cone) -> ignored.
    obs_side = [{"bbox": [0, 300, 120, 520], "range_m": 1.0, "label": "table"}]
    r3 = compute_obstacle_avoidance(obs_side, person_right, cx, fx, cfg)
    assert not r3["active"] and r3["n_considered"] == 0, r3
    print("far-left obstacle        -> ignored:", r3["active"])

    # Obstacle beyond range -> ignored.
    obs_far = [{"bbox": [520, 300, 760, 520], "range_m": 3.5, "label": "couch"}]
    r4 = compute_obstacle_avoidance(obs_far, person_left, cx, fx, cfg)
    assert not r4["active"], r4
    print("beyond-range obstacle    -> ignored:", r4["active"])
    print("\nOK: obstacle-avoidance geometry self-check passed")
