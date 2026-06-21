"""Stair-climb command policy and the stair-aware front-obstacle gate.

Extracted verbatim from main.py: sensor-derived stair-climb forward command,
depth-from-bbox helpers (with person exclusion), and the front-obstacle gate
that scales the forward command near obstacles. Pure functions driven by the
per-frame ``debug_info``/``args`` passed in by the main loop.
"""
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from core.vision.depth_processor import DepthProcessor


def _depth_from_bbox(depth_img: np.ndarray, bbox: Optional[List[float]]) -> Optional[float]:
    if bbox is None:
        return None
    try:
        depth_m = DepthProcessor.foreground_depth_bimodal(
            depth_img,
            tuple(int(round(v)) for v in bbox[:4]),
            return_histogram=False,
        )
        return None if depth_m is None else float(depth_m)
    except Exception:
        return None


def _depth_from_bbox_excluding_person(
    depth_img: np.ndarray,
    stairs_bbox: Optional[List[float]],
    person_bbox: Optional[List[float]] = None,
) -> Optional[float]:
    """Measure stair depth from the depth image, masking out the person's bbox.

    Uses the 25th-percentile of valid (non-zero) pixels in the stair region
    after zeroing any overlap with the person bbox.  Falls back to the standard
    bimodal method when too few pixels remain after masking.
    """
    if stairs_bbox is None:
        return None
    try:
        h, w = depth_img.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in stairs_bbox[:4]]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        region = np.array(depth_img[y1:y2, x1:x2], dtype=np.float32)

        if person_bbox is not None:
            px1, py1, px2, py2 = [int(round(v)) for v in person_bbox[:4]]
            rel_x1 = max(0, px1 - x1);  rel_y1 = max(0, py1 - y1)
            rel_x2 = min(x2 - x1, px2 - x1); rel_y2 = min(y2 - y1, py2 - y1)
            if rel_x2 > rel_x1 and rel_y2 > rel_y1:
                region[rel_y1:rel_y2, rel_x1:rel_x2] = 0.0

        valid = region[region > 0.0]
        if len(valid) < 10:
            # Too few non-person pixels remain to read the stair edge. When a person
            # is in frame, do NOT fall back to the person-inclusive bbox depth -- that
            # returns the near person as the "stair" depth, which trips stairs_near and
            # engages the climb forward-floor on flat ground (the dog then drives
            # into/past the person). Report unknown; the main loop keeps the last
            # sensor-confirmed stair depth. With no person present, the bbox depth is
            # still a valid stair estimate.
            if person_bbox is not None:
                return None
            return _depth_from_bbox(depth_img, stairs_bbox)

        depth_mm = float(np.percentile(valid, 25))
        return (depth_mm * 0.001) if depth_mm > 0 else None
    except Exception:
        return None


def _apply_stair_command_policy(
    args,
    trans_x_cmd: float,
    rotation_cmd: float,
    debug_info: Dict[str, Any],
) -> Tuple[float, float]:
    if not bool(debug_info.get("stairs_detected", False)):
        debug_info["stairs_action_active"] = False
        return float(trans_x_cmd), float(rotation_cmd)

    # Gate: stair behavior requires the person to be actively detected -- EXCEPT for a
    # BRIEF loss while the staircase is already latched. On a brief loss we still hold the
    # forward floor (below) so the climb keeps advancing instead of stranding the policy
    # at vx=0 mid-step, but we suppress centering/recovery yaw: applying yaw amplification
    # without a fresh detection over-rotates the body and falls (the original gate intent).
    # In parkour mode steering is via delta_yaw (the predicted bearing), not this wz, so the
    # robot still aims at the last-known person while the floor keeps it climbing.
    if not bool(debug_info.get("person_detected", False)):
        lost_age = debug_info.get("lost_age_sec")
        lost_grace = debug_info.get("lost_search_timeout_sec")
        brief_loss = (
            lost_age is not None
            and lost_grace is not None
            and float(lost_age) <= float(lost_grace)
        )
        # Bounded stair finish-to-footing: stop early if we have reached flat ground/top or pitch levels off
        stair_demo = debug_info.get("stair_demo")
        if brief_loss and stair_demo and isinstance(stair_demo, dict):
            phase = stair_demo.get("phase")
            robot_data = stair_demo.get("robot", {})
            pitch_deg = robot_data.get("pitch_deg", 0.0)
            if phase in ("top_landing", "flat_follow") or abs(pitch_deg) <= 5.0:
                brief_loss = False
                debug_info["stair_finish_completed"] = True
        if not brief_loss:
            debug_info["stairs_action_active"] = False
            debug_info["stairs_gated_no_person"] = True
            return float(trans_x_cmd), float(rotation_cmd)
        debug_info["stairs_gated_no_person"] = False
        debug_info["stairs_brief_loss_floor"] = True
        rotation_cmd = 0.0

    stair_depth_m = debug_info.get("stairs_depth_m")

    # Approach slowdown: as soon as the YOLO model identifies stairs ahead, ease off the
    # throttle so the dog decelerates INTO the staircase instead of charging the base at
    # full follow speed. This runs during the approach -- before a confirmed depth or the
    # near threshold below engage the full climb policy. trans_x_cmd is the fresh follower
    # output each frame, so scaling it here does not compound across frames.
    approach_scale = float(args.stair_approach_speed_scale)
    approach_x = float(trans_x_cmd)
    if approach_x > 0.0 and approach_scale < 1.0:
        approach_x = approach_x * approach_scale
    approach_slowed = approach_x < float(trans_x_cmd)

    # Require at least one sensor-confirmed (non-latched-only) depth reading before
    # engaging the full near climb policy.  This prevents reaction to distant YOLO
    # detections where depth could not be measured -- but still slow the approach.
    if stair_depth_m is None and not bool(debug_info.get("stairs_depth_ever_confirmed", False)):
        debug_info["stairs_action_active"] = False
        debug_info["stairs_gated_no_depth"] = True
        debug_info["stairs_approach_active"] = bool(approach_slowed)
        debug_info["stairs_approach_speed_mps"] = float(approach_x)
        return float(approach_x), float(rotation_cmd)

    stairs_near = stair_depth_m is None or float(stair_depth_m) <= float(args.stair_near_distance)
    debug_info["stairs_near"] = bool(stairs_near)
    if not stairs_near:
        debug_info["stairs_action_active"] = False
        debug_info["stairs_approach_active"] = bool(approach_slowed)
        debug_info["stairs_approach_speed_mps"] = float(approach_x)
        return float(approach_x), float(rotation_cmd)

    original_x = float(trans_x_cmd)
    original_wz = float(rotation_cmd)

    # Calculate the bounded stair floor before either safety branch so telemetry remains valid
    # when the collision block forces the command to zero.
    max_forward = max(0.0, float(args.trans_x_max) * float(args.stair_speed_scale))
    forward_floor = max(0.0, float(args.stair_forward_floor))
    if max_forward > 0.0:
        forward_floor = min(forward_floor, max_forward)

    # Hard collision floor on stairs: if the smoothed gap drops below the collision floor,
    # zero the drive (no stance-lock -- a blend at speed on the slope nose-dives) so the
    # dog never climbs into the patient.
    _gap_ctrl = debug_info.get("standoff_gap_ctrl_m")
    _climb_block = (
        _gap_ctrl is not None and float(_gap_ctrl) > 1e-3
        and float(_gap_ctrl) < float(args.stair_climb_collision_floor)
    )
    if _climb_block:
        trans_x_cmd = 0.0
        debug_info["stair_follow_collision_block"] = True
    else:
        debug_info["stair_follow_collision_block"] = False
        # Forward floor while climbing: the person-follow PID collapses vx to ~0 once
        # the dog reaches its standoff at the stair base, which strands the (blind) RL
        # policy with no drive to step up. Hold a minimum forward command and cap it at
        # the stair speed limit so the climb keeps advancing instead of parking.
        trans_x_cmd = max(float(trans_x_cmd), forward_floor)
        if max_forward > 0.0 and trans_x_cmd > max_forward:
            trans_x_cmd = max_forward

    # Tame yaw on the stairs. The follower's bbox edge/size penalty amplifies the
    # centering error; on a step that becomes a +/-max yaw saw that twists the body
    # and breaks the climb. Apply a small centering deadband, the (sub-unity) stair
    # centering scale, and a lower stair-specific yaw cap.
    rotation_error_deg = debug_info.get("rotation_error_deg")
    yaw_deadband = max(0.0, float(args.stair_yaw_deadband_deg))
    if rotation_error_deg is not None and abs(float(rotation_error_deg)) <= yaw_deadband:
        rotation_cmd = 0.0
        debug_info["stairs_yaw_deadband_active"] = True
    else:
        rotation_cmd = float(rotation_cmd) * float(args.stair_centering_scale)
        debug_info["stairs_yaw_deadband_active"] = False
    stair_rot_max = max(0.0, float(args.stair_rot_max))
    if stair_rot_max > 0.0:
        rotation_cmd = float(np.clip(rotation_cmd, -stair_rot_max, stair_rot_max))

    debug_info["stairs_action_active"] = True
    debug_info["stairs_approach_active"] = False
    debug_info["stairs_forward_floor_mps"] = float(forward_floor)
    debug_info["stairs_speed_limit_mps"] = float(trans_x_cmd)
    debug_info["stairs_trans_x_before"] = original_x
    debug_info["stairs_rotation_before"] = original_wz
    return float(trans_x_cmd), float(rotation_cmd)


def _apply_front_obstacle_gate(
    args,
    trans_x_cmd: float,
    depth_img: np.ndarray,
    debug_info: Dict[str, Any],
) -> float:
    # On the stairs the stair policy owns the forward command, and the staircase
    # itself reads as a near "obstacle" in the central ROI -- gating here would
    # zero the climb's forward floor. Let the stair policy govern instead. The
    # stair_climbing_latch extends this bypass through a stairs-DETECTION dropout while
    # the dog is still physically climbing (otherwise the next riser is read as a
    # blocking wall and the climb command is zeroed -> the dog wedges on the step;
    # run_sim_20260619_134034). The latch is collision-gated upstream.
    if bool(debug_info.get("stairs_action_active", False)) or bool(debug_info.get("stair_climbing_latch", False)):
        debug_info["front_obstacle_gate_active"] = False
        debug_info["front_obstacle_skipped_on_stairs"] = True
        return float(trans_x_cmd)

    if not bool(args.obstacle_stop_enabled) or trans_x_cmd <= 0.0:
        debug_info["front_obstacle_gate_active"] = False
        return float(trans_x_cmd)

    nearest_m, roi_info = DepthProcessor.central_roi_nearest_depth(
        depth_img,
        width_ratio=args.obstacle_roi_width_ratio,
        height_ratio=args.obstacle_roi_height_ratio,
    )
    debug_info["front_obstacle_depth_m"] = nearest_m
    debug_info["front_obstacle_roi"] = roi_info.get("roi")
    debug_info["front_obstacle_valid_pixels"] = roi_info.get("valid_pixels", 0)
    if nearest_m is None:
        debug_info["front_obstacle_gate_active"] = False
        return float(trans_x_cmd)

    target_depth = debug_info.get("depth_distance_m")
    if target_depth is not None:
        try:
            if float(nearest_m) >= (float(target_depth) - float(args.obstacle_target_clearance)):
                debug_info["front_obstacle_gate_active"] = False
                debug_info["front_obstacle_reason"] = "not_closer_than_target"
                return float(trans_x_cmd)
        except Exception:
            pass

    if nearest_m > args.obstacle_slow_distance:
        debug_info["front_obstacle_gate_active"] = False
        return float(trans_x_cmd)

    original_cmd = float(trans_x_cmd)
    if nearest_m <= args.obstacle_stop_distance:
        trans_x_cmd = 0.0
        scale = 0.0
    else:
        span = max(1e-3, float(args.obstacle_slow_distance) - float(args.obstacle_stop_distance))
        scale = max(0.0, min(1.0, (float(nearest_m) - float(args.obstacle_stop_distance)) / span))
        trans_x_cmd = float(trans_x_cmd) * scale

    debug_info["front_obstacle_gate_active"] = True
    debug_info["front_obstacle_scale"] = float(scale)
    debug_info["front_obstacle_trans_x_before"] = original_cmd
    return float(trans_x_cmd)
