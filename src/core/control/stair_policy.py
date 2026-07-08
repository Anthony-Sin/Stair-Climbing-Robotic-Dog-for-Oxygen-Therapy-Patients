"""Stair-climb command policy and the stair-aware front-obstacle gate.

Extracted verbatim from main.py: sensor-derived stair-climb forward command,
depth-from-bbox helpers (with person exclusion), and the front-obstacle gate
that scales the forward command near obstacles. Pure functions driven by the
per-frame ``debug_info``/``args`` passed in by the main loop.
"""
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.vision.depth_processor import DepthProcessor

# A stair riser reads "near" in the forward ROI just like a body does; it is told apart
# by a strong row-to-row depth gradient (near riser face at the top rows, open tread/
# ground receding toward the bottom rows). Same threshold the front-obstacle gate uses.
_RISER_GRADIENT_MM_PER_ROW = 20.0


@dataclass
class DepthStairGate:
    """Result of the depth-based near-field stair gate (see evaluate_depth_stair_gate)."""
    result: Dict[str, Any]     # raw DepthStairDetector.detect() output (leading_edge etc.)
    confirmed: bool            # stair_detected AND stair_count >= min_count
    person_masked: bool        # the followed person's bbox was zeroed before detect()


def evaluate_depth_stair_gate(
    depth_img_mm: np.ndarray,
    person_bbox: Optional[Sequence[float]],
    detector: Any,
    *,
    min_count: int,
) -> DepthStairGate:
    """Run the geometric depth stair detector, fixing two field bugs BY CONSTRUCTION.

    Extracted verbatim from the main control loop so the exact code of both incident-8.3
    defects is unit-testable off-robot (the loop had zero tests over it):

      * UNITS (P2-2): ``DepthStairDetector.detect`` treats its grid as METRES (it filters
        ``0.06 < d < ~2.2`` m and derives world heights), but the D435 depth is uint16
        MILLIMETRES everywhere else in the pipeline. Feeding mm made every pixel exceed
        the range gate, the valid-row filter emptied, and the detector fired on 0 frames
        (silently dead). We convert mm -> m here.
      * PERSON FALSE-STAIR (incident 8.3): a patient standing ~0.6 m ahead fills the
        detector's central column band; their body (feet->head at ~constant forward
        distance) back-projects into a stack of rising height LEVELS the clusterer reads
        as a multi-riser staircase (observed 8 fake risers at the follow standoff),
        latching stairs_detected from frame 1 and forcing stair mode on flat ground. We
        zero the followed person's bbox in the grid BEFORE detect() so only real terrain
        drives it. (RESIDUAL, per the incident ledger: masking can leave band-edge slivers
        that still occasionally confirm; the full fix additionally gates the depth-only
        latch on recent YOLO stair evidence -- not done here.)

    The input ``depth_img_mm`` is never mutated (the ``* 0.001`` produces a fresh array).
    """
    grid = np.asarray(depth_img_mm, dtype=np.float32) * 0.001   # mm -> m (units fix)
    person_masked = False
    if person_bbox is not None and len(person_bbox) >= 4:
        h, w = grid.shape[:2]
        x1 = max(0, int(round(float(person_bbox[0]))))
        y1 = max(0, int(round(float(person_bbox[1]))))
        x2 = min(w, int(round(float(person_bbox[2]))))
        y2 = min(h, int(round(float(person_bbox[3]))))
        if x2 > x1 and y2 > y1:
            grid[y1:y2, x1:x2] = 0.0
            person_masked = True
    result = detector.detect(grid)
    confirmed = (
        bool(result.get("stair_detected", False))
        and int(result.get("stair_count", 0)) >= int(min_count)
    )
    return DepthStairGate(result=result, confirmed=confirmed, person_masked=person_masked)


def depth_stair_latch_allowed(
    *,
    depth_confirmed: bool,
    now: float,
    last_yolo_stair_ts: float,
    persist_sec: float,
) -> bool:
    """Whether a DEPTH stair confirmation may (re)latch stair mode this frame.

    Only when YOLO-World has corroborated stairs within ``persist_sec`` (the design intent:
    YOLO detects the staircase from AFAR, depth carries it at close range where YOLO blanks).
    This blocks near-floor / person-edge depth false-positives -- which confirm >=2 fake risers
    on FLAT ground -- from latching stair mode and killing plain-follow (incident 8.3 residual).
    YOLO itself latches independently; this only governs the DEPTH-only path.
    """
    if not depth_confirmed:
        return False
    return (float(now) - float(last_yolo_stair_ts)) <= float(persist_sec)


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


def _crest_reached(frame_meta: Optional[Dict[str, Any]], debug_info: Dict[str, Any]) -> bool:
    """Whether the dog has reached the crest / a level landing (climb can finish).

    Detects the crest from whichever signal is available, preferring hardware-real ones:

      * ``sensor_imu_pitch`` (rad) / ``sensor_riser_dist_ahead`` (m) -- real sensors the
        Isaac side MAY thread through the frame sidecar; the body pitch flattening (or no
        riser ahead) means the top is reached. Works on the robot.
      * ``stair_demo`` (sim GROUND-TRUTH only) -- the demo phase levelling to the landing or
        the GT body pitch flattening. Fallback for sim where the sensors are absent.

    Read from ``frame_meta`` (populated EARLY in the loop), not ``debug_info["stair_demo"]``
    which the main loop writes LATER in the same frame -- so the old debug_info read here got
    the default and the brief-loss forward floor never cancelled at the crest (incident 8.5).
    """
    fm = frame_meta if isinstance(frame_meta, dict) else {}
    # 1. Hardware sensors (preferred; work on the robot). Backward-compatible: absent => skip.
    pitch = fm.get("sensor_imu_pitch")
    if pitch is None:
        pitch = debug_info.get("sensor_imu_pitch")
    if pitch is not None:
        try:
            # sensor_imu_pitch is radians; ~5 deg ~= 0.087 rad flat-enough for the landing.
            if abs(float(pitch)) <= 0.0873:
                return True
        except (TypeError, ValueError):
            pass
    riser_ahead = fm.get("sensor_riser_dist_ahead")
    if riser_ahead is None:
        riser_ahead = debug_info.get("sensor_riser_dist_ahead")
    if riser_ahead is not None:
        try:
            # No riser within a tread ahead => crest/landing reached.
            if float(riser_ahead) > 1.0:
                return True
        except (TypeError, ValueError):
            pass
    # 2. Sim ground-truth fallback (stair_demo). Read from frame_meta (populated early).
    stair_demo = fm.get("stair_demo")
    if isinstance(stair_demo, dict):
        phase = stair_demo.get("phase")
        pitch_deg = (stair_demo.get("robot", {}) or {}).get("pitch_deg", 0.0)
        try:
            if phase in ("top_landing", "flat_follow") or abs(float(pitch_deg)) <= 5.0:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _apply_stair_command_policy(
    args,
    trans_x_cmd: float,
    rotation_cmd: float,
    debug_info: Dict[str, Any],
    frame_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    if not bool(debug_info.get("stairs_detected", False)):
        debug_info["stairs_action_active"] = False
        return float(trans_x_cmd), float(rotation_cmd)

    # Climb finished -- robot is on the flat TOP LANDING. Release stair mode even with the
    # person still in view, so the normal follow standoff re-engages on flat ground. While
    # stairs_action_active stays True the standoff is BYPASSED (incident 8.9) and the RL
    # climber's lean-on-creep keeps pushing forward with no distance regulation, so the dog
    # crept right up to the standing patient on the landing (observed GT gap 0.33 m vs the
    # 1.0 m follow target -- "collided with patient"). The robot's own top-landing phase is
    # the discriminator: it is DISTINCT from the flat APPROACH (phase "flat_follow", before
    # the stairs), so this only fires AFTER the climb, not before it. On the robot (no
    # stair_demo sidecar) this is a no-op and the sensor-crest path below still applies.
    fm = frame_meta if isinstance(frame_meta, dict) else {}
    _sd = fm.get("stair_demo")
    if isinstance(_sd, dict) and _sd.get("phase") == "top_landing":
        debug_info["stairs_action_active"] = False
        debug_info["stair_finish_completed"] = True
        debug_info["stairs_top_landing_released"] = True
        return float(trans_x_cmd), float(rotation_cmd)

    # Gate: stair behavior requires the person to be actively detected -- EXCEPT for a
    # BRIEF loss while the staircase is already latched. On a brief loss we still hold the
    # forward floor (below) so the climb keeps advancing instead of stranding the policy
    # at vx=0 mid-step, but we suppress centering/recovery yaw: applying yaw amplification
    # without a fresh detection over-rotates the body and falls (the original gate intent).
    # In parkour mode steering is via delta_yaw (the predicted bearing), not this wz, so the
    # robot still aims at the last-known person while the floor keeps it climbing.
    if not getattr(args, "stair_waypoint_test", False) and not bool(debug_info.get("person_detected", False)):
        lost_age = debug_info.get("lost_age_sec")
        lost_grace = debug_info.get("lost_search_timeout_sec")
        brief_loss = (
            lost_age is not None
            and lost_grace is not None
            and float(lost_age) <= float(lost_grace)
        )
        # Bounded stair finish-to-footing: stop early if we have reached flat ground/top or pitch
        # levels off. Detect the crest from frame_meta (populated early) / hardware sensors, NOT
        # debug_info["stair_demo"] which main.py writes LATER this frame -- the old read got the
        # default so this exit never fired and the brief-loss floor kept pushing at the crest
        # (incident 8.5). _crest_reached also works on the robot via sensor_imu_pitch.
        if brief_loss and _crest_reached(frame_meta, debug_info):
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
    if not getattr(args, "stair_waypoint_test", False) and stair_depth_m is None and not bool(debug_info.get("stairs_depth_ever_confirmed", False)):
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

    # Riser-shape test: a stair riser produces a depth profile where values INCREASE
    # from top row to bottom row (near riser face at top → ground level at bottom).
    # This is the opposite of a flat wall (uniform depth). Suppress the gate when the
    # ROI pattern looks like a riser so 1-frame latch gaps don't zero the climb command.
    try:
        roi = roi_info.get("roi")
        if roi is not None:
            # central_roi_nearest_depth returns roi = (x1, y1, x2, y2) (columns FIRST,
            # then rows), so the crop is depth_img[y1:y2, x1:x2]. The old unpack read it
            # rows-first (ry1,ry2,rx1,rx2 = roi[0..3]) and cropped [x1:y1, x2:y2] with the
            # axes swapped -> an empty/degenerate crop (e.g. depth_img[486:360, 793:662]),
            # so this whole riser-suppression path was dead. Mirror _roi_depth_row_gradient.
            rx1, ry1, rx2, ry2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            _roi_crop = depth_img[ry1:ry2, rx1:rx2]
            if _roi_crop.size > 0:
                # Use mm values directly (depth image is in mm); row-wise minimum depth.
                _row_min = np.array(
                    [_roi_crop[r, _roi_crop[r] > 0].min() if np.any(_roi_crop[r] > 0) else 0
                     for r in range(_roi_crop.shape[0])],
                    dtype=np.float32,
                )
                _valid = _row_min[_row_min > 0]
                if len(_valid) >= 4:
                    # Gradient in mm/row: positive = depth increases toward the bottom
                    # (riser face above, open space below). Threshold ~20 mm/row ≈ a
                    # visible depth gradient across a 0.15 m riser.
                    _grad = float(np.mean(np.diff(_valid)))
                    debug_info["front_obstacle_depth_gradient"] = round(_grad, 1)
                    if _grad > 20.0:
                        debug_info["front_obstacle_gate_active"] = False
                        debug_info["front_obstacle_riser_pattern"] = True
                        return float(trans_x_cmd)
    except Exception:
        pass

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


def _roi_depth_row_gradient(depth_img: np.ndarray, roi) -> Optional[float]:
    """Mean row-to-row change (mm/row) of the nearest depth down an ROI, or None.

    ``roi`` is the ``(x1, y1, x2, y2)`` tuple ``central_roi_nearest_depth`` returns
    (x = columns, y = rows), so the crop is ``depth_img[y1:y2, x1:x2]``. A POSITIVE
    result means depth increases toward the bottom rows -- the profile of a stair riser
    (near face above, open tread/ground below), not a flat body/wall.
    """
    try:
        if roi is None:
            return None
        x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        crop = depth_img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        row_min = np.array(
            [crop[r][crop[r] > 0].min() if np.any(crop[r] > 0) else 0
             for r in range(crop.shape[0])],
            dtype=np.float32,
        )
        valid = row_min[row_min > 0]
        if len(valid) < 4:
            return None
        return float(np.mean(np.diff(valid)))
    except Exception:
        return None


def _stair_loss_forward_block(args, depth_img, debug_info: Dict[str, Any]) -> bool:
    """LIVE near-field guard for the person-loss stair forward drive (returns True => block).

    The STAIR_LOSS_FLOOR path drives a modest forward floor UP the stairs when the patient
    lock is lost mid-climb, gated ONLY on the last-known patient gap -- which goes stale
    exactly when it matters: the patient stops on the step just ahead and detection drops,
    so the remembered gap still reads "far" while a body now fills the near field. This adds
    the missing check: read the nearest lower-center depth THIS frame and block the drive
    when something is close ahead that is NOT a stair riser (a body/wall). A real riser reads
    "near" too, so it is distinguished by the row-wise depth gradient and is NOT blocked
    (blocking on every riser would freeze the climb at the base).
    """
    if depth_img is None:
        # No depth frame at all this iteration -> the depth pipeline is stale, not garbage.
        # The OTHER loss guards (last-known-gap collision block + detection-age ceiling)
        # still apply, so this guard is a no-op here (matches test_no_depth_frame).
        return False
    try:
        nearest_m, roi_info = DepthProcessor.central_roi_nearest_depth(
            depth_img,
            width_ratio=float(args.obstacle_roi_width_ratio),
            height_ratio=float(args.obstacle_roi_height_ratio),
        )
    except Exception:
        # FAIL CLOSED: a depth frame EXISTS but the near-field probe threw (garbage/malformed
        # depth). We cannot rule out a body/wall close ahead, so block the blind forward drive
        # rather than driving into a possibly-close patient (safety-critical, patient-adjacent).
        debug_info["stairs_loss_nearfield_probe_error"] = True
        debug_info["stairs_loss_nearfield_block"] = True
        return True
    debug_info["stairs_loss_nearfield_depth_m"] = (
        None if nearest_m is None else round(float(nearest_m), 3))
    if nearest_m is None or float(nearest_m) > float(args.obstacle_stop_distance):
        debug_info["stairs_loss_nearfield_block"] = False
        return False
    grad = _roi_depth_row_gradient(depth_img, roi_info.get("roi"))
    if grad is not None:
        debug_info["stairs_loss_nearfield_gradient"] = round(float(grad), 1)
    is_riser = grad is not None and grad > _RISER_GRADIENT_MM_PER_ROW
    debug_info["stairs_loss_nearfield_riser"] = bool(is_riser)
    blocked = not is_riser
    debug_info["stairs_loss_nearfield_block"] = bool(blocked)
    return blocked
