"""Flat-ground person-follow command shaping.

Extracted verbatim from main.py: standoff hysteresis + floor-creep pacing
(``_apply_follow_standoff_policy``), reverse-command suppression, and the
carrot-heading update. Pure functions driven by ``args``/``state``/``debug_info``.
"""
import time
import numpy as np
from typing import Any, Dict, List, Optional


def _apply_no_reverse_follow_policy(
    args,
    trans_x_cmd: float,
    debug_info: Dict[str, Any],
    *,
    source: str,
) -> float:
    if trans_x_cmd >= 0.0:
        debug_info.setdefault("reverse_follow_suppressed", False)
        return float(trans_x_cmd)

    debug_info["reverse_follow_suppressed"] = True
    debug_info["reverse_follow_source"] = source
    debug_info["reverse_follow_cmd_before_suppression"] = float(trans_x_cmd)
    debug_info["reverse_follow_reason"] = (
        "hold_position_track_target_until_forward_gap_opens"
    )
    return 0.0


def _apply_follow_standoff_policy(
    args,
    trans_x_cmd: float,
    gap_m: Optional[float],
    leader_speed_mps: float,
    is_walking: bool,
    debug_info: Dict[str, Any],
    state: Dict[str, Any],
) -> float:
    # 0. Gap smoothing (CRITICAL). This MUST run before the stair early-return: the stair
    # collision floor consumes standoff_gap_ctrl_m. The old ordering returned first and silently
    # disabled that safety gate for the whole climb.
    # depth_distance_m is bimodal-noisy: single-frame jumps of
    #    ~0.4<->1.0<->2.0<->0.0 m are routine even at rest. We threshold the gap for BOTH the
    #    too-close stance-lock (downstream) AND the catch-up command (below), so a single spurious
    #    reading would either freeze the creep (-> gap opens -> catch-up -> a ~2 m/s run that
    #    overshoots to within ~0.5 m of the patient) or fire a phantom catch-up directly. Median-
    #    filter the last few VALID readings (0 / None = "no lock", not a distance) and make every
    #    go/hold/catch-up decision on the smoothed value. Until the filter has >=3 samples we do
    #    NOT make the aggressive (freeze / catch-up) calls -- the startup depth transient is exactly
    #    when the noise is worst and the robot is settling from the drop.
    gap_hist = state.setdefault("gap_hist", [])
    if gap_m is not None and float(gap_m) > 1e-3:
        gap_hist.append(float(gap_m))
        if len(gap_hist) > 5:
            del gap_hist[0]
    gap_ctrl = float(np.median(gap_hist)) if len(gap_hist) >= 3 else None
    debug_info["standoff_gap_raw_m"] = None if gap_m is None else float(gap_m)
    debug_info["standoff_gap_ctrl_m"] = gap_ctrl

    # 1. Standoff calculation (speed adaptive). Use the follower's live target distance rather
    # than the flat-ground CLI default. The main loop deliberately switches that live target to
    # --stair-target-distance as soon as stairs are seen; continuing to use args.target_distance
    # here kept the hold boundary at the short flat-ground gap and let the dog catch the patient
    # before the first riser.
    base_standoff = float(debug_info.get("target_distance", args.target_distance))

    # Dynamic stair tightening: when on stairs and the gap has drifted beyond the base standoff,
    # shrink the effective standoff so the robot chases harder the further the patient gets.
    # Without this the old fixed 0.9 m stair standoff let the gap balloon to 1.5 m+ on approach
    # (run_20260624_075819_043: standoff=0.9 → gap grew 0.9→1.5 m → person lost before climb).
    if bool(debug_info.get("stair_close_active", False)) and gap_ctrl is not None:
        _chase_gain = float(getattr(args, "stair_standoff_chase_gain", 0.35))
        _stair_min = float(getattr(args, "stair_target_distance_min", 0.28))
        if _chase_gain > 0.0 and gap_ctrl > base_standoff:
            _tightened = base_standoff - (gap_ctrl - base_standoff) * _chase_gain
            base_standoff = max(_stair_min, _tightened)
        debug_info["stair_standoff_dynamic_m"] = round(base_standoff, 3)

    # leader_speed_mps is depth-derived and spikes to
    #    absurd values when the gap reading jumps (observed up to ~40 m/s on lock flicker), so clamp
    #    it to a sane walking range before it widens the standoff -- otherwise a single bad frame
    #    pins the standoff at its cap and jolts the go/hold decision.
    leader_speed_clamped = float(np.clip(float(leader_speed_mps), 0.0, 1.0))
    standoff = base_standoff + args.follow_standoff_speed_gain * leader_speed_clamped
    standoff = min(1.5, standoff)

    lower_bound = standoff + args.follow_standoff_band_in
    upper_bound = standoff + args.follow_standoff_band_out
    debug_info["standoff_target_m"] = float(standoff)
    debug_info["standoff_lower_bound_m"] = float(lower_bound)
    debug_info["standoff_upper_bound_m"] = float(upper_bound)

    # On stairs, bypass only the go/hold/pace shaping to avoid stalls. Keep the smoothed gap and
    # the correct stair standoff telemetry available to the collision and hold gates.
    if bool(debug_info.get("stairs_action_active", False)):
        debug_info["follow_standoff_gate_active"] = False
        debug_info["follow_standoff_skipped_on_stairs"] = True
        return float(trans_x_cmd)

    if gap_m is None:
        debug_info["follow_standoff_gate_active"] = False
        return float(trans_x_cmd)

    # Settle grace: for the first follow_settle_grace_sec of following, do NOT let the too-close
    # stance-lock fire (flag consumed in the main loop). Startup depth/detection reads a sustained
    # close gap that the median can't reject; freezing then opens the gap and forces a catch-up run.
    _now = time.perf_counter()
    if "first_ctrl_ts" not in state:
        state["first_ctrl_ts"] = _now
    warmup_active = (_now - state["first_ctrl_ts"]) < float(getattr(args, "follow_settle_grace_sec", 2.0))
    debug_info["standoff_warmup_active"] = bool(warmup_active)

    # 2. Hysteretic Go/Hold decision bounds -- on the SMOOTHED gap. Until the filter is warm
    #    (gap_ctrl is None) leave go_state on its current (hysteretic) value rather than reacting to
    #    a raw startup spike.
    if gap_ctrl is not None:
        if gap_ctrl < lower_bound:
            state["go_state"] = False
        elif gap_ctrl > upper_bound:
            state["go_state"] = True

    # 3. Gait gate override: hold when a stopped patient is at/near standoff, but
    #    never suppress approach when the gap is well past the GO threshold. (is_walking is noisy in
    #    sim, but with the lean-on-creep gate a spurious hold here only toggles creep<->creep -- the
    #    actual stance-lock is the too-close gap decision in the main loop, not go_state.)
    if args.follow_gait_gate and not is_walking and gap_ctrl is not None and gap_ctrl <= upper_bound:
        state["go_state"] = False
        debug_info["follow_gait_gate_triggered"] = True
    else:
        debug_info["follow_gait_gate_triggered"] = False
    debug_info["follow_gait_gate_far_override"] = bool(gap_ctrl is not None and gap_ctrl > upper_bound)
        
    original_cmd = float(trans_x_cmd)
    
    # Force hold state override
    if not state["go_state"]:
        trans_x_cmd = 0.0
        
    # 4. Pacing on vx -- LEAN ON THE FLOOR-CREEP (do NOT command catch-up bursts in normal follow).
    #    Verified from the fall-diag logs: with vx=0 and no stance-lock the frozen policy still
    #    FLOOR-CREEPS forward at ~0.5 m/s, which already matches the ~0.5 m/s patient. But ANY
    #    commanded forward advance gets over-run by the policy into a ~1.2 m/s "run" that overshoots
    #    the person, loses the lock at close range, and falls. So:
    #      * catch-up (go_state True, gap > follow_pace_distance): the leader has genuinely walked
    #        far ahead -- command at least the floor to close the gap. The brief run happens in open
    #        space (no overlap risk) and the stop-ramp bleeds it as the gap closes back in.
    #      * follow (go_state True, gap <= follow_pace_distance): command ZERO and let the intrinsic
    #        creep hold the gap (heading still steers toward the person). No burst -> no run.
    #      * hold (go_state False): command zero. Whether this becomes a stance-lock is decided
    #        downstream by the GAP (too-close), NOT here -- see hold gating in the main loop.
    pace_cap_active = False
    pace_hold_active = False

    trot_kp = float(getattr(args, "follow_trot_speed_kp", 0.0))
    if trot_kp > 0.0 and state["go_state"] and gap_ctrl is not None:
        # Dynamic pace-matching follow (creepless walker, e.g. PGTT). PGTT does NOT
        # self-creep on a zero command, so the old lean-on-creep STOPPED the dog whenever
        # the person was within pace-distance (stop/start cycling, run_20260620_172239).
        # A pure proportional trot fixed the stopping but, being P-only, trailed the
        # MOVING person by a steady-state lag (~1.2-1.5 m at target 0.45 m). So FEED
        # FORWARD the leader's measured speed -> the dog matches the person's pace and the
        # proportional term then only has to close to the standoff, so the gap settles at
        # ~standoff instead of far behind. Eases to 0 when the person stops and the gap is
        # closed; clamped to the speed limit. Parkour keeps the creep via --follow-trot-speed-kp 0.
        trot = float(leader_speed_clamped) + trot_kp * (float(gap_ctrl) - float(standoff))
        trans_x_cmd = float(np.clip(trot, 0.0, float(args.trans_x_max)))
        state["pace_state"] = "trot"
        pace_cap_active = True
    elif state["go_state"] and gap_ctrl is not None and gap_ctrl > args.follow_pace_distance:
        # Catch-up (parkour creep mode, trot_kp=0): leader GENUINELY far ahead -> command
        # the floor so the policy actually moves; the intrinsic ~0.5 m/s creep holds otherwise.
        state["pace_state"] = "advance"
        trans_x_cmd = max(float(trans_x_cmd), float(args.follow_pace_floor_speed))
        pace_cap_active = True
    else:
        # Creep mode (trot_kp=0, e.g. parkour) / too-close / hold: lean on the policy's
        # intrinsic creep; never command forward. trans_x_cmd is already zero in the hold case.
        state["pace_state"] = "creep"
        trans_x_cmd = 0.0
        
    # Populate debug info
    debug_info["fused_gap_m"] = float(gap_m)
    debug_info["follow_standoff_gate_active"] = not state["go_state"]
    debug_info["pace_state"] = state["pace_state"]
    debug_info["pace_cap_active"] = pace_cap_active
    debug_info["pace_hold_active"] = pace_hold_active
    debug_info["follow_standoff_trans_x_before"] = original_cmd

    return float(trans_x_cmd)


def _update_carrot_heading(
    args,
    trail: List[List[float]],
    gap_m: float,
    bearing_rad: float,
    leader_speed_mps: float,
    standoff_m: float,
    ego_dx: float,
    ego_dyaw: float,
) -> Optional[float]:
    """Body-frame breadcrumb follower (Method 1, opt-in via --carrot-follow).

    Maintains ``trail`` -- a FIFO of the person's position in the robot's CURRENT body frame
    (x forward, y left), newest last -- and returns the heading (rad, policy yaw convention where
    +left, matching ``-radians(rotation_error_deg)``) to the trail point one ``standoff`` BEHIND the
    newest sample. Steering at that point makes the robot follow the person's PATH rather than
    pointing straight at them, so it does not cut the inside of a turn toward them. Returns None to
    fall back to the direct bearing when the leader is too slow or the trail is shorter than the
    standoff (a stationary person yields a degenerate trail).

    Registration caveats (documented, opt-in v1): the robot's body yaw rate is NOT observable
    controller-side (the frozen policy self-steers and ignores the wz command), so pass
    ``ego_dyaw=0.0`` unless a real estimate exists; and the policy over-runs the forward command, so
    ``ego_dx`` (built from the commanded speed) under-estimates true travel. The trail is kept short
    (arc-length capped) and the heading is slew-limited downstream, which bounds these errors.
    """
    # 1. Re-register stored points into the current body frame (undo this frame's ego-motion).
    if trail:
        c = float(np.cos(-ego_dyaw))
        s = float(np.sin(-ego_dyaw))
        for p in trail:
            x = p[0] - ego_dx
            y = p[1]
            p[0] = c * x - s * y
            p[1] = s * x + c * y
    # 2. Append the current detection.
    trail.append([float(gap_m) * float(np.cos(bearing_rad)),
                  float(gap_m) * float(np.sin(bearing_rad))])
    # 3. Cap the trail by arc length (drop the oldest beyond carrot_trail_len_m).
    max_len = max(0.1, float(args.carrot_trail_len_m))
    acc = 0.0
    cut = 0
    for i in range(len(trail) - 1, 0, -1):
        acc += float(np.hypot(trail[i][0] - trail[i - 1][0], trail[i][1] - trail[i - 1][1]))
        if acc > max_len:
            cut = i
            break
    if cut > 0:
        del trail[:cut]
    # 4. Quality gate: need an actual path to follow.
    if leader_speed_mps < float(args.carrot_min_leader_speed) or len(trail) < 2:
        return None
    # 5. Walk back one standoff along the trail and interpolate the carrot point.
    cfg_standoff = float(args.carrot_standoff_m)
    target = cfg_standoff if cfg_standoff > 0.0 else float(standoff_m)
    if target <= 0.0:
        return None
    acc = 0.0
    for i in range(len(trail) - 1, 0, -1):
        seg = float(np.hypot(trail[i][0] - trail[i - 1][0], trail[i][1] - trail[i - 1][1]))
        if acc + seg >= target:
            t = (target - acc) / seg if seg > 1e-6 else 0.0
            cx = trail[i][0] + t * (trail[i - 1][0] - trail[i][0])
            cy = trail[i][1] + t * (trail[i - 1][1] - trail[i][1])
            return float(np.arctan2(cy, cx))
        acc += seg
    return None  # trail shorter than the standoff -> fall back to the direct bearing
