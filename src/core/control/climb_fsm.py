"""Explicit climb-mode FSM replacing the 8 implicit timestamp latches in main.py.

WHY: the stair-climb mode was encoded across 8+ timestamp+bool variables scattered
across ~400 lines of main.py. Competing latches (stair_climb_latch_until,
_climbing_persist_until, stair_climb_committed, _stair_approach_commit, _flat_loss_glide,
_stairs_seen_ts, last_on_stairs_ts, stop_ramp_active) could simultaneously be active in
conflicting states and adding any new behaviour required auditing all 8 conditions.

This class owns ALL of those variables in one place. main.py calls
``fsm.update(...)`` once per frame and reads ``fsm.state`` (a named string),
``fsm.stair_climb_committed``, ``fsm.stair_climb_latch_until``, etc.

BEHAVIOUR: identical to the old scattered logic -- this is a PURE refactor.
No gates, thresholds, or timing constants are changed. The FSM is a container
that makes state visible (debug_info["fsm_state"]) and lets new states be added
as a single enum value + transition rule.

States (mutually exclusive, the "dispatch mode"):
  FLAT_FOLLOW         normal flat-ground following (motion_allowed path)
  STAIR_APPROACH      stairs detected far, dog approaching
  STAIR_NEAR          stairs_action_active OR close-dropout; climb gait forced
  COMMITTED_CLIMB     stair_climb_committed; forward drive bypasses follow gates
  STAIR_LOSS_FLOOR    person lost mid-climb (stair latch active); modest forward floor
  STAIR_APPROACH_COMMIT stair seen + person lost + riser in depth range; creep straight
  FLAT_LOSS_GLIDE     person briefly lost on flat, clear path ahead, AND was last seen
                      roughly straight ahead; gentle straight glide. Yields to the
                      follower's recovery yaw when the patient turned off-axis.
  STOP                catch-all; controller.stop()
"""

import time
from typing import Optional

# Flat-loss glide only engages when the patient was last seen within this bearing of
# dead-ahead. Beyond it the patient clearly turned, so the straight glide is wrong and
# would suppress the recovery yaw -- fall through to the turn-to-re-acquire path instead.
GLIDE_MAX_BEARING_DEG = 8.0


class ClimbFSM:
    """All climb-mode state in one place.

    Call ``update()`` once per frame with the per-frame sensor signals, then read
    the public properties to get the dispatch mode and any command overrides.
    """

    # All valid state names (used in debug_info["fsm_state"])
    STATES = (
        "FLAT_FOLLOW",
        "STAIR_APPROACH",
        "STAIR_NEAR",
        "COMMITTED_CLIMB",
        "STAIR_LOSS_FLOOR",
        "STAIR_APPROACH_COMMIT",
        "FLAT_LOSS_GLIDE",
        "STOP",
    )

    def __init__(self, args):
        self._args = args
        self._t = time.perf_counter

        # --- replicated latch variables (were spread across main.py) ---
        self.stop_ramp_vx: float = 0.0
        self.stop_ramp_active: bool = False
        self.stop_ramp_last_ts: float = self._t()
        self.last_on_stairs_ts: float = 0.0
        self.stair_climb_committed: bool = False
        self.stair_climb_commit_ts: float = 0.0
        self.stair_climb_latch_until: float = 0.0
        self._climbing_persist_until: float = 0.0
        self._stairs_seen_ts: float = 0.0
        self.last_person_gap_m: Optional[float] = None

        # --- computed this frame (read by main.py dispatch) ---
        self.state: str = "FLAT_FOLLOW"
        self.stairs_now: bool = False
        self.stairs_recent: bool = False
        self.stair_close_dropout: bool = False
        self.stair_approach_commit: bool = False
        self.flat_loss_glide: bool = False
        self.climbing_latched: bool = False
        self.committed_stair_floor: float = 0.0

    # ------------------------------------------------------------------
    # Main update: call once per frame, pass in all sensor signals
    # ------------------------------------------------------------------

    def update(
        self,
        current_time: float,
        *,
        stairs_detected: bool,
        stairs_action_active: bool,
        stairs_depth_m: Optional[float],
        last_stairs_depth_m: Optional[float],
        stairs_depth_ever_confirmed: bool,
        person_detected: bool,
        depth_distance_m: Optional[float],
        front_near_m: Optional[float],
        standoff_gap_ctrl_m: Optional[float],
        lost_age_sec: Optional[float],
        motion_allowed: bool,
        last_seen_bearing_deg: Optional[float] = None,
        recovery_yaw_active: bool = False,
        stair_climb_committed_in: Optional[bool] = None,  # external override (unused normally)
    ) -> dict:
        """Update all latch variables and derive the new state.

        Returns a dict of debug fields to merge into debug_info.
        """
        args = self._args

        # --- remember last patient gap while patient is visible ---
        if person_detected and depth_distance_m is not None and float(depth_distance_m) > 1e-3:
            self.last_person_gap_m = float(depth_distance_m)

        # --- _stairs_seen_ts: last time YOLO saw the stairs ---
        if stairs_detected:
            self._stairs_seen_ts = current_time
        stairs_seen_recent = (current_time - self._stairs_seen_ts) < float(args.stair_seen_persist_sec)

        # --- genuine stairs (depth or YOLO action active) ---
        genuine_stairs = bool(stairs_action_active)

        # --- on-stairs latch ---
        if genuine_stairs:
            self.last_on_stairs_ts = current_time
        stairs_recent = (current_time - self.last_on_stairs_ts) < float(args.stair_hold_suppress_sec)
        stairs_now = genuine_stairs or stairs_recent

        # --- climbing persistence latch (the wedge fix) ---
        near_riser = (front_near_m is not None and float(front_near_m) <= float(args.obstacle_slow_distance))
        patient_ahead = (
            standoff_gap_ctrl_m is not None
            and float(standoff_gap_ctrl_m) > float(args.stair_target_distance) + 0.5
        )
        at_riser = (front_near_m is not None and float(front_near_m) <= float(args.stair_depth_engage_distance))
        depth_climb_engage = bool(stairs_seen_recent and at_riser)

        if (genuine_stairs or depth_climb_engage
                or (current_time < self._climbing_persist_until and (near_riser or patient_ahead))):
            self._climbing_persist_until = current_time + 6.0
        climbing_latched = current_time < self._climbing_persist_until

        # --- climb-gait latch (heading hold during dropout) ---
        if bool(getattr(args, "stair_climb_latch", True)):
            if stairs_action_active:
                self.stair_climb_latch_until = current_time + float(args.stair_climb_max_sec)

        # --- close-range YOLO dropout fix ---
        stair_close_dropout = (
            stairs_recent and not genuine_stairs
            and last_stairs_depth_m is not None
            and float(last_stairs_depth_m) <= float(args.stair_near_distance)
        )

        # --- committed floor value ---
        committed_stair_floor = max(0.0, min(
            max(float(args.stair_forward_floor), float(args.stair_loss_forward_floor)),
            float(args.trans_x_max) * float(args.stair_speed_scale),
        ))

        # --- stair_climb_committed ---
        if bool(getattr(args, "stair_climb_commit", True)):
            near_conf_stairs = (
                stairs_depth_ever_confirmed
                and (
                    (stairs_depth_m is not None
                     and float(stairs_depth_m) <= float(args.stair_climb_commit_distance))
                    or (last_stairs_depth_m is not None
                        and float(last_stairs_depth_m) <= float(args.stair_climb_commit_distance))
                )
            )
            if self.stair_climb_committed and (
                    current_time - self.stair_climb_commit_ts) > float(args.stair_climb_max_sec):
                self.stair_climb_committed = False
            if near_conf_stairs and stairs_now and not self.stair_climb_committed:
                self.stair_climb_committed = True
                self.stair_climb_commit_ts = current_time

        # --- stair approach commit ---
        stair_approach_commit = (
            stairs_seen_recent
            and not stairs_now
            and front_near_m is not None
            and float(args.stair_depth_engage_distance) < float(front_near_m) <= 1.5
        )

        # --- flat-loss glide ---
        # The straight glide is ONLY correct when the patient was lost heading roughly
        # straight ahead (a brief YOLO blink on the forward axis). When the patient TURNED
        # off-axis (a zigzag apex), gliding straight drives away from them AND -- because the
        # dispatch hard-zeroes yaw during the glide -- it actively SUPPRESSES the recovery
        # turn. So yield to the recovery yaw: skip the glide when the follower is already
        # turning to re-acquire (recovery_yaw_active) or when the last-seen bearing was
        # clearly off-centre. last_seen_bearing_deg is None pre-loss / when unknown -> glide
        # stays available for the straight-loss case it was built for.
        _glide_heading_ok = (
            last_seen_bearing_deg is None
            or abs(float(last_seen_bearing_deg)) <= GLIDE_MAX_BEARING_DEG
        )
        flat_loss_glide = (
            not person_detected
            and not stairs_now
            and not stair_approach_commit
            and lost_age_sec is not None
            and float(lost_age_sec) <= float(getattr(args, "follow_loss_glide_sec", 4.0))
            and front_near_m is not None and float(front_near_m) > 0.9
            and _glide_heading_ok
            and not recovery_yaw_active
        )

        # --- derive FSM state ---
        if self.stair_climb_committed and motion_allowed:
            state = "COMMITTED_CLIMB"
        elif motion_allowed and not stair_approach_commit and not flat_loss_glide:
            if stairs_now:
                state = "STAIR_NEAR"
            elif stairs_detected:
                state = "STAIR_APPROACH"
            else:
                state = "FLAT_FOLLOW"
        elif (not person_detected and climbing_latched
              and not self.stair_climb_committed and not motion_allowed):
            state = "STAIR_LOSS_FLOOR"
        elif stair_approach_commit:
            state = "STAIR_APPROACH_COMMIT"
        elif flat_loss_glide:
            state = "FLAT_LOSS_GLIDE"
        else:
            state = "STOP"

        # --- write back to instance for main.py to read ---
        self.state = state
        self.stairs_now = stairs_now
        self.stairs_recent = stairs_recent
        self.stair_close_dropout = stair_close_dropout
        self.stair_approach_commit = stair_approach_commit
        self.flat_loss_glide = flat_loss_glide
        self.climbing_latched = climbing_latched
        self.committed_stair_floor = committed_stair_floor
        self._stairs_seen_recent = stairs_seen_recent  # needed by dispatch

        return {
            "fsm_state": state,
            "stair_climb_committed": bool(self.stair_climb_committed),
            "stair_climbing_latch": bool(climbing_latched),
            "stairs_hold_suppress_latched": bool(stairs_recent and not genuine_stairs),
            "stair_close_dropout": bool(stair_close_dropout),
            "stair_approach_commit_eligible": bool(stair_approach_commit),
            "flat_loss_glide_eligible": bool(flat_loss_glide),
            "depth_climb_engage": bool(depth_climb_engage),
        }
