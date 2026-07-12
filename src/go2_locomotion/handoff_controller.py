"""Task 2c: the WALK(PGTT) <-> CLIMB state machine (``HandoffController``).

Owns the stall detector, the depth stair detector, and one climber instance, and
runs the walk<->climb transition every locomotion step (see the module docstring
of ``pgtt_stair_handoff`` for the full Task-2 contract). Split out of
``pgtt_stair_handoff`` (which now re-exports ``HandoffController``) so the FSM is
its own module; the config + detectors it depends on live in sibling modules.

TIMING IS CALLER-DEFINED ACCUMULATED TIME (incident 8.6). Every window/timeout is
measured against ``self._t`` -- a clock advanced by ``self._t += dt`` from the CALLER's
``dt`` -- NOT the wall-clock ``now`` argument. Mixing the two (``_commit_until = now +
window`` while the stall detector accumulated sim ``dt``) meant a window meant one
duration in the ~8.5x-slower headless sim and another on the robot. ``now`` is still
accepted for signature/log compatibility but is NOT used for any timing decision.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np

from go2_locomotion.closed_loop_stair_climber import ClosedLoopStairClimber
from go2_locomotion.handoff_config import HandoffConfig
from go2_locomotion.handoff_detectors import DepthStairDetector, StallDetector

try:
    from sim_logging_utils import log_event
except Exception:  # pragma: no cover - logging helper is optional
    def log_event(logger, level, action, message, **fields):
        if logger is not None:
            logger.log(level, "%s %s", message, fields)


def _wrap_pi(a: float) -> float:
    """Wrap an angle (rad) to [-pi, pi] so a heading error is the shortest signed turn."""
    return float((float(a) + math.pi) % (2.0 * math.pi) - math.pi)


def stair_engage_person_ghost_veto(
    *,
    leading_edge_distance_m: Optional[float],
    person_gap_m: Optional[float],
    window_m: float,
) -> bool:
    """True => veto a stair ENGAGE this frame: the depth detector's leading-edge reading is
    close enough to the (sim-GT) followed-person distance that the "staircase" it is reading
    is actually the standing/close PATIENT, not real stairs (incident 8.3 class: a body
    back-projects into a stack of fake risers -- see ``handoff_engage`` callers below).

    Pure / host-safe (no Isaac imports, unit-testable in isolation): both inputs are plain
    floats the caller (``HandoffController.update``) already has in scope -- ``det.get(
    "leading_edge_distance")`` (the depth detector's own reading, already read for
    ``near_enough`` a few lines above each call site) and ``person_gap_m`` (an explicit
    parameter of ``update`` since its introduction -- the sim GT planar dog<->patient
    distance, threaded from ``isaac_env._run_pgtt_handoff``'s live ``_patient_state``; ``None``
    on real hardware, where this veto is therefore always a no-op -- incident 8.5: never
    re-derived from a downstream read).

    ``None`` for either input (no GT available, e.g. real hardware, or no leading-edge
    reading this frame) means the comparison cannot be made at all -- returns False (no veto)
    rather than guessing; this mirrors how the REST of this module's GT-only safety nets
    (``stairs_ahead_gt``, ``_gt_lead_m`` in core/main.py) are sim-only and inert elsewhere.

    ``window_m`` default (``HandoffConfig.ghost_engage_gap_window_m``, see that field's
    docstring for the full numeric derivation) is 0.5 m, NOT the 0.35 m originally suggested
    by the task brief -- re-measured against run_sim_20260712_013638_835's actual ghost engage
    (leading_edge_m=0.648 vs a GT patient gap stable at 1.08-1.11 m -- diff ~0.448 m, which
    0.35 m would have missed) and confirmed to still leave ~0.9 m of margin below that same
    run's real engage (diff ~1.25 m).
    """
    if leading_edge_distance_m is None or person_gap_m is None:
        return False
    return abs(float(leading_edge_distance_m) - float(person_gap_m)) <= float(window_m)


def stair_entry_lead_ok(
    *,
    patient_lead_m: Optional[float],
    min_lead_m: float,
) -> bool:
    """True => the patient has enough of a head start onto the staircase for the dog to
    COMMIT (ENGAGE) the climb this frame -- the S1 stair-entry head-start gate.

    Pure / host-safe, same shape as ``stair_engage_person_ghost_veto`` above: the caller
    (``HandoffController.update``) already has ``patient_lead_m`` in scope as an explicit
    parameter (isaac_env.py's ``_run_pgtt_handoff`` threads ``_patient_state.x - base_x`` --
    the SAME along-path measure isaac_env's ``PATIENT_HARD_WAIT_LEAD_M`` pacing check uses,
    so the two constants describe one consistent "lead" axis; see
    ``HandoffConfig.stair_entry_min_lead_m``'s docstring for the full deadlock-interval math
    against that constant).

    ``patient_lead_m`` ``None`` (no sim GT -- real hardware, or no patient-tracking sidecar)
    means the comparison cannot be made at all -- returns True (gate is a no-op) rather than
    guessing, mirroring how the ghost-veto's GT-only safety net is inert off-sim.
    """
    if patient_lead_m is None:
        return True
    return float(patient_lead_m) >= float(min_lead_m)


class HandoffController:
    """Task 2c: WALK(PGTT) <-> CLIMB(ClosedLoopStairClimber) state machine.

    Owns the stall detector, the depth stair detector, and one climber instance.
    ``update`` is called every locomotion step; it returns whether the climber should
    drive this step (and its joint targets) or whether PGTT should keep walking.
    """

    def __init__(
        self,
        cfg: HandoffConfig,
        pgtt_policy: Any,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.pgtt = pgtt_policy
        self.logger = logger
        self.stall = StallDetector(cfg)
        self.detector = DepthStairDetector(cfg)
        self.climber = ClosedLoopStairClimber()
        self.state = "walk"
        self._climb_start_z: Optional[float] = None
        self._climb_t0 = 0.0
        self._climb_elapsed = 0.0   # SIM time spent in the current climb (dt-accumulated)
        self._elapsed_dt = 0.0      # total accumulated sim-time; guards stair_commit arm delay
        self._cooldown_until = 0.0
        self._commit_until = 0.0
        self._committing = False
        self._last_detect_ts = -1.0
        self._det = dict(self.detector.last)
        self._climbs_done = 0
        self._last_reason = ""
        # --- top-of-stairs egress state ---
        self._top_clear_sec = 0.0          # debounce accumulator for "no stairs ahead"
        self._egress = False               # in the post-crest egress sub-phase
        self._egress_sec = 0.0             # time spent in egress
        self._egress_x0: Optional[float] = None   # base_x at the crest (world +x == up-stairs)
        self._egress_travel = 0.0          # |base_x - _egress_x0| (for telemetry)
        self._egress_vx_floor: Optional[float] = None  # last person-gated egress floor (telemetry)
        self._crest_logged = False
        # --- vertical-progress watchdog state ---
        self._climb_progress_z: Optional[float] = None  # highest base_z reached this climb
        self._climb_stall_sec = 0.0        # time since the body last gained height
        self._climb_stall_retries = 0      # mid-incline stall/hard-cap HOLDs (never hand the incline to PGTT)
        # Heading captured the instant stair-commit begins; the commit heading-lock holds
        # THIS (not absolute yaw 0, which is "up +x" in sim but "power-on pose" on the robot).
        self._commit_yaw0: Optional[float] = None
        # --- post-climb re-acquisition state ---
        self._post_climb_reacquire: bool = False  # True after top_egress_done until yaw realigned
        # --- S1 stair-entry head-start gate: rate-limit the veto log (8.8) -----------------
        self._lead_gate_last_log_t: float = -1e9   # self._elapsed_dt at the last veto log

    def reset(self) -> None:
        self.stall.reset()
        self.detector.reset()
        self.climber.reset()
        self.state = "walk"
        self._climb_start_z = None
        self._climb_elapsed = 0.0
        self._elapsed_dt = 0.0
        self._cooldown_until = 0.0
        self._commit_until = 0.0
        self._committing = False
        self._last_detect_ts = -1.0
        self._det = dict(self.detector.last)
        self._top_clear_sec = 0.0
        self._egress = False
        self._egress_sec = 0.0
        self._egress_x0 = None
        self._egress_travel = 0.0
        self._egress_vx_floor = None
        self._crest_logged = False
        self._climb_progress_z = None
        self._climb_stall_sec = 0.0
        self._climb_stall_retries = 0
        self._commit_yaw0 = None
        self._post_climb_reacquire = False
        self._lead_gate_last_log_t = -1e9

    def update(
        self,
        *,
        now: float,
        dt: float,
        go2: Any,
        depth_hw: Any,
        cmd_vx: float,
        stairs_action_active: bool,
        base_z: Optional[float] = None,   # None == no vertical odometry (real w/o base-height)
        body_speed: Optional[float] = None,
        roll: float,
        pitch: float,
        roll_rate: float,
        pitch_rate: float,
        height_above_step: Optional[float],
        foot_contacts: Optional[np.ndarray] = None,
        person_detected: bool = True,
        yaw: float = 0.0,
        y_lateral: float = 0.0,
        body_fwd: Optional[float] = None,
        riser_dist_ahead: Optional[float] = None,
        base_x: float = 0.0,
        person_gap_m: Optional[float] = None,
        stairs_ahead_gt: Optional[bool] = None,
        forward_goal_dist_m: Optional[float] = None,
        caller_hold: bool = False,
        patient_lead_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        # ``caller_hold`` (incident 8.15 / F1 extension, run_sim_20260711_155123_326): the
        # perception controller's stance-hold decision, passed as an explicit argument (8.5)
        # from the SAME capture the F1 clamp uses (isaac_env.py ~L2160 ``_motion_hold_requested
        # = bool(hold)``, applied to PGTT at ~L2342). The F1 clamp closed the WALK path, but
        # this FSM ran BEFORE it and could still ENGAGE a new climb whose hot-swap branch
        # returns early (isaac_env.py ~L2243 / ~L2300) -- bypassing the clamp entirely. That
        # run: with the caller asserting fsm=STOP / hold=True / vx=0 continuously, the walk-
        # state commit vx_floor kept the stall detector armed, the depth detector read the
        # STANDING PATIENT (x=8.42, gap ~0.8 m) as an 8-step staircase (level_heights 0.285-
        # 0.669 m -- knee-to-hip; incident 8.3's person-as-risers signature), a "wedge_stall"
        # climb engaged at 20:00:56 and the dog was driven up the patient's legs (pitch +85.7
        # deg) and flipped backward at x=8.63, roll 180 deg. RULE: while the caller holds,
        # this FSM must not START a climb nor push any walk-state forward floor; every
        # legitimate engage (stair-base approach_room, riser wedge during follow, the
        # STAIR_LOSS_FLOOR blind-carry) happens with the caller commanding motion
        # (hold=False). An ONGOING climb (state=="climb") is deliberately NOT clamped: a
        # hold arriving mid-incline must not strand the climber (incidents 8.9 / 8.15 second
        # correction -- blind-carry momentum is itself the stability strategy there).
        caller_hold = bool(caller_hold)
        # Advance the caller-defined accumulated clock. ALL timing below is measured against
        # THIS (self._elapsed_dt), never the wall-clock ``now`` (incident 8.6).
        self._elapsed_dt += max(0.0, float(dt))
        _t = self._elapsed_dt
        # Throttled depth detection (the parkour cam only refreshes ~10 Hz).
        if depth_hw is not None and (_t - self._last_detect_ts) >= float(self.cfg.stair_detect_period_sec):
            self._det = self.detector.detect(depth_hw)
            self._last_detect_ts = _t
        det = self._det

        # --- Stair-commit ("person walked up out of frame -> keep going up") ----------
        # When stairs are detected ahead and the person is LOST, commit to driving
        # straight up the staircase: hold heading (yaw -> 0, the +x stair axis) and steer
        # back toward the centerline (y -> 0), with a forward floor. This is the "memory"
        # that the person ascended, so the dog follows up instead of drifting off-axis.
        # NOTE (sim): yaw/y_lateral are the base pose (legit on the real robot from the
        # IMU + the depth-detected stair center); the stairs are assumed along +x here.
        stairs_ahead = bool(det.get("stair_detected", False)) or bool(stairs_action_active)
        # Arm delay: stair_commit can't fire until stair_commit_arm_after_secs of sim-time
        # have elapsed. Prevents the commit heading-hold from engaging while the body is
        # still settling from spawn (a 7° roll at t=0.075s + immediate wz correction
        # spirals the robot sideways before it reaches the riser -- seen with 0.198m riser
        # detectable at 1.074m range from Go2X=1.0 spawn in the waypoint test).
        commit_armed = self._elapsed_dt >= float(self.cfg.stair_commit_arm_after_secs)
        # Keep the commit timer alive during the entire climb so it is still active after
        # disengage (enabling the post_climb_reacquire spin-in-place). Without this, depth
        # stairs detection goes False once the robot climbs past the riser horizon (~halfway
        # up), commit_until expires 25s later, and disengage finds committing=False -- the
        # spin-in-place heading correction never fires (run_20260624_073431_729: timer
        # expired at 11:39:34 but disengage wasn't until 11:40:48, robot stayed at 116° yaw).
        #
        # TERRAIN GATE: also require ground-truth riser confirmation (riser_dist_ahead not None,
        # meaning a real step rise is within ~1.5 m) OR that we are already in the climb state.
        # Without this, the depth camera sees the PERSON's body at ~0.6 m gap as "6 stairs"
        # (leading_edge = person gap), fires stair_commit 6.7 m from the real riser after the
        # 1 s arm delay, and overrides the person-follow wz for the entire flat approach so the
        # robot never tracks the person's lateral zigzag (run_20260625_004009_738: commit at
        # t≈1 s, wz≈0 for the full approach, max robot |y| 0.15 m vs person ±1.0 m zigzag).
        _terrain_confirms = (riser_dist_ahead is not None) or (self.state == "climb")
        if (self.cfg.stair_commit_enabled and commit_armed
                and (stairs_ahead or self.state == "climb") and not person_detected
                and _terrain_confirms):
            self._commit_until = _t + float(self.cfg.stair_commit_max_sec)
        committing = (
            bool(self.cfg.stair_commit_enabled)
            and (_t < self._commit_until)
            and not person_detected
        )
        # Person re-detected: clear the post-climb re-acquisition flag so normal follow resumes.
        if person_detected and self._post_climb_reacquire:
            self._post_climb_reacquire = False
        self._committing = committing
        # Capture the heading the INSTANT commitment begins and hold THAT. In sim, yaw 0 ==
        # up the +x staircase so commit_yaw0 ~ 0 and the behaviour is ~unchanged; on the real
        # robot absolute yaw 0 is "wherever it booted", so driving yaw -> 0 would steer toward
        # the power-on orientation instead of up the stairs. Holding the commit-time heading
        # (the shortest-turn error to it) fixes that on hardware and is a no-op in sim.
        if committing:
            if self._commit_yaw0 is None:
                self._commit_yaw0 = float(yaw)
        else:
            self._commit_yaw0 = None
        _commit_yaw_err = (
            _wrap_pi(float(yaw) - float(self._commit_yaw0))
            if self._commit_yaw0 is not None else 0.0
        )
        wz_override: Optional[float] = None
        vx_floor: Optional[float] = None
        if committing:
            # Heading lock: hold the commit-time heading (err -> 0) and steer back toward the
            # centerline (y -> 0). Applies in BOTH walk and climb states so the blind_rl
            # backend can use it as a seed when no person bearing history is available (see
            # isaac_env.py blind_rl path: _ho["wz_override"] fallback). Without this, a climb
            # entered with person_detected=False had _last_climb_wz=None and wz fell through to
            # the controller's ~0 value, letting the robot yaw 116° uncorrected during the climb
            # (run_20260624_073431_729: yaw 1° -> 116° over 130s of blind_rl climb).
            wz_override = float(np.clip(
                -float(self.cfg.stair_commit_yaw_kp) * _commit_yaw_err
                - float(self.cfg.stair_commit_lat_kp) * float(y_lateral),
                -float(self.cfg.stair_commit_wz_max), float(self.cfg.stair_commit_wz_max)))
            if self.state == "walk" and not caller_hold:
                # Post-climb re-acquisition: while the robot is still off-axis after the climb,
                # suppress the forward floor so it spins in place (not arcs sideways) to face
                # forward again. Clear the flag once the heading error is small enough.
                # caller_hold gate (see header note): this floor drove _eff_cmd>0 against a
                # commanded stance-hold, arming the stall detector and manufacturing the
                # run 155123 wedge_stall engage at the patient.
                _yaw_large = abs(_commit_yaw_err) > math.radians(float(self.cfg.post_climb_yaw_threshold_deg))
                if self._post_climb_reacquire and _yaw_large:
                    vx_floor = None  # spin in place; no forward push while pointing sideways
                else:
                    if self._post_climb_reacquire:
                        self._post_climb_reacquire = False  # re-aligned; resume normal commit
                    vx_floor = float(self.cfg.stair_commit_vx_floor)

        # Stall = "commanded forward yet not moving forward". Use the EFFECTIVE forward
        # command incl. the stair-commit floor: the dog wedged at the riser IS being driven
        # up (vx_floor) even when the controller sends 0 (person lost), and the stall must
        # see that to fire and trigger the climb hand-off. body_fwd = heading-frame forward
        # velocity (not total speed, whose lateral jitter never settles at a riser).
        _eff_cmd = max(float(cmd_vx), float(vx_floor) if vx_floor is not None else 0.0)
        stalled = self.stall.update(dt, _eff_cmd, body_fwd if body_fwd is not None else body_speed)

        climb = False
        targets = None
        debug_tread_creep = False
        top_egress = False             # set True during the post-crest egress push
        climb_vx_floor: Optional[float] = None  # person-gated egress forward floor (or None)

        if self.state == "walk":
            le = det.get("leading_edge_distance")
            near_enough = (le is not None) and (float(le) <= float(self.cfg.handoff_distance_m))
            controller_ok = bool(stairs_action_active) or (not self.cfg.require_controller_stairs)
            armed = _t >= self._cooldown_until
            has_stairs = (
                bool(det.get("stair_detected", False))
                and int(det.get("stair_count", 0)) >= int(self.cfg.stair_min_count)
                and controller_ok
            )
            # Tread-center standoff (Rec 7): when the leading edge is known but the robot
            # is farther than the engage window, emit a creep-forward vx_floor so the front
            # feet arrive at the tread center (not jammed against the riser face) before
            # the climb engages. This ensures the swing foot has clearance to arc up onto
            # the tread instead of starting from behind the riser.
            _le_m = det.get("leading_edge_distance") if le is not None else None
            if (_le_m is not None and has_stairs
                    and float(_le_m) > float(self.cfg.climb_engage_standoff_m) + 0.10
                    and self._committing
                    and not caller_hold):  # no creep push against a commanded stance-hold
                # Too far to engage yet — creep forward to the standoff zone
                vx_floor = max(vx_floor or 0.0, float(self.cfg.stair_commit_vx_floor))
                debug_tread_creep = True
            else:
                debug_tread_creep = False

            # PREFERRED engage: the first riser is at a STANDOFF ahead (room to step) and
            # the person has gone up (committing). Engaging here -- before the dog jams its
            # front feet into the riser -- gives the climber space to swing a foot onto the
            # tread instead of shoving backward into the riser and flipping (run ..193034).
            room = (
                riser_dist_ahead is not None
                and float(self.cfg.climb_min_room_m) <= float(riser_dist_ahead) <= float(self.cfg.climb_engage_standoff_m)
            )
            approach_engage = has_stairs and room and self._committing
            # FALLBACK engage: already wedged at the riser (stall) -- a backstop for when
            # the riser distance is unknown. Likely jammed, so less ideal.
            stall_engage = has_stairs and stalled and near_enough
            # E2 (2026-07-12 review of run_sim_20260712_013638_835): person-as-risers ghost
            # veto (incident 8.3 class). That run's second engage fired reason="wedge_stall",
            # riser_dist_ahead_m=null, level_heights_m=[0.0, 0.232, 0.32, ...] -- a standing
            # patient read as an 11-step staircase -- while the dog was FOLLOWING (caller_hold
            # False, so the 8.15-corr-3 ENGAGE veto did not apply). See
            # stair_engage_person_ghost_veto's docstring for the full numeric derivation of
            # ghost_engage_gap_window_m. Applied to BOTH engage arms (not just stall_engage,
            # which is what that run happened to hit) since a person-as-risers reading could in
            # principle also satisfy the room/riser_dist_ahead test for approach_engage.
            ghost_veto = stair_engage_person_ghost_veto(
                leading_edge_distance_m=le,
                person_gap_m=person_gap_m,
                window_m=float(self.cfg.ghost_engage_gap_window_m),
            )
            # caller_hold veto (see header note; run_sim_20260711_155123_326): no NEW climb
            # may start while the perception controller commands a stance-hold -- both
            # spurious landing engages (20:00:56 wedge_stall at the patient, 20:01:42
            # post-flip) fired during a continuous caller STOP/hold=True/vx=0 stretch.
            # S1 stair-entry head-start gate (2026-07-12 review of run_sim_20260712_023126_786,
            # run 14): no NEW climb may start until the patient has pulled ``stair_entry_min_
            # lead_m`` ahead -- see stair_entry_lead_ok's docstring and HandoffConfig.
            # stair_entry_min_lead_m's docstring for the full numeric derivation + the
            # non-deadlock interval math against isaac_env.py's PATIENT_HARD_WAIT_LEAD_M.
            # Applied to BOTH engage arms, same reasoning as the ghost veto above: run 14's
            # collision engaged via stall_engage (already jammed) at a lead of only 0.765 m,
            # but a person-as-risers-adjacent approach_engage could equally fire too close.
            # PRE-CLIMB ONLY BY CONSTRUCTION (incident 8.15): this whole block only runs
            # while ``self.state == "walk"``, so an ONGOING climb is never touched by this
            # gate -- it can only ever prevent a climb from STARTING.
            lead_ok = stair_entry_lead_ok(
                patient_lead_m=patient_lead_m,
                min_lead_m=float(self.cfg.stair_entry_min_lead_m),
            )
            trigger = (bool(self.cfg.enabled) and armed
                       and (approach_engage or stall_engage)
                       and not caller_hold
                       and not ghost_veto
                       and lead_ok)
            if not trigger and (approach_engage or stall_engage) and armed and not caller_hold and ghost_veto:
                log_event(
                    self.logger, logging.INFO, "handoff_engage_vetoed_ghost",
                    "Stair ENGAGE vetoed: depth leading edge matches the GT patient distance -- "
                    "the 'staircase' reading is the standing patient, not real stairs",
                    reason=("approach_room" if approach_engage else "wedge_stall"),
                    leading_edge_m=le, person_gap_m=person_gap_m,
                    window_m=float(self.cfg.ghost_engage_gap_window_m),
                    stair_count=int(det.get("stair_count", 0)),
                    level_heights_m=det.get("level_heights_m"),
                )
            if (not trigger and (approach_engage or stall_engage) and armed and not caller_hold
                    and not ghost_veto and not lead_ok):
                # Rate-limited (8.8): this can otherwise log every frame for the whole hold.
                if (self._elapsed_dt - self._lead_gate_last_log_t) >= 2.0:
                    self._lead_gate_last_log_t = self._elapsed_dt
                    log_event(
                        self.logger, logging.INFO, "handoff_engage_vetoed_lead",
                        "Stair ENGAGE held: patient lead below the required stair-entry "
                        "head start -- waiting for her to pull further ahead before committing",
                        reason=("approach_room" if approach_engage else "wedge_stall"),
                        patient_lead_m=patient_lead_m,
                        required_lead_m=float(self.cfg.stair_entry_min_lead_m),
                        stair_count=int(det.get("stair_count", 0)),
                    )
            if trigger:
                reason = "approach_room" if approach_engage else "wedge_stall"
                backend = str(self.cfg.climb_backend)
                # The policy backends (parkour vision / blind RL) always attempt (the user's
                # goal); the IK backend is gated by climb_attempt (it flips in PhysX) -- off
                # => decision logged but the dog stays upright at the riser via the stair-commit.
                do_climb = (backend in ("parkour", "blind_rl")) or bool(self.cfg.climb_attempt)
                if not do_climb:
                    self._cooldown_until = _t + float(self.cfg.re_eval_cooldown_sec)
                    log_event(
                        self.logger, logging.INFO, "handoff_climb_suppressed",
                        "Stall/standoff at stairs -- climb decision made but the IK climb is off",
                        reason=reason, stair_count=int(det.get("stair_count", 0)),
                        riser_dist_ahead_m=riser_dist_ahead, leading_edge_m=le)
                else:
                    self.state = "climb"
                    self._climb_start_z = float(base_z) if base_z is not None else None
                    self._climb_t0 = _t
                    self._climb_elapsed = 0.0
                    self._climb_progress_z = float(base_z) if base_z is not None else None
                    self._climb_stall_sec = 0.0
                    self._climb_stall_retries = 0
                    if backend not in ("parkour", "blind_rl"):
                        self.climber.reset()
                        # Anti-jolt: seed the IK climber's slew limiter from the CURRENT
                        # joint pose so its first target ramps from where PGTT left the legs.
                        try:
                            q_act = self.pgtt.current_act_positions(go2)
                            if q_act is not None:
                                self.climber._prev_target = np.asarray(q_act, dtype=np.float32)
                        except Exception:
                            pass
                        # Velocity-matched hand-off (Rec 5): seed crawl speed from the
                        # measured body speed so the RL->IK transition ramps down smoothly
                        # instead of jumping from ~0.5 m/s to 0.07 m/s in one frame.
                        if body_speed is not None:
                            self.climber.set_initial_speed(float(body_speed), ramp_sec=0.5)
                    self._last_reason = reason
                    log_event(
                        self.logger, logging.INFO, "handoff_engage",
                        "Handing off PGTT -> %s stair climber" % backend,
                        engage_reason=reason, backend=backend,
                        riser_dist_ahead_m=riser_dist_ahead,
                        stair_count=int(det.get("stair_count", 0)),
                        leading_edge_m=le,
                        level_heights_m=det.get("level_heights_m"),
                        climbs_done=int(self._climbs_done),
                    )

        use_parkour = False
        if self.state == "climb":
            climb = True
            if str(self.cfg.climb_backend) in ("parkour", "blind_rl"):
                # The policy backend (parkour vision OR blind RL) is run by isaac_env (it owns
                # the torque-mode drive swap + depth submit); the FSM here only owns the
                # done/abort/timeout transition. No IK targets. (use_parkour == "use the
                # hot-swap policy path in isaac_env", which then branches on the backend.)
                use_parkour = True
                targets = None
            else:
                advance = float(cmd_vx) > 0.03
                targets = self.climber.step(
                    dt,
                    roll=float(roll), pitch=float(pitch),
                    roll_rate=float(roll_rate), pitch_rate=float(pitch_rate),
                    height_above_step=height_above_step,
                    foot_contacts=foot_contacts,
                    body_speed=body_speed,
                    advance=advance,
                )
            gained = (
                (float(base_z) - self._climb_start_z)
                if (base_z is not None and self._climb_start_z is not None) else 0.0
            )
            tilt = max(abs(float(roll)), abs(float(pitch)))
            # "Fully off the stairs" requires not just a flat profile ahead but the dog having
            # LEVELLED OUT -- not still pitched nose-down on the last tread. Per user directive:
            # keep the blind-RL climber (the only policy that can BOTH drive the dog forward off
            # the last step AND level the body there) until the dog is clear of stairs AND upright,
            # THEN switch to PGTT. Handing back while still angled leaves it dragging and stuck one
            # step short (run_sim_20260704_230322: dog stuck at x=6.26, pitch -12deg, h 0.16 m).
            crest_level = tilt <= float(getattr(self.cfg, "top_crest_level_tilt_rad", 0.17))
            # SIM-time elapsed (dt-accumulated), NOT wall clock: `now` is wall time and the
            # sim runs ~6x slower, so a wall-time window timed out the climb after only ~3s
            # of SIM climbing and bounced it back to PGTT (run ..203910). dt is the sim step.
            self._climb_elapsed += float(dt)
            elapsed = self._climb_elapsed
            is_policy = str(self.cfg.climb_backend) in ("parkour", "blind_rl")

            # --- "no more stairs ahead" (crest detection) ----------------------------------
            # PRIMARY signal = the depth detector reports a flat profile ahead (no riser, count
            # 0). CROSS-CHECK = the caller's ground-truth terrain reading (stairs_ahead_gt:
            # True == a step still rises above the dog's current tread within reach; False ==
            # flat ahead; None == not provided). Require BOTH to read clear so a transient flat
            # depth profile BETWEEN two risers mid-climb (one tread momentarily fills the band)
            # does not falsely declare the top -- and so the GT cross-check alone carries the
            # waypoint test where the parkour depth may be absent (det then reads trivially
            # clear). Debounced by top_clear_debounce_sec.
            det_clear = (not bool(det.get("stair_detected", False))) and int(det.get("stair_count", 0)) == 0
            ray_clear = (stairs_ahead_gt is None) or (stairs_ahead_gt is False)
            # Crest signal. When a GROUND-TRUTH terrain read is available (sim, or any port that
            # supplies stairs_ahead_gt), TRUST IT ALONE: a flat read ahead genuinely means the top
            # is reached, because mid-climb there is ALWAYS a riser within a tread ahead, so unlike
            # the depth detector the GT read has no "transient flat between two risers" false-top.
            # Requiring det_clear too used to WEDGE the dog on the last riser forever: a nose-down,
            # dragging climber (incident 8.9) keeps its depth camera framed on the treads it just
            # climbed, so det_clear never becomes true, the crest is never declared, and the
            # incident-8.8 safety refuses to hand the (still-"on-stairs") legs back to PGTT -> the
            # climber retries the top step indefinitely (run_sim_20260704_211236: stuck at x=6.27,
            # top step 6.27). GT-False is IMPOSSIBLE on the incline (a riser is always ahead), so
            # trusting GT NEVER hands the incline back to the flat walker (incident 8.8 preserved).
            # Real robot without a terrain read (stairs_ahead_gt None) falls back to the detector.
            if stairs_ahead_gt is not None:
                stairs_clear = bool(ray_clear)
            else:
                stairs_clear = det_clear and ray_clear
            egress_on = bool(self.cfg.top_egress_enabled) and is_policy
            if stairs_clear:
                self._top_clear_sec += float(dt)
            else:
                self._top_clear_sec = 0.0
                # A re-appearing riser (another flight after a landing) cancels any egress so
                # the climb policy keeps climbing instead of handing back on the intermediate
                # landing.
                if self._egress:
                    self._egress = False
                    self._egress_x0 = None
                    self._egress_sec = 0.0
                    log_event(self.logger, logging.INFO, "handoff_egress_cancel",
                              "More stairs detected during egress -- resuming climb",
                              stair_count=int(det.get("stair_count", 0)))
            # Crest requires BOTH a debounced flat-ahead reading AND the dog being level (see
            # crest_level above): stay in the climber until it has finished walking off the last
            # step and stood up, so PGTT only ever inherits an upright dog on the flat.
            crest = (self._top_clear_sec >= float(self.cfg.top_clear_debounce_sec)) and crest_level

            # --- enter egress at a sustained, debounced crest ------------------------------
            if egress_on and crest and not self._egress:
                self._egress = True
                self._egress_sec = 0.0
                self._egress_x0 = float(base_x)
                if not self._crest_logged:
                    self._crest_logged = True
                    log_event(self.logger, logging.INFO, "handoff_crest",
                              "Top of staircase reached -- egress before handing back to PGTT",
                              base_x=round(float(base_x), 3),
                              height_gained_m=round(float(gained), 3))

            # --- egress sub-phase: stay in the climb policy, person-gated forward push ------
            if self._egress:
                top_egress = True
                self._egress_sec += float(dt)
                self._egress_travel = abs(float(base_x) - float(self._egress_x0)) \
                    if self._egress_x0 is not None else 0.0
                # Person-collision gate: only push forward off the crest when the patient is
                # far enough ahead; otherwise emit a zero floor so the climb policy HOLDS in
                # place (the blind/parkour net stands) and never drives into the patient.
                clear_of_person = (person_gap_m is None) or (float(person_gap_m) >= float(self.cfg.top_egress_standoff_m))
                # Forward-goal gate: if there is an explicit goal ahead (e.g. the waypoint-test
                # target), stop pushing once we are within top_egress_goal_stop_m of it so the
                # egress does not overrun it and walk off the landing.
                at_goal = (forward_goal_dist_m is not None
                           and float(forward_goal_dist_m) <= float(self.cfg.top_egress_goal_stop_m))
                climb_vx_floor = float(self.cfg.top_egress_vx) if (clear_of_person and not at_goal) else 0.0
                self._egress_vx_floor = climb_vx_floor
                egress_done = (self._egress_travel >= float(self.cfg.top_egress_distance_m)
                               or self._egress_sec >= float(self.cfg.top_egress_max_sec)
                               or at_goal)
            else:
                egress_done = False

            # --- vertical-progress watchdog ------------------------------------------------
            # Keep climbing while the body is still RISING; only give up if it stops gaining
            # height for climb_stall_timeout_sec (genuinely wedged). This replaces the old fixed
            # timeout as the "give up" signal so a slow-but-progressing climb is NOT cut off a
            # step short of the top. NOT counted during egress (the top is flat, base_z plateaus
            # there by design -- egress_done governs that exit instead).
            if base_z is None:
                # No vertical odometry (real robot without a base-height source): the
                # progress watchdog CANNOT measure a wedge, so disable it EXPLICITLY. A
                # pinned base_z=0 would otherwise read as "no progress" and force-abort
                # every real climb after climb_stall_timeout_sec -> hand back to PGTT mid-
                # staircase -> flip. tilt-abort + climb_max_sec remain the real backstops.
                climb_stuck = False
            elif self._climb_progress_z is None or float(base_z) > self._climb_progress_z + float(self.cfg.climb_progress_min_m):
                self._climb_progress_z = float(base_z)
                self._climb_stall_sec = 0.0
                climb_stuck = False
            else:
                if not self._egress:
                    self._climb_stall_sec += float(dt)
                climb_stuck = (not self._egress) and (self._climb_stall_sec >= float(self.cfg.climb_stall_timeout_sec))

            # --- exit decision -------------------------------------------------------------
            # Policy backends (parkour / blind RL) exit on a COMPLETED egress (the primary path)
            # -- crested AND walked the rear feet onto the flat. Otherwise they hand back only on
            # a tilt-abort or the progress watchdog (no vertical gain). `climb_max_sec` is now an
            # ABSOLUTE runaway backstop. The IK backend keeps its per-riser height-gain exit.
            done_ik = (not is_policy) and (gained >= float(self.cfg.climb_riser_height_m))
            done_egress = self._egress and egress_done and stairs_clear
            hard_cap = elapsed >= float(self.cfg.climb_max_sec)
            abort = tilt >= float(self.cfg.climb_abort_tilt_rad)
            # --- SAFETY: never hand the INCLINE back to the flat-ground walker (incident 8.8) ----
            # The two "give up" exits -- the vertical-progress stall watchdog (climb_stuck) and the
            # runaway hard-cap -- both set state="walk", handing the legs to PGTT. PGTT is a
            # FLAT-GROUND policy; taking the legs mid-staircase (26 deg incline) topples the dog.
            # run_sim_20260703_193958 handed back on a mid-stair stall at z=1.607 m -- still UPRIGHT
            # (14 deg tilt, ~step 9 of 14) -- and flipped 95 s later. So these give-up exits may fire
            # ONLY once we have POSITIVE evidence the top is reached: `stairs_clear` (BOTH the depth
            # detector AND the GT terrain read flat ahead, the same debounced crest signal used by
            # egress; on the real robot stairs_ahead_gt is None so this reduces to the depth clear).
            # This mirrors the design already applied to `climb_max_sec` (see handoff_config.py).
            # The give-up exits may hand back to PGTT ONLY once the dog is FULLY off the stairs:
            # flat ahead (stairs_clear) AND levelled out (crest_level). While still on the incline
            # OR still pitched nose-down on the last tread, keep the blind-RL climber driving -- it
            # is the only policy that can walk the dog forward off the last step and stand it up;
            # handing back early leaves it dragging (incident 8.8 flip / the last-tread wedge).
            _fully_off = bool(stairs_clear and crest_level)
            if hard_cap and not _fully_off:
                # Runaway backstop is a BACKSTOP, not an event -- suppress it until fully off so a
                # slow-but-progressing climb runs to the real crest + level instead of bailing.
                hard_cap = False
            if climb_stuck and not _fully_off:
                # Wedged or still nose-down at the top: HOLD the climber (the only controller that can
                # balance on the stair AND drive the dog forward off the last tread), reset the
                # progress window for a fresh attempt, and emit a heartbeat WARNING so a stuck climb is
                # VISIBLE (incident 8.8: a safe-hold must announce itself) rather than silently
                # surrendering. The dog keeps trying until it is off + upright or the wall-cap ends it.
                self._climb_stall_retries += 1
                log_event(
                    self.logger, logging.WARNING, "handoff_climb_stall_hold",
                    "Climb stalled on the incline -- HOLDING the climber (NOT handing back to the "
                    "flat walker mid-staircase); resetting the progress window and retrying",
                    retry=int(self._climb_stall_retries),
                    height_gained_m=round(float(gained), 3),
                    elapsed_sec=round(float(elapsed), 2), tilt_rad=round(float(tilt), 3),
                    base_z=(round(float(base_z), 3) if base_z is not None else None),
                )
                self._climb_stall_sec = 0.0
                self._climb_progress_z = float(base_z) if base_z is not None else None
                climb_stuck = False
            if done_ik or done_egress or abort or climb_stuck or hard_cap:
                reason = ("riser_climbed" if done_ik else
                          "top_egress_done" if done_egress else
                          "aborted_tilt" if abort else
                          "climb_stalled" if climb_stuck else "timeout")
                self.state = "walk"
                self._cooldown_until = _t + float(self.cfg.re_eval_cooldown_sec)
                self.stall.reset()
                self._egress = False
                self._egress_x0 = None
                self._egress_sec = 0.0
                self._top_clear_sec = 0.0
                self._crest_logged = False
                top_egress = False
                climb_vx_floor = None
                _stall_sec_at_exit = self._climb_stall_sec
                self._climb_progress_z = None
                self._climb_stall_sec = 0.0
                if done_ik or done_egress:
                    self._climbs_done += 1
                if done_egress:
                    self._post_climb_reacquire = True
                    # Cap the remaining commit window to a short re-acquisition budget so the
                    # 0.22 m/s vx_floor doesn't drive the robot past the stair top for 25 s
                    # (run_20260625_004009_738: x=9.694 m, 3.4 m overshoot, then fell off edge).
                    self._commit_until = min(
                        self._commit_until,
                        _t + float(self.cfg.post_egress_commit_sec),
                    )
                log_event(
                    self.logger, logging.INFO, "handoff_disengage",
                    "Stair climber handing back to PGTT walker",
                    reason=reason, height_gained_m=round(float(gained), 3),
                    elapsed_sec=round(elapsed, 2), tilt_rad=round(float(tilt), 3),
                    egress_travel_m=round(float(self._egress_travel), 3),
                    stall_sec=round(float(_stall_sec_at_exit), 2),
                    climbs_done=int(self._climbs_done),
                )

        return {
            "climb": bool(climb),
            "use_parkour": bool(use_parkour),
            "targets_act": targets,
            "state": self.state,
            "stalled": bool(stalled),
            "committing": bool(self._committing),
            "wz_override": wz_override,
            "vx_floor": vx_floor,
            "top_egress": bool(top_egress),
            "climb_vx_floor": climb_vx_floor,
            "tread_creep_active": bool(debug_tread_creep) if self.state == "walk" else False,
            "detect": det,
            "telemetry": self.telemetry(),
        }

    def telemetry(self) -> Dict[str, Any]:
        det = self._det
        t = {
            "handoff_state": self.state,
            "handoff_climbs_done": int(self._climbs_done),
            "stair_commit": bool(self._committing),
            "stair_count": int(det.get("stair_count", 0)),
            "stair_leading_edge_m": det.get("leading_edge_distance"),
            "stair_detected": bool(det.get("stair_detected", False)),
            "handoff_egress": bool(self._egress),
            "handoff_top_clear_sec": round(float(self._top_clear_sec), 3),
            "handoff_egress_sec": round(float(self._egress_sec), 3),
            "handoff_egress_travel_m": round(float(self._egress_travel), 3),
            "handoff_egress_vx_floor": self._egress_vx_floor,
            "handoff_climb_stall_sec": round(float(self._climb_stall_sec), 3),
            "handoff_climb_stall_retries": int(self._climb_stall_retries),
            "post_climb_reacquire": bool(self._post_climb_reacquire),
        }
        t.update(self.stall.telemetry())
        if self.state == "climb":
            t["climber"] = self.climber.telemetry()
        return t
