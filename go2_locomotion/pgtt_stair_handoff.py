"""Dual-policy walk<->climb handoff for the PGTT walker.

WHY: PGTT is an excellent flat-ground walker but has NO stair-climb path (its
``step`` never receives the scripted-climb signal). The deterministic
``ClosedLoopStairClimber`` (lifted from commit 7ddb1f7) CAN attempt a step-up and
fails safe upright. This module is the glue that, while PGTT walks, watches for the
walker to STALL in front of a staircase and, when it does, hands the leg targets to
the climber for ONE riser, then hands back to PGTT -- exactly the Task-2 contract:

    if walking_policy_is_stalled()          # StallDetector (2a)
       and stair_detector.stair_detected    # DepthStairDetector (2b)
       and stair_detector.stair_count >= 2:
           switch_to -> ClosedLoopStairClimber (2c)
           climb one stair
           switch_back -> PGTT

Nothing here is hardcoded into the control loop: every threshold lives in
``HandoffConfig`` so it is tunable from the isaac_env argparse / run_sim launcher.
The detector runs on the body-mounted parkour depth camera (the robot's real depth
sensor feed), NOT a flat ground-truth heightmap, per the Task-2b requirement.

The climber emits 12 joint targets in PGTT ACT order (FR/FL/RR/RL x hip/thigh/calf)
-- identical to PgttLocomotionPolicy.ACT_ORDER -- so they apply through the walker's
existing name-based joint map and the same Kp40 position drive (see
PgttLocomotionPolicy.apply_external_act_targets).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from go2_locomotion.closed_loop_stair_climber import ClosedLoopStairClimber

try:
    from sim_logging_utils import log_event
except Exception:  # pragma: no cover - logging helper is optional
    def log_event(logger, level, action, message, **fields):
        if logger is not None:
            logger.log(level, "%s %s", message, fields)


# Max riser height (m) the flat-ground PGTT trot clears WITHOUT the climb policy.
# Risers TALLER than this are "real stairs" that need the handoff. This is the
# robot-config leg-clearance the Task-2b detector reads as its default min-riser
# threshold (so the threshold is sourced from config, not hardcoded in the
# detector). ~0.08 m is a conservative knee-height step for the Unitree Go2; the
# walker handles curbs/thresholds below it, taller risers trip the climber.
GO2_LEG_CLEARANCE_M = 0.08


@dataclass
class HandoffConfig:
    """All dual-policy handoff tunables (exposed via the isaac_env argparse)."""

    enabled: bool = True

    # --- 2a: dynamic stall detector ------------------------------------------
    # Stall == "commanded forward but not actually moving", sustained. A deliberate
    # follow HOLD (cmd_vx ~ 0) is NOT a stall -- the gate requires a live forward
    # command, so only a blocked walker (e.g. wedged at a riser) trips it.
    stall_speed_mps: float = 0.06        # measured body speed below this == "not moving"
    stall_cmd_min_mps: float = 0.05      # only a stall if we ARE commanding >= this forward
    stall_divergence_mps: float = 0.12   # commanded-vs-actual gap must exceed this
    stall_consec_sec: float = 0.6        # condition must hold this long (the "N timesteps")
    # Net forward displacement (m) the body must make over the stall window to NOT count
    # as stalled. A dog wedged at a riser bobs (body_vx oscillates +/-0.15) so the
    # instantaneous-speed gate above never sustains; integrating displacement over the
    # window cancels the bob and detects "commanded forward but going nowhere".
    stall_min_progress_m: float = 0.04

    # --- 2b: depth-edge stair detector ---------------------------------------
    stair_min_riser_m: float = GO2_LEG_CLEARANCE_M  # count a tread only if >= this above ground
    stair_min_count: int = 2             # need >= this many stairs to hand off (Task-2 spec)
    stair_max_range_m: float = 1.60      # only consider stair structure within this forward range
    stair_cam_height_m: float = 0.40     # parkour depth cam height above ground at a level stand
    stair_cam_vfov_deg: float = 56.5     # parkour depth cam vertical FOV (87 deg hFOV @ 106x60)
    stair_cam_pitch_deg: float = 0.5     # downward mount pitch of the parkour cam
    stair_band_frac: float = 0.40        # central column fraction used to build the depth profile
    stair_min_valid_px: int = 3          # min valid pixels per row to trust its median depth
    stair_detect_period_sec: float = 0.05  # throttle: re-run detection at most this often
    # Also require the controller's stairs_action_active gate? DEFAULT FALSE -- that
    # YOLO+depth gate is FLAKY (fired 0% in run_20260620_202910, blocking the handoff)
    # whereas the Isaac depth detector reliably reports the riser count; gate on the
    # reliable signal. Set True to additionally require the controller's confirmation.
    require_controller_stairs: bool = False

    # --- 2c: handoff / climb-one-stair ---------------------------------------
    # leading edge must be within this to switch. The near-horizontal parkour cam's
    # nearest VISIBLE tread sits ~0.75 m ahead even when the dog is at the base, so
    # this is looser than the controller's true 0.45 m proximity gate
    # (stairs_action_active, required by default) which enforces the tight standoff.
    handoff_distance_m: float = 0.90
    climb_riser_height_m: float = 0.15   # IK backend: "one stair climbed" == body rose this -> hand back
    climb_max_sec: float = 90.0          # ABSOLUTE hard cap on a continuous climb (s) -- a runaway
                                         # backstop only. The real "give up" signal is the vertical-
                                         # progress watchdog below: a fixed 20s cap used to cut off a
                                         # slow-but-progressing 14-step climb ~1 step short of the top
                                         # (run ..090155: height 1.957/2.10m at timeout), dropping it
                                         # onto PGTT mid-staircase -> flip. Keep it large so a genuine
                                         # climb runs to the actual crest, then egress hands back.
    # Vertical-progress watchdog: keep climbing while the body is still RISING; hand back only when it
    # stops gaining height for this long (genuinely wedged), not on a fixed wall clock. NOT applied
    # during egress (the top landing is flat, so base_z plateaus there by design).
    climb_stall_timeout_sec: float = 8.0  # hand back if no vertical progress for this long
    climb_progress_min_m: float = 0.05    # "progress" == base_z rose at least this much (< one riser)
    climb_abort_tilt_rad: float = 0.70   # bail to PGTT if the climber tips past this (~40 deg)
    re_eval_cooldown_sec: float = 1.5    # stay in WALK this long after a climb before re-arming
    # Whether to PHYSICALLY run the closed-loop climber when the switch triggers (the
    # handoff DECISION is always computed/logged either way). DEFAULT OFF: even when
    # engaged WITH ROOM at a standoff (not jammed), the climber NOSE-DIVES the dog the
    # instant it takes the legs (pitch 0 -> -40deg in 0.15s, run_20260620_195641) and
    # flips -- it targets the parkour stance under PGTT's softer kd, so the front drops.
    # This is the documented MJX->PhysX climb-transfer dead-end ([[project_pgtt_integration]]),
    # NOT a handoff bug. Off keeps the dog UPRIGHT at the riser under the stair-commit
    # (passes the no-fall check). Turn ON (--handoff-climb-attempt) only when tuning the
    # climber's stance/gains for PhysX or validating on real hardware.
    climb_attempt: bool = False
    # Climb backend: "parkour" hot-swaps the active policy to the Extreme-Parkour depth/
    # vision RL net (the trained perceptive climber -- PGTT walks, parkour climbs, then
    # back); "blind_rl" hot-swaps to the proprioceptive (blind) rl_sar Go2 RL net instead
    # (same gain-swap + continuous-climb contract, no depth); "ik" uses the deterministic
    # ClosedLoopStairClimber (flips in PhysX). For the policy backends (parkour, blind_rl)
    # the FSM only DECIDES (state == climb); isaac_env runs the policy + swaps the drive
    # gains. The policy backends always attempt (climb_attempt is the IK safety gate only).
    climb_backend: str = "parkour"
    # When the climb IS attempted, engage WITH ROOM: only when the first riser is between
    # climb_min_room_m and climb_engage_standoff_m ahead of the base (front feet not jammed).
    climb_engage_standoff_m: float = 0.65   # engage once the riser is this close ahead of base (m)
    climb_min_room_m: float = 0.40          # but NOT if closer than this (front feet jammed)

    # --- stair-commit: "person walked up out of frame -> keep going up" ----------
    # When stairs are detected ahead and the person is LOST (they ascended out of the
    # camera view), commit to going straight up: hold heading up the staircase and keep
    # a forward floor, instead of drifting open-loop (wz=0) off the side of the stairs
    # (run_20260620_184341: lost person at t=13.9 s, drifted to y=1.6 m -- 0.9 m off the
    # 0.70 m-half-width staircase -- and walked forward on flat ground beside it).
    stair_commit_enabled: bool = True
    stair_commit_yaw_kp: float = 2.0     # gain driving body yaw -> 0 (face up the +x staircase)
    stair_commit_lat_kp: float = 1.2     # gain steering back toward the stair centerline (y -> 0)
    stair_commit_wz_max: float = 0.6     # cap on the commit yaw-rate command (rad/s)
    stair_commit_vx_floor: float = 0.22  # forward floor (m/s) so it keeps approaching/climbing
    stair_commit_max_sec: float = 25.0   # keep committing this long after the last stairs+loss frame

    # --- top-of-stairs egress -> PGTT handback ------------------------------------
    # The PRIMARY climb exit. The old exit was the arbitrary `climb_max_sec` timeout, which
    # could hand back MID-CLIMB (long staircase) or, once at the top, keep the climb policy
    # running on flat ground until the timer expired. Either way PGTT could resume while the
    # dog still straddled the top riser and fall. Instead: when the dog crests (NO more risers
    # ahead -- depth detector AND ground-truth terrain both clear, debounced), STAY in the
    # climb policy and walk forward a short distance to pull the REAR feet off the last riser,
    # THEN hand back to PGTT on flat ground. The forward push is gated on the patient gap so
    # the dog never drives into the person waiting on the landing. `climb_max_sec` is kept as a
    # long safety BACKSTOP, and the tilt-abort still applies.
    top_egress_enabled: bool = True       # master toggle (default ON; the launcher flag disables)
    top_clear_debounce_sec: float = 0.6   # sustained "no stairs ahead" before declaring the crest
    top_egress_distance_m: float = 0.50   # forward travel past the crest to clear the rear feet
    top_egress_max_sec: float = 4.0       # hard cap on the egress push (backstop)
    top_egress_vx: float = 0.22           # forward floor (m/s) during egress (person-gated)
    top_egress_standoff_m: float = 0.60   # only push forward in egress if the person is >= this away
    top_egress_goal_stop_m: float = 0.12  # stop the egress push once within this of a forward GOAL
                                          # (e.g. the waypoint-test target) so it does not overrun it


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
        cmd = max(0.0, float(cmd_vx))
        fwd = float(body_fwd) if body_fwd is not None else 0.0
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

    def reset(self) -> None:
        self.stall.reset()
        self.detector.reset()
        self.climber.reset()
        self.state = "walk"
        self._climb_start_z = None
        self._climb_elapsed = 0.0
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

    def update(
        self,
        *,
        now: float,
        dt: float,
        go2: Any,
        depth_hw: Any,
        cmd_vx: float,
        stairs_action_active: bool,
        base_z: float,
        body_speed: Optional[float],
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
    ) -> Dict[str, Any]:
        # Throttled depth detection (the parkour cam only refreshes ~10 Hz).
        if depth_hw is not None and (now - self._last_detect_ts) >= float(self.cfg.stair_detect_period_sec):
            self._det = self.detector.detect(depth_hw)
            self._last_detect_ts = now
        det = self._det

        # --- Stair-commit ("person walked up out of frame -> keep going up") ----------
        # When stairs are detected ahead and the person is LOST, commit to driving
        # straight up the staircase: hold heading (yaw -> 0, the +x stair axis) and steer
        # back toward the centerline (y -> 0), with a forward floor. This is the "memory"
        # that the person ascended, so the dog follows up instead of drifting off-axis.
        # NOTE (sim): yaw/y_lateral are the base pose (legit on the real robot from the
        # IMU + the depth-detected stair center); the stairs are assumed along +x here.
        stairs_ahead = bool(det.get("stair_detected", False)) or bool(stairs_action_active)
        if self.cfg.stair_commit_enabled and stairs_ahead and not person_detected:
            self._commit_until = float(now) + float(self.cfg.stair_commit_max_sec)
        committing = (
            bool(self.cfg.stair_commit_enabled)
            and (float(now) < self._commit_until)
            and not person_detected
        )
        self._committing = committing
        wz_override: Optional[float] = None
        vx_floor: Optional[float] = None
        if committing and self.state == "walk":
            wz_override = float(np.clip(
                -float(self.cfg.stair_commit_yaw_kp) * float(yaw)
                - float(self.cfg.stair_commit_lat_kp) * float(y_lateral),
                -float(self.cfg.stair_commit_wz_max), float(self.cfg.stair_commit_wz_max)))
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
            armed = now >= self._cooldown_until
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
            _tread_depth_est = 0.28  # typical commercial tread depth (m); no sensor for it
            if (_le_m is not None and has_stairs
                    and float(_le_m) > float(self.cfg.climb_engage_standoff_m) + 0.10
                    and self._committing):
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
            trigger = bool(self.cfg.enabled) and armed and (approach_engage or stall_engage)
            if trigger:
                reason = "approach_room" if approach_engage else "wedge_stall"
                backend = str(self.cfg.climb_backend)
                # The policy backends (parkour vision / blind RL) always attempt (the user's
                # goal); the IK backend is gated by climb_attempt (it flips in PhysX) -- off
                # => decision logged but the dog stays upright at the riser via the stair-commit.
                do_climb = (backend in ("parkour", "blind_rl")) or bool(self.cfg.climb_attempt)
                if not do_climb:
                    self._cooldown_until = float(now) + float(self.cfg.re_eval_cooldown_sec)
                    log_event(
                        self.logger, logging.INFO, "handoff_climb_suppressed",
                        "Stall/standoff at stairs -- climb decision made but the IK climb is off",
                        reason=reason, stair_count=int(det.get("stair_count", 0)),
                        riser_dist_ahead_m=riser_dist_ahead, leading_edge_m=le)
                else:
                    self.state = "climb"
                    self._climb_start_z = float(base_z)
                    self._climb_t0 = float(now)
                    self._climb_elapsed = 0.0
                    self._climb_progress_z = float(base_z)
                    self._climb_stall_sec = 0.0
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
            gained = (float(base_z) - self._climb_start_z) if self._climb_start_z is not None else 0.0
            tilt = max(abs(float(roll)), abs(float(pitch)))
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
            crest = self._top_clear_sec >= float(self.cfg.top_clear_debounce_sec)

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
            if self._climb_progress_z is None or float(base_z) > self._climb_progress_z + float(self.cfg.climb_progress_min_m):
                self._climb_progress_z = float(base_z)
                self._climb_stall_sec = 0.0
            elif not self._egress:
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
            if done_ik or done_egress or abort or climb_stuck or hard_cap:
                reason = ("riser_climbed" if done_ik else
                          "top_egress_done" if done_egress else
                          "aborted_tilt" if abort else
                          "climb_stalled" if climb_stuck else "timeout")
                self.state = "walk"
                self._cooldown_until = float(now) + float(self.cfg.re_eval_cooldown_sec)
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
        }
        t.update(self.stall.telemetry())
        if self.state == "climb":
            t["climber"] = self.climber.telemetry()
        return t
