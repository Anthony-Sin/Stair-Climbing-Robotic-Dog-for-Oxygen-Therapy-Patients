"""Dual-policy walk<->climb handoff config + leg-clearance constant.

WHY: All handoff tunables live in one dataclass (``HandoffConfig``) so they are
exposed uniformly via the isaac_env argparse / run_sim launcher rather than
hardcoded in the control loop. ``GO2_LEG_CLEARANCE_M`` is the robot-config
leg-clearance the Task-2b detector reads as its default min-riser threshold, so
that threshold is sourced from config rather than baked into the detector.

Split out of ``pgtt_stair_handoff`` (which now re-exports these) so the config
is a single-responsibility module the perception + FSM modules import.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    # Minimum accumulated sim-time (s) before stair_commit can arm. Prevents the commit
    # heading-hold from firing while the robot is still settling from spawn (GoX=1.0 puts
    # the 0.198m riser in detection range from frame 1; the immediate wz correction while
    # the body is -7° rolled spirals the dog sideways before it reaches the riser).
    stair_commit_arm_after_secs: float = 1.0

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

    # --- post-climb re-acquisition: spin-in-place before resuming forward follow -------
    # After a successful top-egress handback, the blind/parkour climb policy may have rotated
    # the robot off-axis (e.g. 80 deg sideways). The stair-commit wz_override already steers
    # yaw -> 0, but its 0.22 m/s vx_floor makes the robot ARC sideways while rotating instead
    # of spinning in place -- the patient is never re-acquired. Solution: suppress the forward
    # floor while |yaw| exceeds this threshold so the robot spins in place to face forward
    # first, then resumes the commit floor once re-aligned (patient re-detected or yaw small).
    post_climb_yaw_threshold_deg: float = 20.0  # suppress fwd floor while |yaw| > this (deg)
    # After a successful top-egress handback, cap the stair_commit forward floor to this
    # window. Without this, the 25 s commit timer keeps driving the robot forward past the
    # stair top (run_20260625_004009_738: x=9.694 m, 3.4 m overshoot, then fell off edge).
    post_egress_commit_sec: float = 5.0
