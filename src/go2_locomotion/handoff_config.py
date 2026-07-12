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

from go2_locomotion.tilt_limits import FSM_ABORT_TILT_RAD


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
    #
    # E2 DECISION (2026-07-12 review of run_sim_20260712_013638_835, kept False): re-checked
    # against that run's REAL (non-ghost) engage -- wall-clock-correlated to
    # vision_main_trace.jsonl frame_timing, sim_t=39.515s, robot x=1.612m (the engage itself
    # logged riser_dist_ahead_m=0.4, right at the base) -- debug_info["stairs_action_active"]
    # read False THERE (stairs_action_active_genuine also False; only stairs_detected/
    # stair_climbing_latch were True). The controller's flag did not go True until x=1.888m
    # (later, per the grader), by which point riser_dist_ahead would likely have dropped below
    # climb_min_room_m (front feet jammed) -- flipping this to True would have delayed the
    # smooth "approach_room" engage into a worse "wedge_stall" one, or missed the window
    # entirely (independently re-verified by scanning vision_main_trace.jsonl for the first
    # True frame: stairs_action_active does not go True until sim_t=52.43s, x=1.884m -- ~13 s
    # / 0.27 m LATER than the real engage at sim_t=39.515s, x=1.612m). Left False; the
    # person-as-risers ghost (incident 8.3 class) is instead vetoed
    # directly by leading-edge-vs-GT-patient-distance -- see
    # ``stair_engage_person_ghost_veto`` / ``ghost_engage_gap_window_m`` below.
    require_controller_stairs: bool = False

    # --- E2: person-as-risers ghost-engage veto (incident 8.3 class) ---------
    # A standing/close patient back-projects into the depth detector as a stack of fake
    # risers (CLAUDE.md 8.3/8.15-corr-3); when the GT patient distance is available (sim
    # only -- see HandoffController.update's person_gap_m parameter, already threaded from
    # isaac_env._run_pgtt_handoff's live sim GT), a leading_edge_distance reading that lands
    # within this window of the GT patient distance is that patient, not real stairs, and
    # ENGAGE is vetoed. Sized from run_sim_20260712_013638_835's ghost engage (05:48:23.9
    # wall time -> wall-clock-correlated vision_main_trace.jsonl frame at delta=0.11s):
    # leading_edge_m=0.648 (from the handoff_engage log) vs a GT planar patient gap that was
    # STABLE at 1.08-1.11 m across the full +/-2s bracket around the engage (computed from
    # frame_meta.gt_patient and stair_demo.robot.x_m/y_m, both true-sim-state and much less
    # noisy than the perceived depth_distance_m, which swung 0.63-1.36 m in the same window)
    # -- a measured diff of ~0.45 m. The task brief's originally-suggested 0.35 m window does
    # NOT cover this measured diff (0.448 > 0.35) -- verified numerically, not assumed (see
    # CLAUDE.md's "verify geometric claims numerically" lesson) -- so this ships at 0.5 m
    # instead, which still leaves ~0.9 m of clearance below the SAME run's real engage
    # separation (leading_edge_m=0.451 vs a GT patient distance of ~1.7 m => diff ~1.25 m,
    # never vetoed at any window below ~1.2 m).
    ghost_engage_gap_window_m: float = 0.5

    # --- S1: stair-entry head-start gate (2026-07-12 review of the S1/S2 patient-
    # clearance pass; runs 13/14 both graded proximity failures DURING the climb rather
    # than at the final state) ------------------------------------------------
    # ENGAGE (state=="walk" -> "climb") is additionally gated on the patient having a
    # sufficient head start onto the staircase: ``patient_lead_m`` (sim GT, the SAME
    # along-path measure -- ``patient.x - base_x`` -- as ``PATIENT_HARD_WAIT_LEAD_M`` in
    # isaac_env.py's ``update_person_patrol``) must be ``None`` (real hardware, no GT --
    # mirrors ``ghost_engage_gap_window_m``'s None contract, this gate is a no-op there)
    # OR >= this value. run_sim_20260712_023126_786 (run 14) engaged via "wedge_stall" at a
    # GT planar gap of just 0.765 m (handoff_engage log, sim t=28.35-28.5, base_x=1.76-1.8)
    # -- already inside the ``climb_gap_brake_scale`` taper zone (brake_stop_m default
    # 0.85 m) -- then closed to 0.452 m eight climb-steps later (fall_diag sim t=30.975,
    # base_x=2.126) while ``policy_cmd`` stayed pinned at the unbraked
    # ``--stair-forward-floor`` (0.16 m/s) because of a SEPARATE bug in the caller's brake
    # clamp (see ``core.control.stair_policy.mid_climb_floor_capped_command``, S2 of the
    # same review). S1 stops the dog COMMITTING onto the staircase at all until she has
    # pulled well ahead, so S2's mid-climb brake never needs to arrest a close-quarters
    # climb in the first place. ~2.4 m is about 4-5 tread depths (``step_depth_m`` ~0.5-
    # 0.6 m in this demo's stair preset), i.e. she is already several steps up before the
    # dog takes its first riser.
    #
    # DEADLOCK CHECK (incident 8.7: state the anchors, don't just assert the property):
    # isaac_env.py's ``PATIENT_HARD_WAIT_LEAD_M`` (patient hard-STOPS once she leads the
    # dog by more than that constant) was raised from 1.7 -> 3.2 m in the SAME change so
    # the interval [stair_entry_min_lead_m, PATIENT_HARD_WAIT_LEAD_M) = [2.4, 3.2) stays
    # NON-EMPTY: while the dog holds below this gate's lead (it vetoes ENGAGE, and the
    # near-riser forward push is separately braked to ~0 by the S2-fixed
    # ``climb_gap_brake_scale`` once she is close, so she keeps opening the gap), she is
    # BELOW her own 3.2 m hard-wait threshold and keeps walking -- both sides can never
    # hold simultaneously. (``PATIENT_PACE_GAP_MAX_M`` = 2.7 sits inside [gate, 3.2) --
    # she eases toward her slow-walk floor through [1.6, 2.7], then continues at that
    # floor from 2.7 to 3.2 -- so the hand-off from "gate releases" to "she'd otherwise
    # freeze" is a smooth pace-down, not an abrupt stop either side.)
    #
    # 2.4 -> 2.0 (2026-07-12, runs 18+19): the non-empty interval guarantees the lead
    # GROWS, but not how FAST near the gate -- inside her ease band the CLOSURE RATE
    # collapses (run 19 vetoed_lead trail: 2.065 -> 2.215 -> 2.339 over ~24 wall-s,
    # ~0.01 m/s; run 18 died 0.008 m short at 2.392) and the evaluator's robot_settled
    # idle-15s exit wins the race while the dog presses the riser waiting. 2.0 is BELOW
    # the slow-closure knee (run 19 hit 2.065 with ~25 s of run left) yet still ~2.4x the
    # 0.84 m tailgate entry that caused run 14's 0.452 m mid-climb gap -- she is 3-4
    # treads up before the dog's first riser.
    # 2.0 -> 2.2 (run 23, run_sim_20260712_120703_280): full chain finally completed
    # (engage->climb->crest->settle upright at x=6.92) but GT min patient gap was 0.53 vs
    # the 0.65 grader floor during the blind-carry climb toward her crest hard-wait; +0.2 m
    # of entry lead carries through the climb. Run 19's closure-rate trail showed 2.215
    # reached with ~12 s to spare, and post-unwedge runs reach the gate far faster.
    stair_entry_min_lead_m: float = 2.2

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
    # Bail to PGTT if the climber tips past this (~40 deg). SINGLE-SOURCED from
    # go2_locomotion.tilt_limits so it stays strictly BELOW the watchdog's climb-mode
    # latch-damp limit -- the graceful abort must be reachable before the collapse.
    climb_abort_tilt_rad: float = FSM_ABORT_TILT_RAD
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
