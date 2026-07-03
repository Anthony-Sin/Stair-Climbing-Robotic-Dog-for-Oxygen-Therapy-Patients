"""50 Hz low-level control node: /lowstate -> dual-policy + handoff -> /lowcmd.

This is the real-time heart of the port (run in its OWN process so vision/GPU jitter
never blows the 20 ms budget). It is a THIN rclpy shell -- every numeric/safety
decision lives in the host-tested pure modules:
  * LowStateArticulation  -- /lowstate -> the articulation the policies expect
  * DualPolicyRunner      -- PGTT walk <-> blind_rl climb via the handoff FSM
  * lowcmd_builder        -- the /lowcmd fields (mode/q/dq/kp/kd/tau) + CRC
  * SafetyWatchdog        -- tilt/staleness fail-safe -> damping

Startup gate: it will not write /lowcmd until (a) the sport-mode release node reports
``released`` and (b) /lowstate is fresh -- otherwise the MCU ignores /lowcmd anyway.

NOTE: the unitree_go message field names and the LowCmd CRC byte layout are the two
things that can only be byte-verified on the robot; the preflight CRC roundtrip (P6)
is the gate for the CRC, and field access is defensive where it can be.
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import replace
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

# Native Unitree ROS 2 messages (unitree_ros2 driver).
from unitree_go.msg import LowState, LowCmd

from real.control.lowstate_articulation import LowStateArticulation, GO2_DOF_NAMES
from real.control.dual_policy_runner import DualPolicyRunner, RobotState
from real.control.lowcmd_builder import (
    build_low_cmd_fields, build_damping_fields, crc32_core, N_CMD_SLOTS,
)
from real.control.safety_watchdog import SafetyWatchdog, go2_joint_limits
from real.control.command_gate import classify_command_age, FRESH, HOLD, DAMP
from real.control.follow_command import FollowCommand, FOLLOW_CMD_TOPIC, DEPTH_TOPIC
from real.perception.heightscan_provider import height_fn_from_grid
from real.perception.depth_to_policy import preprocess as preprocess_depth
from real.logging.real_telemetry import RealTelemetry
from real.ros2.qos import sensor_qos, reliable_qos, latched_qos
from go2_locomotion.pgtt_heightmap import PGTT_N_POINTS

from datetime import datetime

# Damping-command kd (mirrors lowcmd_builder.build_damping_fields default): the gain the
# resume ramp starts from, so the ramp is continuous with the fault-state damping.
_DAMP_KD = 5.0


def _quat_to_rpy(w, x, y, z):
    """[w,x,y,z] -> (roll, pitch, yaw) radians."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


class LowLevelControlNode(Node):
    def __init__(self) -> None:
        super().__init__("low_level_control")
        g = ReentrantCallbackGroup()
        # The control tick gets its OWN mutually-exclusive group so the executor can
        # NEVER start tick N+1 while tick N is still running. A single >20 ms tick
        # (torch-JIT, GC) would otherwise re-enter runner.step() on another executor
        # thread and corrupt the RL policy's prev_action / gait-phase state -> the
        # intermittent, unreproducible limb jerks. Subscriptions stay reentrant.
        ctrl_group = MutuallyExclusiveCallbackGroup()

        # ---- parameters (override via ros2 launch / real_robot.yaml) -------------
        p = self.declare_parameter
        self._lowcmd_topic = p("lowcmd_topic", "/lowcmd").value
        lowstate_topic = p("lowstate_topic", "/lowstate").value
        sport_state_topic = p("sport_state_topic", "/go2/sport_state").value
        self._sport_state_topic = sport_state_topic
        # unitree_ros2 SportModeState (body velocity + trunk height). Feeds the handoff
        # stall detector so ``stall_engage`` can actually fire (nothing subscribed before,
        # so the stall path was permanently off and the climb handoff was unreachable in
        # flat heightscan mode). Optional: absent -> body velocity stays None (guard off).
        sportmode_topic = p("sportmode_state_topic", "/sportmodestate").value
        stair_topic = p("stair_topic", "/go2/stair_detection").value
        heightscan_topic = p("heightscan_topic", "/go2/heightscan").value
        self._control_hz = float(p("control_hz", 50.0).value)
        self._require_released = bool(p("require_sport_released", True).value)
        # Follow-command staleness thresholds. Past cmd_timeout_sec we HOLD (zero the
        # velocity, keep the current walk/climb mode, balance in place); past cmd_damp_sec
        # we DAMP (safe-stop). This is THE guard against executing a dead vision process's
        # last command forever. cmd_timeout matches the MPPI sidecar's 0.4 s gate.
        self._cmd_timeout_sec = float(p("cmd_timeout_sec", 0.4).value)
        self._cmd_damp_sec = float(p("cmd_damp_sec", 1.0).value)
        self._climb_backend = str(p("climb_backend", "blind_rl").value)
        # Defaults point at the existing shared weights; the robot's launch/real_robot.yaml
        # overrides these to the on-device copies under real/models/.
        self._pgtt_path = str(p("pgtt_policy_path", "src/sim/models/pgtt/pgtt_go2_level17.npz").value)
        self._rl_path = str(p("rl_policy_path", "src/sim/models/locomotion/go2_robot_lab_policy.pt").value)
        walk_kp = float(p("walk_kp", 40.0).value)
        walk_kd = float(p("walk_kd", 0.5).value)
        climb_kp = float(p("climb_kp", 20.0).value)
        climb_kd = float(p("climb_kd", 0.5).value)
        # Telemetry ON by default: this is the safety platform's flight recorder next to
        # an oxygen patient; it must not depend on someone remembering to pass a flag. An
        # empty telemetry_dir now auto-derives a run dir instead of silently disabling.
        self._record_telemetry = bool(p("record_telemetry", True).value)
        telemetry_dir = str(p("telemetry_dir", "").value)

        # ---- shared state (written by callbacks, read by the 50 Hz tick) ---------
        # Staleness receipt clocks are MONOTONIC (time.monotonic), NOT the ROS/system
        # clock: an NTP step (forward >1 s -> spurious DAMP->collapse; backward -> blinds
        # the gate) must never move a single-process liveness measurement. ROS time buys
        # nothing here. The publisher's WIRE stamp stays ROS-clock (transport-skew EWMA).
        self._lock = threading.Lock()
        self._low_state = None
        self._low_state_ts: Optional[float] = None   # MONOTONIC receipt time of last /lowstate
        self._cmd = FollowCommand()
        self._last_cmd_ts: Optional[float] = None   # MONOTONIC receipt time of last follow command
        self._malformed_cmd_count = 0
        # Transport-skew tracking (task: the 13th-float stamp was dead). Each command
        # carries the publisher's ROS-clock stamp; we EWMA (ros_receipt - stamp) to make
        # genuine transport delay (DDS bursts, executor backlog) visible. Receipt-time
        # gating stays the control default; this is observability + a counted alert.
        self._cmd_skew_ewma: Optional[float] = None
        self._cmd_skew_alert_count = 0
        self._cmd_skew_alert_last_ts: float = 0.0
        self._depth_m: Optional[np.ndarray] = None
        self._heightscan: Optional[np.ndarray] = None
        self._stair = {}
        # Body velocity + trunk height from SportModeState (None until a message arrives).
        # body_speed feeds the stall detector; body_fwd is the heading-frame forward speed;
        # base_z is the trunk height. Absent -> the handoff disables the guards that need them.
        self._body_speed: Optional[float] = None
        self._body_fwd: Optional[float] = None
        self._base_z: Optional[float] = None
        self._sport_released = not self._require_released
        self._articulation: Optional[LowStateArticulation] = None
        # Flat-mode DepthStairDetector for the riser-distance fallback (lazily built).
        self._depth_stair_detector = None

        # Control-condition tracking for honest logging + exit_reason. ``_condition`` is
        # the current edge-de-duplicated state ("ok"/"cmd_hold"/"cmd_lost"/watchdog
        # reason); ``_exit_reason`` is what finish() records; a latched watchdog fault
        # freezes it (a tilt/stale trip is the run's outcome, not a later clean exit).
        self._condition = ""
        self._exit_reason = "completed"
        self._latched_fault = False

        # ---- LATCHED fault gate (fault-exit is a designed, terminal state) --------
        # A DAMP (fault) must NOT auto-re-arm: classify_command_age / the watchdog stale
        # path are stateless, so without a latch the next fresh command would snap
        # straight back to FULL-GAIN PD targets from policies whose state froze at fault
        # time, against a sagged/fallen body. Once latched, we KEEP publishing damping
        # until an explicit ~/rearm service call, and any resume is posture-gated and
        # GAIN-RAMPED (kp/kd ramp from damping up to nominal over a short window).
        self._damp_latched = False
        self._damp_latch_reason = ""
        self._rearm_requested = False       # set by the ~/rearm service, consumed on the tick
        self._resume_ramp_t0: Optional[float] = None  # monotonic start of the gain ramp, or None
        self._resume_ramp_sec = float(p("resume_ramp_sec", 1.5).value)
        # Posture sanity for a resume: tilt must be below this (upright enough to stand up).
        self._resume_max_tilt_rad = float(p("resume_max_tilt_rad", 0.35).value)

        # ---- policies + runner + watchdog ---------------------------------------
        self._runner = self._build_runner(walk_kp, walk_kd, climb_kp, climb_kd)
        # Wire the joint-limit watchdog class: inject the real Go2 limits at construction
        # so the check actually runs against THIS tick's computed targets before publish
        # (previously the limits were never supplied, so the joint-limit path was dead).
        _jlo, _jhi = go2_joint_limits()
        self._watchdog = SafetyWatchdog(joint_lower=_jlo, joint_upper=_jhi)
        self._walk_kp, self._walk_kd = float(walk_kp), float(walk_kd)

        # Tick-period jitter measurement (P: the loop feeds a HARDCODED nominal dt to
        # runner.step; these measure the ACTUAL wall period so Jetson jitter is visible
        # in the trace. Observability only -- the dt fed to the policy stays nominal).
        self._last_tick_ts: Optional[float] = None
        self._nominal_tick_s = 1.0 / max(1.0, self._control_hz)
        self._jitter_warn_last_ts: float = 0.0

        # Run telemetry (throttled to ~10 Hz) in the perf_tracker run-dir layout. On by
        # default; if no dir was configured we auto-derive one so it is never silently off.
        self._tel = None
        self._tel_count = 0
        if self._record_telemetry:
            if not telemetry_dir:
                telemetry_dir = self._default_telemetry_dir()
            try:
                self._tel = RealTelemetry(telemetry_dir)
                self._tel.start(timestamp=datetime.now().isoformat(timespec="seconds"),
                                command="real/ros2/low_level_control_node.py --ros2")
                self.get_logger().info(f"telemetry recording -> {telemetry_dir}")
            except Exception as exc:
                # Never let the flight recorder take down the 50 Hz loop -- but say so.
                self.get_logger().error(f"telemetry init failed ({exc}); continuing without it")
                self._tel = None

        # ---- ROS I/O -------------------------------------------------------------
        self._lowcmd_pub = self.create_publisher(LowCmd, self._lowcmd_topic, reliable_qos())
        self.create_subscription(LowState, lowstate_topic, self._on_lowstate, sensor_qos(), callback_group=g)
        self.create_subscription(Float32MultiArray, FOLLOW_CMD_TOPIC, self._on_cmd, sensor_qos(), callback_group=g)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, sensor_qos(), callback_group=g)
        self.create_subscription(Float32MultiArray, heightscan_topic, self._on_heightscan, sensor_qos(), callback_group=g)
        self.create_subscription(Float32MultiArray, stair_topic, self._on_stair, sensor_qos(), callback_group=g)
        # SportModeState -> body velocity (arms stall_engage) + trunk height. Imported and
        # subscribed defensively: some unitree_ros2 builds omit the type, and its absence
        # must NOT crash the control node -- it just leaves the velocity-gated path off.
        try:
            from unitree_go.msg import SportModeState  # type: ignore
            self.create_subscription(SportModeState, sportmode_topic, self._on_sportmode,
                                     sensor_qos(), callback_group=g)
            self.get_logger().info(f"SportModeState subscribed on {sportmode_topic} "
                                   "(body velocity -> stall_engage armed)")
        except Exception as exc:
            self.get_logger().warning(
                f"SportModeState unavailable ({exc}); stall_engage stays OFF "
                "(approach_engage still carries the handoff)"
            )
        # Latched QoS: sport_startup publishes "released" ONCE, latched. A VOLATILE/
        # BestEffort subscriber would never receive that latched history if it (re)starts
        # late -> it waits forever, robot flat, no error. Match the publisher's profile.
        self.create_subscription(String, sport_state_topic, self._on_sport_state, latched_qos(), callback_group=g)
        # Fault re-arm service (task: fault-exit is a designed, terminal state). A latched
        # DAMP/tilt fault is TERMINAL until an operator calls this; the resume it triggers
        # is posture-gated + gain-ramped, never a snap back to full gain.
        self._rearm_srv = self.create_service(Trigger, "~/rearm", self._on_rearm)
        self.create_timer(1.0 / max(1.0, self._control_hz), self._control_tick, callback_group=ctrl_group)

        self._interlock_guard()
        self._log_engage_paths()
        self.get_logger().info(f"low_level_control up @ {self._control_hz:.0f} Hz, backend={self._climb_backend}")

    # ------------------------------------------------------------- startup guards
    def _interlock_guard(self) -> None:
        """Refuse to start if the Docker/sidecar sport stack is up (two driving stacks,
        no physics interlock -- see DEPLOY.md). The native low-level stack RELEASES sport
        and DAMPs on staleness; the sidecar keeps sport ACTIVE and StopMove-stands. Running
        both fights over /lowcmd vs the sport service. Detected via GO2_SIDECAR_ACTIVE
        (set by the docker launch); override with GO2_ALLOW_SPORT_ACTIVE=1 if you have
        manually confirmed the sidecar is down.
        """
        if os.environ.get("GO2_SIDECAR_ACTIVE", "").strip() not in ("", "0", "false", "False"):
            if os.environ.get("GO2_ALLOW_SPORT_ACTIVE", "").strip() in ("1", "true", "True"):
                self.get_logger().error(
                    "GO2_SIDECAR_ACTIVE set but GO2_ALLOW_SPORT_ACTIVE=1 -- proceeding "
                    "with the NATIVE low-level stack anyway. Ensure the Docker/sidecar "
                    "sport stack is actually DOWN or the two will fight over the robot."
                )
            else:
                raise RuntimeError(
                    "REFUSING TO START: GO2_SIDECAR_ACTIVE is set -- the Docker/sidecar "
                    "sport stack appears to be up. The native low-level stack (this node) "
                    "and the sidecar are mutually exclusive by physics (one releases sport "
                    "and DAMPs on staleness, the other keeps sport active). Stop the "
                    "sidecar, or set GO2_ALLOW_SPORT_ACTIVE=1 to override once you have "
                    "confirmed it is down. See src/real/DEPLOY.md."
                )

    def _log_engage_paths(self) -> None:
        """BOOT log enumerating which climb-engage paths are armed and which inputs feed
        them, so a hardware run visibly reports whether the climb feature is reachable (it
        was UNREACHABLE in the shipped flat config: stall_engage had no velocity source and
        approach_engage had no riser source without LiDAR).

          * approach_engage needs riser_dist_ahead. Source = the LiDAR heightscan node's
            /go2/stair_detection (lidar mode) OR the D435 DepthStairDetector leading-edge
            fallback (flat mode, wired here). So approach_engage is ARMED in BOTH modes.
          * stall_engage needs body velocity from SportModeState -- ARMED only once
            SportModeState messages arrive; absent -> stall_engage stays OFF.
        """
        self.get_logger().info(
            "climb-engage paths: approach_engage=ARMED "
            f"(riser_dist_ahead <- LiDAR /go2/stair_detection if present, else D435 "
            "DepthStairDetector leading-edge fallback -- reachable in flat mode); "
            f"stall_engage=ARMED-WHEN-VELOCITY (body velocity <- SportModeState); "
            f"climb_backend={self._climb_backend}. If SportModeState is absent, "
            "stall_engage stays OFF and approach_engage carries the handoff."
        )

    # ------------------------------------------------------------- construction
    def _build_runner(self, walk_kp, walk_kd, climb_kp, climb_kd) -> DualPolicyRunner:
        from go2_locomotion.pgtt_locomotion_policy import PgttLocomotionPolicy, PgttPolicyConfig
        from go2_locomotion.rl_locomotion_policy import RLLocomotionPolicy, RLLocomotionPolicyConfig
        from go2_locomotion.pgtt_stair_handoff import HandoffController, HandoffConfig

        dof = list(GO2_DOF_NAMES)
        pgtt = PgttLocomotionPolicy(
            PgttPolicyConfig(policy_path=self._pgtt_path, kp=walk_kp, kd=walk_kd),
            dof, height_fn=(lambda x, y: 0.0), logger=self.get_logger(),
        )
        blind = None
        if self._climb_backend == "blind_rl":
            blind = RLLocomotionPolicy(
                RLLocomotionPolicyConfig(policy_path=self._rl_path, kp=climb_kp, kd=climb_kd),
                dof, logger=self.get_logger(),
            )
        handoff = HandoffController(
            HandoffConfig(climb_backend=self._climb_backend), pgtt, logger=self.get_logger(),
        )
        runner = DualPolicyRunner(
            pgtt, blind, handoff, climb_backend=self._climb_backend,
            walk_kp=walk_kp, walk_kd=walk_kd, climb_kp=climb_kp, climb_kd=climb_kd,
            logger=self.get_logger(),
        )
        runner.reset()
        return runner

    # ------------------------------------------------------------- subscriptions
    def _on_lowstate(self, msg) -> None:
        with self._lock:
            self._low_state = msg
            self._low_state_ts = time.monotonic()   # MONOTONIC receipt clock (NTP-immune)

    def _on_cmd(self, msg) -> None:
        with self._lock:
            try:
                cmd = FollowCommand.unpack(msg.data)
                self._cmd = cmd
                self._last_cmd_ts = time.monotonic()   # MONOTONIC staleness clock (NTP-immune)
                # Transport-skew EWMA: compare the publisher's ROS-clock stamp against our
                # ROS-clock receipt (both system-clock, so comparable). Receipt-time gating
                # stays the control default; this only surfaces genuine transport delay.
                if cmd.stamp:
                    skew = self._now() - float(cmd.stamp)
                    a = 0.1
                    self._cmd_skew_ewma = (skew if self._cmd_skew_ewma is None
                                           else (1.0 - a) * self._cmd_skew_ewma + a * skew)
            except Exception:
                self._malformed_cmd_count += 1
                if self._malformed_cmd_count == 1:
                    self.get_logger().warning(
                        "dropped a malformed follow command (unpack failed); "
                        "further drops counted in _malformed_cmd_count"
                    )

    def _on_depth(self, msg) -> None:
        # 16UC1 millimetres -> float32 metres (the stair detector + mask expect metres).
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
            with self._lock:
                self._depth_m = arr.astype(np.float32) / 1000.0
        except Exception:
            pass

    def _on_heightscan(self, msg) -> None:
        with self._lock:
            self._heightscan = np.asarray(msg.data, dtype=np.float32)

    def _on_stair(self, msg) -> None:
        # [stair_detected, stair_count, leading_edge_distance, riser_dist_ahead]
        d = list(msg.data)
        with self._lock:
            self._stair = {
                "riser_dist_ahead": d[3] if len(d) > 3 and not math.isnan(d[3]) else None,
            }

    def _on_sportmode(self, msg) -> None:
        """SportModeState -> body planar speed + heading-forward speed + trunk height.

        ``velocity`` is body-frame [vx, vy, vz] (m/s); ``position`` is [x, y, z] (m). We
        publish body_speed (planar magnitude, drives the stall detector), body_fwd (vx,
        heading-frame forward), and base_z (trunk height) so the handoff's velocity- and
        height-gated guards are actually fed on the real robot.
        """
        try:
            vel = getattr(msg, "velocity", None)
            pos = getattr(msg, "position", None)
            vx = float(vel[0]) if vel is not None and len(vel) > 0 else 0.0
            vy = float(vel[1]) if vel is not None and len(vel) > 1 else 0.0
            speed = math.hypot(vx, vy)
            bz = float(pos[2]) if pos is not None and len(pos) > 2 else None
        except Exception:
            return
        with self._lock:
            self._body_speed = speed
            self._body_fwd = vx
            self._base_z = bz

    def _on_sport_state(self, msg) -> None:
        with self._lock:
            self._sport_released = (str(msg.data).strip().lower() == "released")

    def _on_rearm(self, request, response):
        """~/rearm (std_srvs/Trigger): clear a LATCHED DAMP/tilt fault -> posture-gated,
        gain-ramped resume. This is the ONLY way out of a latched fault (the fault state
        is terminal by design). The actual clear + posture check + ramp start happen on
        the control tick (single-threaded w.r.t. the runner); here we just request it.
        """
        with self._lock:
            latched = self._damp_latched or self._watchdog.faulted
            self._rearm_requested = True
        response.success = True
        response.message = ("re-arm requested; resume is posture-gated + gain-ramped"
                            if latched else "no latched fault; re-arm is a no-op")
        self.get_logger().warning("~/rearm called: clearing latched fault on the next tick "
                                  "(posture-gated, gain-ramped resume)")
        return response

    # ------------------------------------------------------------------ control
    def _control_tick(self) -> None:
        tick_t0 = time.perf_counter()
        # Measure the ACTUAL tick period (Jetson-jitter observability). Guard the first
        # tick (no prior timestamp). This does NOT change the dt fed to the policy.
        tick_dt_meas_ms: Optional[float] = None
        if self._last_tick_ts is not None:
            tick_dt_meas_ms = (tick_t0 - self._last_tick_ts) * 1000.0
        self._last_tick_ts = tick_t0

        with self._lock:
            low_state = self._low_state
            ts = self._low_state_ts
            cmd = self._cmd
            last_cmd_ts = self._last_cmd_ts
            depth = self._depth_m
            heightscan = self._heightscan
            stair = dict(self._stair)
            released = self._sport_released
            body_speed = self._body_speed
            body_fwd = self._body_fwd
            base_z = self._base_z
            rearm = self._rearm_requested
            self._rearm_requested = False
            skew_ewma = self._cmd_skew_ewma
        now = self._now()                 # ROS clock: telemetry + wire-stamp skew only
        now_mono = time.monotonic()       # MONOTONIC: the staleness gates run on THIS

        # Warn (throttled) when the measured period blows the real-time budget. Throttle
        # to >1.5x nominal and at most once/second so a jitter storm can't flood the log.
        if tick_dt_meas_ms is not None and tick_dt_meas_ms > 1.5 * self._nominal_tick_s * 1000.0:
            if (tick_t0 - self._jitter_warn_last_ts) >= 1.0:
                self._jitter_warn_last_ts = tick_t0
                self.get_logger().warn(
                    f"control tick jitter: measured {tick_dt_meas_ms:.1f} ms "
                    f"(nominal {self._nominal_tick_s * 1000.0:.1f} ms) -- Jetson budget overrun"
                )

        # Gate: do not touch /lowcmd until sport mode is released AND state is fresh.
        if low_state is None or (self._require_released and not released):
            return

        roll, pitch, yaw = self._rpy(low_state)

        # ---- LATCHED FAULT: terminal until an explicit ~/rearm ------------------------
        # A DAMP/tilt fault is NOT self-clearing (the gates are stateless -- a fresh command
        # would otherwise snap back to full-gain PD against a fallen body). Stay damped until
        # ~/rearm, then only resume on a posture sanity check with a GAIN-RAMPED stand.
        if self._damp_latched:
            if not rearm:
                self._publish_damping()
                return
            tilt_now = max(abs(float(roll)), abs(float(pitch)))
            if tilt_now > self._resume_max_tilt_rad:
                self.get_logger().error(
                    f"~/rearm refused: body tilt {tilt_now:.2f} rad exceeds resume limit "
                    f"{self._resume_max_tilt_rad:.2f} rad -- level the dog and re-arm again"
                )
                self._publish_damping()
                return
            # Posture OK: clear the latch, reset the watchdog, and start the gain ramp.
            self._watchdog.reset()
            self._damp_latched = False
            self._damp_latch_reason = ""
            self._latched_fault = False
            self._resume_ramp_t0 = now_mono
            self.get_logger().warning(
                "~/rearm accepted: posture OK -> gain-ramped resume "
                f"({self._resume_ramp_sec:.1f} s ramp to nominal gains)"
            )

        # Tilt limit is climb-mode aware: during a climb tolerate more tilt so the FSM's
        # graceful abort (0.70) fires BEFORE the watchdog latch-damp. Use last tick's active
        # backend (the runner tracks it) since the watchdog evaluates before this step.
        self._watchdog.set_climb_mode(self._runner.in_climb)
        verdict = self._watchdog.evaluate(
            now=now_mono, last_state_ts=ts, roll=roll, pitch=pitch,
        )
        if not verdict.ok:
            # Tilt / joint-limit latch; a stale-lowstate gap is TRANSIENT and auto re-arms in
            # the watchdog, so don't freeze the run's exit_reason on it. Log once + damp.
            _latching = verdict.reason != "lowstate_stale"
            self._enter_condition(verdict.reason, latching=_latching)
            if _latching:
                self._damp_latched = True         # terminal until ~/rearm
                self._damp_latch_reason = verdict.reason
            self._publish_damping()
            return

        # Command staleness: NEVER execute the last follow command forever. If vision
        # (process A) dies/lags, degrade gracefully -- HOLD (zero velocity, keep the
        # current walk/climb mode so we don't hand back mid-stair, balance in place),
        # then DAMP if it stays gone. classify_command_age is pure + unit-tested. Age is
        # measured on the MONOTONIC receipt clock (NTP-immune).
        cmd_age = float(now_mono - last_cmd_ts) if last_cmd_ts is not None else float("inf")
        freshness = classify_command_age(
            age_sec=cmd_age, ever_received=last_cmd_ts is not None,
            timeout_sec=self._cmd_timeout_sec, damp_sec=self._cmd_damp_sec,
        )
        if freshness == DAMP:
            self._enter_condition("cmd_lost", latching=True)
            self._damp_latched = True             # a lost command stream is terminal until ~/rearm
            self._damp_latch_reason = "cmd_lost"
            self._publish_damping()
            return
        if freshness == HOLD:
            self._enter_condition("cmd_hold")
            cmd = replace(cmd, vx=0.0, wz=0.0, yaw_err=0.0, hold=True)
        else:
            self._enter_condition("ok")

        # Transport-skew alert (task: use the 13th-float stamp). Counted + throttled so a
        # DDS burst / executor backlog becomes visible without flooding the log. Never gates
        # control (receipt-time gating already did); pure observability.
        if skew_ewma is not None and skew_ewma > self._cmd_timeout_sec:
            self._cmd_skew_alert_count += 1
            if (now_mono - self._cmd_skew_alert_last_ts) >= 1.0:
                self._cmd_skew_alert_last_ts = now_mono
                self.get_logger().warn(
                    f"follow-command transport skew EWMA {skew_ewma * 1000.0:.0f} ms "
                    f"(> {self._cmd_timeout_sec * 1000.0:.0f} ms timeout) -- DDS/executor "
                    f"delay; count={self._cmd_skew_alert_count}"
                )

        if self._articulation is None:
            self._articulation = LowStateArticulation(low_state)
        else:
            self._articulation.update(low_state)

        state = self._build_robot_state(low_state, roll, pitch, yaw, stair,
                                        body_speed, body_fwd, base_z, depth)

        # --- per-stage wall-ms (perf_counter deltas only -- no I/O on the hot path) ---
        _t = time.perf_counter()
        masked_depth = preprocess_depth(depth, cmd.person_bbox) if depth is not None else None
        preprocess_ms = (time.perf_counter() - _t) * 1000.0

        _t = time.perf_counter()
        out = self._runner.step(
            self._articulation, cmd,
            # NOTE: dt is the NOMINAL 1/control_hz on purpose. The frozen policies were
            # tuned at a fixed control dt; feeding the MEASURED (jittery) dt could
            # destabilize them. Jitter is only OBSERVED (tick_dt_ms), never fed back.
            depth_106x60=masked_depth, height_fn=self._make_height_fn(heightscan),
            state=state, dt=1.0 / self._control_hz, now=now,
        )
        policy_ms = (time.perf_counter() - _t) * 1000.0

        # Joint-limit watchdog against THIS tick's computed targets (before they publish).
        # The limits were injected at construction, so this path is now live: an out-of-range
        # target latch-damps instead of driving the motors past their envelope. Runs only in
        # walk mode -- the blind-RL climb legitimately commands deep calf angles the flat
        # envelope would reject, and it has its own motor-side clamp.
        if not self._runner.in_climb:
            jverdict = self._watchdog.evaluate(
                now=now_mono, last_state_ts=ts, roll=roll, pitch=pitch,
                targets=out.targets_isaac,
            )
            if not jverdict.ok and jverdict.reason == "joint_limit":
                self._enter_condition("joint_limit", latching=True)
                self._damp_latched = True
                self._damp_latch_reason = "joint_limit"
                self._publish_damping()
                return

        # Gain ramp on a fault resume: ramp kp/kd from damping up to the policy's nominal
        # gains over ``resume_ramp_sec`` so a re-arm stands the dog up gently instead of
        # snapping to full gain against a just-recovered pose.
        kp, kd = float(out.kp), float(out.kd)
        if self._resume_ramp_t0 is not None:
            frac = (now_mono - self._resume_ramp_t0) / max(1e-3, self._resume_ramp_sec)
            if frac >= 1.0:
                self._resume_ramp_t0 = None
            else:
                frac = max(0.0, frac)
                kp = frac * kp           # from 0 (damping) up to nominal
                kd = _DAMP_KD + frac * (kd - _DAMP_KD)

        _t = time.perf_counter()
        fields = build_low_cmd_fields(out.targets_isaac, kp, kd)
        self._publish_lowcmd(fields)
        publish_ms = (time.perf_counter() - _t) * 1000.0

        if self._tel is not None:
            self._tel_count += 1
            if self._tel_count % 5 == 0:   # ~10 Hz from the 50 Hz loop
                self._tel.record_fall_diag(
                    pitch_deg=math.degrees(pitch), roll_deg=math.degrees(roll),
                    policy_cmd=[cmd.vx, 0.0, cmd.wz],
                    action_norm=out.telemetry.get("action_norm"),
                )
                # Per-stage latency evidence (matches the sim's frame_timing stage_ms).
                # tick_total captured LAST so it covers this tick's real work; the JSONL
                # write itself is outside the 20 ms budget concern (only ~10 Hz).
                tick_total_ms = (time.perf_counter() - tick_t0) * 1000.0
                self._tel.record_frame_timing(
                    stage_ms={
                        "preprocess": preprocess_ms,
                        "policy": policy_ms,
                        "publish": publish_ms,
                        "tick_total": tick_total_ms,
                    },
                    tick_dt_ms=tick_dt_meas_ms,
                )

    def _build_robot_state(self, low_state, roll, pitch, yaw, stair,
                           body_speed=None, body_fwd=None, base_z=None, depth=None) -> RobotState:
        gyro = getattr(low_state.imu_state, "gyroscope", None) or getattr(low_state.imu_state, "gyro", None)
        roll_rate = float(gyro[0]) if gyro is not None else 0.0
        pitch_rate = float(gyro[1]) if gyro is not None else 0.0
        # riser_dist_ahead: prefer the LiDAR heightscan node's forward riser distance
        # (lidar mode). In FLAT heightscan mode nothing publishes it, so approach_engage
        # could never fire; fall back to the D435 DepthStairDetector's leading edge so the
        # climb handoff is reachable without LiDAR. body_speed/body_fwd/base_z now come from
        # SportModeState (arms stall_engage + the vertical-progress watchdog); absent -> None
        # and the handoff disables the guard that needs them.
        riser = stair.get("riser_dist_ahead")
        if riser is None and depth is not None:
            riser = self._depth_riser_dist_ahead(depth)
        return RobotState(
            roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate,
            yaw=yaw, riser_dist_ahead=riser,
            body_speed=body_speed, body_fwd=body_fwd, base_z=base_z,
        )

    def _depth_riser_dist_ahead(self, depth) -> Optional[float]:
        """FLAT-mode fallback for ``riser_dist_ahead``: the D435 DepthStairDetector's
        leading-edge distance (the first riser the dog faces). Lets ``approach_engage``
        fire without LiDAR. Detector is lazily built + reused; exceptions -> None (guard off).
        """
        try:
            if self._depth_stair_detector is None:
                from go2_locomotion.handoff_detectors import DepthStairDetector
                from go2_locomotion.handoff_config import HandoffConfig
                self._depth_stair_detector = DepthStairDetector(HandoffConfig())
            det = self._depth_stair_detector.detect(depth)
            le = det.get("leading_edge_distance")
            return float(le) if le is not None else None
        except Exception:
            return None

    def _make_height_fn(self, heightscan):
        """Build PGTT's height_fn from the latest /go2/heightscan grid, else flat.

        When the LiDAR heightscan node is running (``heightscan_mode: lidar``) it
        publishes the 99-cell body-frame elevation grid; we rebuild PGTT's height_fn
        from it here. With no heightscan (flat bring-up mode) PGTT walks blind-flat.
        """
        if heightscan is not None and len(heightscan) >= PGTT_N_POINTS:
            return height_fn_from_grid(heightscan)
        return lambda x, y: 0.0

    # --------------------------------------------------------------- publishing
    def _publish_lowcmd(self, fields) -> None:
        msg = LowCmd()
        for i in range(N_CMD_SLOTS):
            m = msg.motor_cmd[i]
            m.mode = int(fields.mode[i])
            m.q = float(fields.q[i])
            m.dq = float(fields.dq[i])
            m.kp = float(fields.kp[i])
            m.kd = float(fields.kd[i])
            m.tau = float(fields.tau[i])
        msg.crc = self._finalize_crc(msg)
        self._lowcmd_pub.publish(msg)

    def _publish_damping(self) -> None:
        self._publish_lowcmd(build_damping_fields())

    def _finalize_crc(self, low_cmd) -> int:
        """CRC32 over the serialized LowCmd (the MCU rejects a missing/wrong CRC).

        VALIDATE ON HARDWARE via the preflight roundtrip: the uint32 serialization
        layout below must match the firmware's struct order exactly. crc32_core is the
        correct Unitree algorithm; only the field ordering here is the risk. If the
        preflight CRC check fails, fix the serialization order here (or set the node's
        ``use_sdk_crc`` to fall back to unitree_sdk2py's checksum on the robot).
        """
        words = []
        for i in range(N_CMD_SLOTS):
            m = low_cmd.motor_cmd[i]
            words.append(int(m.mode) & 0xFFFFFFFF)
            for f in (m.q, m.dq, m.kp, m.kd, m.tau):
                words.append(int(np.float32(f).view(np.uint32)))
        return crc32_core(words)

    # ------------------------------------------------------------------- helpers
    def _rpy(self, low_state):
        q = getattr(low_state.imu_state, "quaternion", None)
        if q is None or len(q) < 4:
            return 0.0, 0.0, 0.0
        return _quat_to_rpy(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _enter_condition(self, condition: str, *, latching: bool = False) -> None:
        """Log a control-condition transition once per edge and thread it into exit_reason.

        ``condition`` is one of "ok" / "cmd_hold" / "cmd_lost" / a watchdog reason. A
        ``latching`` fault (tilt/stale from the watchdog) freezes ``_exit_reason`` -- it
        is the run's outcome and a later clean shutdown must not overwrite it to
        "completed". Non-latching conditions update ``_exit_reason`` live so the run
        summary reflects the state at shutdown (e.g. ended mid-HOLD -> "cmd_hold").
        """
        if condition == self._condition:
            return
        self._condition = condition
        if condition == "ok":
            self.get_logger().info("control OK: follow command fresh, watchdog clear")
        elif condition == "cmd_hold":
            self.get_logger().warning(
                "follow command STALE -> HOLD (zero velocity, balancing in place; "
                "keeping current walk/climb mode)"
            )
        else:
            self.get_logger().error(f"control FAULT -> {condition}: damping")
        if self._latched_fault:
            return
        if latching:
            self._exit_reason, self._latched_fault = condition, True
        elif condition == "ok":
            self._exit_reason = "completed"
        else:
            self._exit_reason = condition

    def _default_telemetry_dir(self) -> str:
        """Auto-derive ``run_logs/real/real_run_<ts>`` under the repo root when telemetry
        is enabled but no dir was configured, so the flight recorder is never silently off.
        """
        here = os.path.abspath(__file__)   # .../src/real/ros2/low_level_control_node.py
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(repo_root, "run_logs", "real", f"real_run_{ts}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LowLevelControlNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish_damping()
        except Exception:
            pass
        try:
            if node._tel is not None:
                # Honest outcome: the watchdog/staleness reason if one tripped, else
                # "completed". Never hard-code "completed" over a real fault.
                node._tel.finish(exit_reason=getattr(node, "_exit_reason", "completed"))
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
