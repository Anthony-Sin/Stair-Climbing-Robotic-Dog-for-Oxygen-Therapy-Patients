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

# Native Unitree ROS 2 messages (unitree_ros2 driver).
from unitree_go.msg import LowState, LowCmd

from real.control.lowstate_articulation import LowStateArticulation, GO2_DOF_NAMES
from real.control.dual_policy_runner import DualPolicyRunner, RobotState
from real.control.lowcmd_builder import (
    build_low_cmd_fields, build_damping_fields, crc32_core, N_CMD_SLOTS,
)
from real.control.safety_watchdog import SafetyWatchdog
from real.control.command_gate import classify_command_age, FRESH, HOLD, DAMP
from real.control.follow_command import FollowCommand, FOLLOW_CMD_TOPIC, DEPTH_TOPIC
from real.perception.heightscan_provider import height_fn_from_grid
from real.perception.depth_to_policy import preprocess as preprocess_depth
from real.logging.real_telemetry import RealTelemetry
from real.ros2.qos import sensor_qos, reliable_qos, latched_qos
from go2_locomotion.pgtt_heightmap import PGTT_N_POINTS

from datetime import datetime


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
        self._lock = threading.Lock()
        self._low_state = None
        self._low_state_ts: Optional[float] = None
        self._cmd = FollowCommand()
        self._last_cmd_ts: Optional[float] = None   # receipt time of last follow command
        self._malformed_cmd_count = 0
        self._depth_m: Optional[np.ndarray] = None
        self._heightscan: Optional[np.ndarray] = None
        self._stair = {}
        self._sport_released = not self._require_released
        self._articulation: Optional[LowStateArticulation] = None

        # Control-condition tracking for honest logging + exit_reason. ``_condition`` is
        # the current edge-de-duplicated state ("ok"/"cmd_hold"/"cmd_lost"/watchdog
        # reason); ``_exit_reason`` is what finish() records; a latched watchdog fault
        # freezes it (a tilt/stale trip is the run's outcome, not a later clean exit).
        self._condition = ""
        self._exit_reason = "completed"
        self._latched_fault = False

        # ---- policies + runner + watchdog ---------------------------------------
        self._runner = self._build_runner(walk_kp, walk_kd, climb_kp, climb_kd)
        self._watchdog = SafetyWatchdog()

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
        # Latched QoS: sport_startup publishes "released" ONCE, latched. A VOLATILE/
        # BestEffort subscriber would never receive that latched history if it (re)starts
        # late -> it waits forever, robot flat, no error. Match the publisher's profile.
        self.create_subscription(String, sport_state_topic, self._on_sport_state, latched_qos(), callback_group=g)
        self.create_timer(1.0 / max(1.0, self._control_hz), self._control_tick, callback_group=ctrl_group)
        self.get_logger().info(f"low_level_control up @ {self._control_hz:.0f} Hz, backend={self._climb_backend}")

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
            self._low_state_ts = self._now()

    def _on_cmd(self, msg) -> None:
        with self._lock:
            try:
                self._cmd = FollowCommand.unpack(msg.data)
                self._last_cmd_ts = self._now()   # node-local staleness clock
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

    def _on_sport_state(self, msg) -> None:
        with self._lock:
            self._sport_released = (str(msg.data).strip().lower() == "released")

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
        now = self._now()

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
        verdict = self._watchdog.evaluate(
            now=now, last_state_ts=ts, roll=roll, pitch=pitch,
        )
        if not verdict.ok:
            # Tilt / joint-limit latch; a stale-lowstate gap is TRANSIENT and auto re-arms in
            # the watchdog, so don't freeze the run's exit_reason on it. Log once + damp.
            _latching = verdict.reason != "lowstate_stale"
            self._enter_condition(verdict.reason, latching=_latching)
            self._publish_damping()
            return

        # Command staleness: NEVER execute the last follow command forever. If vision
        # (process A) dies/lags, degrade gracefully -- HOLD (zero velocity, keep the
        # current walk/climb mode so we don't hand back mid-stair, balance in place),
        # then DAMP if it stays gone. classify_command_age is pure + unit-tested.
        cmd_age = float(now - last_cmd_ts) if last_cmd_ts is not None else float("inf")
        freshness = classify_command_age(
            age_sec=cmd_age, ever_received=last_cmd_ts is not None,
            timeout_sec=self._cmd_timeout_sec, damp_sec=self._cmd_damp_sec,
        )
        if freshness == DAMP:
            self._enter_condition("cmd_lost")
            self._publish_damping()
            return
        if freshness == HOLD:
            self._enter_condition("cmd_hold")
            cmd = replace(cmd, vx=0.0, wz=0.0, yaw_err=0.0, hold=True)
        else:
            self._enter_condition("ok")

        if self._articulation is None:
            self._articulation = LowStateArticulation(low_state)
        else:
            self._articulation.update(low_state)

        state = self._build_robot_state(low_state, roll, pitch, yaw, stair)

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

        _t = time.perf_counter()
        fields = build_low_cmd_fields(out.targets_isaac, out.kp, out.kd)
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

    def _build_robot_state(self, low_state, roll, pitch, yaw, stair) -> RobotState:
        gyro = getattr(low_state.imu_state, "gyroscope", None) or getattr(low_state.imu_state, "gyro", None)
        roll_rate = float(gyro[0]) if gyro is not None else 0.0
        pitch_rate = float(gyro[1]) if gyro is not None else 0.0
        # base_z / body velocity come from SportModeState if a node republishes it; left
        # at defaults (None) otherwise -- the handoff stall detector handles None.
        return RobotState(
            roll=roll, pitch=pitch, roll_rate=roll_rate, pitch_rate=pitch_rate,
            yaw=yaw, riser_dist_ahead=stair.get("riser_dist_ahead"),
        )

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
