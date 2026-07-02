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
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
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
from real.control.follow_command import FollowCommand, FOLLOW_CMD_TOPIC, DEPTH_TOPIC
from real.perception.heightscan_provider import height_fn_from_grid
from real.perception.depth_to_policy import preprocess as preprocess_depth
from real.logging.real_telemetry import RealTelemetry
from real.ros2.qos import sensor_qos, reliable_qos
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

        # ---- parameters (override via ros2 launch / real_robot.yaml) -------------
        p = self.declare_parameter
        self._lowcmd_topic = p("lowcmd_topic", "/lowcmd").value
        lowstate_topic = p("lowstate_topic", "/lowstate").value
        sport_state_topic = p("sport_state_topic", "/go2/sport_state").value
        stair_topic = p("stair_topic", "/go2/stair_detection").value
        heightscan_topic = p("heightscan_topic", "/go2/heightscan").value
        self._control_hz = float(p("control_hz", 50.0).value)
        self._require_released = bool(p("require_sport_released", True).value)
        self._climb_backend = str(p("climb_backend", "blind_rl").value)
        # Defaults point at the existing shared weights; the robot's launch/real_robot.yaml
        # overrides these to the on-device copies under real/models/.
        self._pgtt_path = str(p("pgtt_policy_path", "sim/models/pgtt/pgtt_go2_level17.npz").value)
        self._rl_path = str(p("rl_policy_path", "sim/models/locomotion/go2_robot_lab_policy.pt").value)
        walk_kp = float(p("walk_kp", 40.0).value)
        walk_kd = float(p("walk_kd", 0.5).value)
        climb_kp = float(p("climb_kp", 20.0).value)
        climb_kd = float(p("climb_kd", 0.5).value)
        self._record_telemetry = bool(p("record_telemetry", False).value)
        telemetry_dir = str(p("telemetry_dir", "").value)

        # ---- shared state (written by callbacks, read by the 50 Hz tick) ---------
        self._lock = threading.Lock()
        self._low_state = None
        self._low_state_ts: Optional[float] = None
        self._cmd = FollowCommand()
        self._depth_m: Optional[np.ndarray] = None
        self._heightscan: Optional[np.ndarray] = None
        self._stair = {}
        self._sport_released = not self._require_released
        self._articulation: Optional[LowStateArticulation] = None

        # ---- policies + runner + watchdog ---------------------------------------
        self._runner = self._build_runner(walk_kp, walk_kd, climb_kp, climb_kd)
        self._watchdog = SafetyWatchdog()

        # Tick-period jitter measurement (P: the loop feeds a HARDCODED nominal dt to
        # runner.step; these measure the ACTUAL wall period so Jetson jitter is visible
        # in the trace. Observability only -- the dt fed to the policy stays nominal).
        self._last_tick_ts: Optional[float] = None
        self._nominal_tick_s = 1.0 / max(1.0, self._control_hz)
        self._jitter_warn_last_ts: float = 0.0

        # Optional run telemetry (throttled to ~10 Hz) in the perf_tracker run-dir layout.
        self._tel = None
        self._tel_count = 0
        if self._record_telemetry and telemetry_dir:
            self._tel = RealTelemetry(telemetry_dir)
            self._tel.start(timestamp=datetime.now().isoformat(timespec="seconds"),
                            command="real/ros2/low_level_control_node.py --ros2")

        # ---- ROS I/O -------------------------------------------------------------
        self._lowcmd_pub = self.create_publisher(LowCmd, self._lowcmd_topic, reliable_qos())
        self.create_subscription(LowState, lowstate_topic, self._on_lowstate, sensor_qos(), callback_group=g)
        self.create_subscription(Float32MultiArray, FOLLOW_CMD_TOPIC, self._on_cmd, sensor_qos(), callback_group=g)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, sensor_qos(), callback_group=g)
        self.create_subscription(Float32MultiArray, heightscan_topic, self._on_heightscan, sensor_qos(), callback_group=g)
        self.create_subscription(Float32MultiArray, stair_topic, self._on_stair, sensor_qos(), callback_group=g)
        self.create_subscription(String, sport_state_topic, self._on_sport_state, sensor_qos(), callback_group=g)
        self.create_timer(1.0 / max(1.0, self._control_hz), self._control_tick, callback_group=g)
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
            except Exception:
                pass

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
            self._publish_damping()
            return

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
                node._tel.finish(exit_reason="completed")
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
