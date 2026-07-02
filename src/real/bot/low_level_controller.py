"""Low-level motor controller interface for Unitree Go2.

Subscribes to `/rt/lowstate` and publishes commands to `/rt/lowcmd` at 50Hz.
Implements safety watchdogs (timeout, roll/pitch thresholds) to prevent hardware damage.
"""

import logging
import threading
import time
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

LOGGER = logging.getLogger("cable.vision.low_level_controller")


class LowLevelController:
    """Manages low-level DDS communication and joint torque/position commands."""

    def __init__(self, network_interface: str = "eth0", timeout_sec: float = 10.0):
        self.network_interface = network_interface
        self.timeout = timeout_sec
        self.state_sub: Optional[ChannelSubscriber] = None
        self.cmd_pub: Optional[ChannelPublisher] = None
        self.latest_state: Optional[LowState_] = None
        self.state_lock = threading.Lock()
        self.crc = CRC()
        self.is_initialized = False
        self.last_state_ts = 0.0

    def initialize(self) -> bool:
        """Initialize DDS connection and setup subscriber/publisher."""
        try:
            LOGGER.info(
                f"Initializing low-level controller on interface {self.network_interface}..."
            )
            # DDS Channel Initialize
            ChannelFactoryInitialize(0, self.network_interface)

            # Subscribe to low state
            self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
            self.state_sub.Init(self._state_callback, 10)

            # Create command publisher
            self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.cmd_pub.Init()

            self.is_initialized = True
            LOGGER.info("Low-level controller initialized successfully")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to initialize low-level DDS channel: {e}")
            self.is_initialized = False
            return False

    def _state_callback(self, msg: LowState_):
        with self.state_lock:
            self.latest_state = msg
            self.last_state_ts = time.monotonic()

    def get_state(self) -> Optional[LowState_]:
        """Get the latest cached LowState message if it is fresh."""
        with self.state_lock:
            if self.latest_state is None:
                return None
            # Watchdog check: if state is older than 0.25 seconds, mark as stale/None
            if (time.monotonic() - self.last_state_ts) > 0.25:
                LOGGER.warning("Low-level state is stale!")
                return None
            return self.latest_state

    def send_joint_commands(self, target_positions, kp=40.0, kd=1.0) -> bool:
        """Send joint angles targets (explicit-PD) to the 12 leg joints.

        Args:
            target_positions: Array of 12 joint targets in policy order.
            kp: Proportional stiffness gain (N.m/rad).
            kd: Derivative damping gain (N.m/(rad/s)).
        """
        if not self.is_initialized or self.cmd_pub is None:
            LOGGER.error("Low-level controller not initialized")
            return False

        # Check safety before sending commands
        state = self.get_state()
        if state is None:
            LOGGER.error("Safety watchdog triggered: LowState lost or stale. Damping motors!")
            self.safety_shutdown()
            return False

        # Pitch/Roll safety checks
        roll, pitch = state.imu_state.rpy[0], state.imu_state.rpy[1]
        max_tilt = max(abs(roll), abs(pitch))
        if max_tilt > 0.52:  # ~30 degrees safety threshold
            LOGGER.error(f"Safety watchdog triggered: Tilt ({max_tilt:.3f} rad) exceeds limit. Damping motors!")
            self.safety_shutdown()
            return False

        try:
            cmd = unitree_go_msg_dds__LowCmd_()

            # Set leg joints (0-11)
            for i in range(12):
                cmd.motor_cmd[i].mode = 0x01  # Active servo control
                cmd.motor_cmd[i].q = float(target_positions[i])
                cmd.motor_cmd[i].kp = float(kp)
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = float(kd)
                cmd.motor_cmd[i].tau = 0.0

            # Set remaining motors to passive/stop mode
            for i in range(12, 20):
                cmd.motor_cmd[i].mode = 0x00
                cmd.motor_cmd[i].q = 0.0
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = 0.0
                cmd.motor_cmd[i].tau = 0.0

            # Calculate and set CRC
            cmd.crc = self.crc.Crc(cmd)

            # Publish LowCmd
            return self.cmd_pub.Write(cmd)

        except Exception as e:
            LOGGER.error(f"Failed to publish LowCmd: {e}")
            self.safety_shutdown()
            return False

    def safety_shutdown(self) -> None:
        """Safely disable all motor torques by sending zero gains and modes."""
        if not self.is_initialized or self.cmd_pub is None:
            return

        try:
            cmd = unitree_go_msg_dds__LowCmd_()
            for i in range(20):
                cmd.motor_cmd[i].mode = 0x00  # Damping mode
                cmd.motor_cmd[i].q = 0.0
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = 0.0
                cmd.motor_cmd[i].tau = 0.0
            cmd.crc = self.crc.Crc(cmd)
            self.cmd_pub.Write(cmd)
            LOGGER.warning("Emergency low-level motor damping command sent successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to send safety shutdown: {e}")

    def shutdown(self) -> None:
        """Close DDS connections and safely damp motors."""
        if self.is_initialized:
            self.safety_shutdown()
            self.is_initialized = False
            LOGGER.info("Low-level controller shut down")
