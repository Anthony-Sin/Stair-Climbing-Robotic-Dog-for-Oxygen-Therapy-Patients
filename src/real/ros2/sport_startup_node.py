"""The ONE unitree_sdk2 node: release the built-in sport mode so /lowcmd is honored.

This is the single, isolated place the SDK is unavoidable (the port decision): the
Go2 boots into the MCU sport service, which fights /lowcmd until released via the
Motion Switcher. Everything else in the port is pure ROS 2; confining the SDK here
keeps it from leaking into the shared control logic.

Boot sequence (safe order, per Unitree guidance -- lay down BEFORE releasing so the
motors going passive does not drop a standing dog):
    StandUp -> BalanceStand -> StandDown -> MotionSwitcher.ReleaseMode() (loop until
    released) -> publish /go2/sport_state = "released".
The low-level control node waits for "released" before it writes /lowcmd, then PGTT
brings the dog back up from the folded pose (a gentle ramp is a HIL tuning item).

Shutdown restores the factory service: SelectMode("ai") -> BalanceStand -> StandDown.

/go2/sport_state is published with TRANSIENT_LOCAL (latched) QoS so a late-joining
low-level node still receives the last state.
"""
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# unitree_sdk2py: the only SDK import in the whole port.
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

# Shared latched profile -- the low-level control node subscribes with the SAME profile
# so a late/restarted control node still receives the "released" gate (see qos.py).
from real.ros2.qos import latched_qos


class SportStartupNode(Node):
    def __init__(self) -> None:
        super().__init__("sport_startup")
        self._iface = str(self.declare_parameter("network_interface", "eth0").value)
        self._release_on_start = bool(self.declare_parameter("release_on_start", True).value)
        self._timeout = float(self.declare_parameter("sdk_timeout_sec", 10.0).value)

        self._state_pub = self.create_publisher(String, "/go2/sport_state", latched_qos())
        self._publish_state("init")

        ChannelFactoryInitialize(0, self._iface)
        self._sport = SportClient()
        self._sport.SetTimeout(self._timeout)
        self._sport.Init()
        self._switcher = MotionSwitcherClient()
        self._switcher.SetTimeout(self._timeout)
        self._switcher.Init()
        self._released = False

        if self._release_on_start:
            # One-shot, slightly delayed so the latched state publisher + subscribers connect.
            self._boot_timer = self.create_timer(1.0, self._boot_once)

    # ------------------------------------------------------------------ sequence
    def _boot_once(self) -> None:
        self._boot_timer.cancel()
        try:
            self.get_logger().info("Sport startup: StandUp -> BalanceStand -> StandDown")
            self._sport.StandUp()
            time.sleep(2.0)
            self._sport.BalanceStand()
            time.sleep(1.0)
            self._publish_state("standing")
            # Lay down BEFORE releasing so passive motors do not drop a standing dog.
            self._sport.StandDown()
            time.sleep(1.5)
            self._release_sport_mode()
        except Exception as exc:  # pragma: no cover - hardware path
            self.get_logger().error(f"sport startup failed: {exc}")
            self._publish_state("error")

    def _release_sport_mode(self) -> None:
        """Loop ReleaseMode until the Motion Switcher reports no active mode."""
        for attempt in range(10):
            try:
                self._switcher.ReleaseMode()
            except Exception as exc:  # pragma: no cover
                self.get_logger().warning(f"ReleaseMode attempt {attempt} raised: {exc}")
            time.sleep(0.5)
            if self._mode_released():
                self._released = True
                self.get_logger().info("Sport mode released; /lowcmd is now honored")
                self._publish_state("released")
                return
        self.get_logger().error("Sport mode did NOT release after retries -- /lowcmd will be ignored")
        self._publish_state("release_failed")

    def _mode_released(self) -> bool:
        """True when CheckMode reports no active control mode (best-effort across SDK versions)."""
        try:
            code, data = self._switcher.CheckMode()
            name = (data or {}).get("name", "") if isinstance(data, dict) else ""
            return not bool(name)
        except Exception:
            # If CheckMode is unavailable on this firmware, assume the loop's releases took.
            return True

    # ------------------------------------------------------------------ teardown
    def restore_factory(self) -> None:
        """Hand control back to the built-in service (mirror of the boot release)."""
        try:
            self._switcher.SelectMode("ai")
            time.sleep(0.5)
            self._sport.BalanceStand()
            time.sleep(0.5)
            self._sport.StandDown()
            self._publish_state("ai")
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f"restore_factory failed: {exc}")

    # ------------------------------------------------------------------- helpers
    def _publish_state(self, state: str) -> None:
        self._state_pub.publish(String(data=state))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SportStartupNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_factory()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
