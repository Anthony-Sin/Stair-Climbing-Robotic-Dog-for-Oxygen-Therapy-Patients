"""
Robot controller interface for Unitree robots.

Handles high-level sport mode commands and can dynamically switch to low-level joint
PD control using the custom perceptive locomotion policy when stairs are encountered.
"""

import logging
import threading
import time
from typing import Optional

try:
    from unitree_sdk2py.go2.sport.sport_client import SportClient
except ImportError:
    SportClient = None

from core.telemetry.structured_logging import build_ecs_extra

LOGGER = logging.getLogger("cable.vision.robot_controller")


class RobotController:
    """
    Handles robot control interface and movement commands, with dynamic switching
    between high-level sport mode and low-level joint policy control.
    """
    
    def __init__(self, network_interface: str = 'eth0', timeout: float = 10.0,
                 low_level_locomotion: bool = False,
                 base_model_path: str = "src/sim/isaac/assets/policies/parkour/base_jit.pt",
                 vision_model_path: str = "src/sim/isaac/assets/policies/parkour/vision_weight.pt"):
        self.network_interface = network_interface
        self.timeout = timeout
        self.low_level_locomotion = low_level_locomotion
        self.base_model_path = base_model_path
        self.vision_model_path = vision_model_path
        
        self.sport_client: Optional[SportClient] = None
        self.is_initialized = False
        
        # Low-level control state
        self.low_level_controller = None
        self.policy = None
        self.motion_switcher = None
        self.in_low_level = False
        
        # Background loop variables
        self.low_level_loop_running = False
        self.low_level_thread: Optional[threading.Thread] = None
        self.low_level_lock = threading.Lock()
        
        # Shared variables for 50Hz low-level loop
        self.low_level_vx = 0.0
        self.low_level_yaw_err = 0.0
        self.low_level_stairs_active = False
        self.low_level_hold = False
        self.low_level_depth_img = None
        self.low_level_person_bbox = None
        
    def initialize(self) -> bool:
        """
        Initialize robot connection and put robot in ready state
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            LOGGER.info(
                "Robot controller initialize requested",
                extra=build_ecs_extra(
                    component="vision.robot_controller",
                    action="initialize_start",
                    cable={
                        "robot": {
                            "network_interface": self.network_interface,
                            "timeout_sec": self.timeout,
                        }
                    },
                ),
            )
            print("WARNING: Please ensure there are no obstacles around the robot while running person following mode.")
            input("Press Enter to continue...")
            
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            ChannelFactoryInitialize(0, self.network_interface)
            
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(self.timeout)
            self.sport_client.Init()
            
            print("Standing up robot...")
            self.sport_client.StandUp()
            time.sleep(2)  # Wait for robot to stand up
            
            print("Switching to balance stand mode...")
            self.sport_client.BalanceStand()
            time.sleep(1)  # Wait for balance mode to take effect
            
            if self.low_level_locomotion:
                # The sdk2/parkour low-level path (LowLevelController + the real-side
                # ParkourLocomotionPolicy fork) was a drifting, import-broken dead chain and
                # has been REMOVED (see the src/ refactor incident ledger). The supported
                # native low-level runtime is the ROS 2 stack (real/ros2/low_level_control_node,
                # selected by --ros2, which is the default). Fail fast instead of pretending.
                raise RuntimeError(
                    "low_level_locomotion=True is no longer supported here: the legacy "
                    "sdk2/parkour low-level chain was removed. Use the native ROS 2 stack "
                    "(--ros2 -> real/ros2/low_level_control_node). See src/real/DEPLOY.md."
                )

            self.is_initialized = True
            print("Robot controller initialized successfully")
            LOGGER.info(
                "Robot controller initialized",
                extra=build_ecs_extra(
                    component="vision.robot_controller",
                    action="initialize_success",
                    cable={
                        "robot": {
                            "network_interface": self.network_interface,
                            "timeout_sec": self.timeout,
                        }
                    },
                ),
            )
            return True
            
        except Exception as e:
            print(f"Failed to initialize robot controller: {e}")
            LOGGER.error(
                "Robot controller initialization failed",
                extra=build_ecs_extra(
                    component="vision.robot_controller",
                    action="initialize_failed",
                    cable={"error": {"message": str(e)}},
                ),
            )
            self.is_initialized = False
            return False
    
    def move(self, trans_x: float, trans_y: float, rotation: float, stairs_detected: bool = False,
             yaw_err: float = 0.0, **kwargs) -> bool:
        """
        Send movement command to robot (or update active joint control variables if low-level)
        """
        if not self.is_initialized or self.sport_client is None:
            print("Robot controller not initialized")
            return False
            
        # The low-level (sdk2/parkour) locomotion path was removed (see initialize()); this
        # controller is now high-level sport-mode Move only. The native low-level joint stack
        # is real/ros2/low_level_control_node, selected via --ros2 (the default).
        self.sport_client.Move(trans_x, trans_y, rotation)
        return True

    def stop(self) -> bool:
        """
        Stop robot movement
        
        Returns:
            True if stop command sent successfully, False otherwise
        """
        return self.move(0.0, 0.0, 0.0)
    
    def shutdown(self) -> bool:
        """
        Safely shutdown robot and put it in rest position
        
        Returns:
            True if shutdown successful, False otherwise
        """
        if not self.is_initialized:
            return True
        
        try:
            self.low_level_loop_running = False
            if self.low_level_thread is not None:
                self.low_level_thread.join(timeout=1.0)
                self.low_level_thread = None
                
            if self.low_level_controller is not None:
                self.low_level_controller.shutdown()
                
            if self.in_low_level and self.motion_switcher is not None:
                self.motion_switcher.SelectMode("ai")
                self.in_low_level = False
                
            if self.sport_client is not None:
                print("Stopping robot movement...")
                self.sport_client.StopMove()
                time.sleep(0.5)
                
                print("Sitting down robot...")
                self.sport_client.StandDown()
            
            self.is_initialized = False
            print("Robot controller shutdown successfully")
            LOGGER.info(
                "Robot controller shutdown completed",
                extra=build_ecs_extra(
                    component="vision.robot_controller",
                    action="shutdown_success",
                ),
            )
            return True
            
        except Exception as e:
            print(f"Failed to shutdown robot controller: {e}")
            LOGGER.error(
                "Robot controller shutdown failed",
                extra=build_ecs_extra(
                    component="vision.robot_controller",
                    action="shutdown_failed",
                    cable={"error": {"message": str(e)}},
                ),
            )
            return False
    
    def is_ready(self) -> bool:
        """Check if robot controller is ready for commands"""
        return self.is_initialized and (self.sport_client is not None)
