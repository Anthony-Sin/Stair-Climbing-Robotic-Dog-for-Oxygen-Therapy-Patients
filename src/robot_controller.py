"""
Robot controller interface for Unitree robots.

Robot Coordinate System:
- X-axis (trans_x): Forward(+) / Backward(-) movement
- Y-axis (trans_y): Left(+) / Right(-) movement  
- Rotation: Counter-clockwise(+) / Clockwise(-) rotation
"""

import logging
import time
from typing import Optional
from unitree_sdk2py.go2.sport.sport_client import SportClient

from structured_logging import build_ecs_extra


LOGGER = logging.getLogger("cable.vision.robot_controller")


class RobotController:
    """
    Handles robot control interface and movement commands
    """
    
    def __init__(self, network_interface: str = 'eth0', timeout: float = 10.0):
        self.network_interface = network_interface
        self.timeout = timeout
        self.sport_client: Optional[SportClient] = None
        self.is_initialized = False
        
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
    
    def move(self, trans_x: float, trans_y: float, rotation: float) -> bool:
        """
        Send movement command to robot
        
        Args:
            trans_x: X-axis translation velocity (-1.0 to 1.0) - forward(+)/backward(-) movement
            trans_y: Y-axis translation velocity (-1.0 to 1.0) - left(+)/right(-) movement 
            rotation: Rotation velocity (-1.0 to 1.0) - counter-clockwise(+)/clockwise(-) rotation
            
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.is_initialized or self.sport_client is None:
            print("Robot controller not initialized")
            LOGGER.warning(
                "Robot move requested while controller not initialized",
                extra=build_ecs_extra(
                    component="vision.robot_controller",
                    action="move_rejected",
                    cable={"robot": {"initialized": False}},
                ),
            )
            return False
        
        try:
            self.sport_client.Move(trans_x, trans_y, rotation)
            LOGGER.debug(
                "Robot move command sent",
                extra=build_ecs_extra(
                    component="vision.robot_controller",
                    action="move_command",
                    cable={
                        "robot": {
                            "command": {
                                "vx": float(trans_x),
                                "vy": float(trans_y),
                                "wz": float(rotation),
                            }
                        }
                    },
                ),
            )
            return True
        except Exception as e:
            print(f"Failed to send move command: {e}")
            LOGGER.error(
                "Robot move command failed",
                extra=build_ecs_extra(
                    component="vision.robot_controller",
                    action="move_failed",
                    cable={
                        "robot": {
                            "command": {
                                "vx": float(trans_x),
                                "vy": float(trans_y),
                                "wz": float(rotation),
                            }
                        },
                        "error": {"message": str(e)},
                    },
                ),
            )
            return False
    
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
        if not self.is_initialized or self.sport_client is None:
            return True
        
        try:
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
        return self.is_initialized and self.sport_client is not None
