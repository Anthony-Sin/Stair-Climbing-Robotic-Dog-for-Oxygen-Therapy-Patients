"""
Robot controller interface for Unitree robots.

Handles high-level sport mode commands and can dynamically switch to low-level joint
PD control using the custom perceptive locomotion policy when stairs are encountered.
"""

import logging
import threading
import time
from typing import Optional
import numpy as np

try:
    from unitree_sdk2py.go2.sport.sport_client import SportClient
except ImportError:
    SportClient = None

from core.telemetry.structured_logging import build_ecs_extra

LOGGER = logging.getLogger("cable.vision.robot_controller")


class PhysicalGo2Articulation:
    """Mock Isaac Sim articulation interface for the physical robot's LowState."""

    def __init__(self, low_state):
        self.low_state = low_state

    def get_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        # Position is not used by the policy. Quaternion is [w, x, y, z].
        quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)
        return (np.zeros(3, dtype=np.float32), quat)

    def get_angular_velocity(self) -> np.ndarray:
        # imu_state.gyro contains body-frame roll/pitch/yaw angular rates.
        # get_angular_velocity must return world-frame rates.
        # omega_world = rot @ gyro
        quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)
        from go2_locomotion_utils import quat_to_matrix
        rot = quat_to_matrix(quat)
        gyro = np.array(self.low_state.imu_state.gyro, dtype=np.float32)
        return rot @ gyro

    def get_joint_positions(self) -> np.ndarray:
        # Return joint positions in the SDK order (0-11)
        return np.array([self.low_state.motor_state[i].q for i in range(12)], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        # Return joint velocities in the SDK order (0-11)
        return np.array([self.low_state.motor_state[i].dq for i in range(12)], dtype=np.float32)

    def set_joint_efforts(self, efforts: np.ndarray) -> None:
        # No-op on the physical robot (target angles are sent to DDS LowCmd directly)
        pass


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
                LOGGER.info("Initializing low-level controller and perceptive policy...")
                from real.bot.low_level_controller import LowLevelController
                from real.bot.parkour_locomotion_policy import ParkourLocomotionPolicy, ParkourPolicyConfig
                from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
                
                self.low_level_controller = LowLevelController(self.network_interface)
                if not self.low_level_controller.initialize():
                    LOGGER.error("Failed to initialize low-level controller")
                    return False
                    
                # Define config & instantiate policy
                dof_names = [
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                ]
                cfg = ParkourPolicyConfig(
                    base_model_path=self.base_model_path,
                    vision_model_path=self.vision_model_path,
                    device="cpu"
                )
                self.policy = ParkourLocomotionPolicy(cfg, dof_names=dof_names)
                
                self.motion_switcher = MotionSwitcherClient()
                self.motion_switcher.Init()
                LOGGER.info("Low-level components successfully initialized.")

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
            
        stairs_action_active = kwargs.get("stairs_action_active", False)
        hold = kwargs.get("hold", False)
        depth_img = kwargs.get("depth_img")
        person_bbox = kwargs.get("person_bbox")

        if self.low_level_locomotion:
            if stairs_action_active:
                # Transition to low-level mode if not already there
                if not self.in_low_level:
                    LOGGER.info("STAIRS ENGAGED: Transitioning to low-level joint mode...")
                    self.sport_client.StopMove()
                    time.sleep(0.1)
                    
                    if self.motion_switcher is not None:
                        self.motion_switcher.ReleaseMode()
                        
                    self.policy.reset()
                    self.in_low_level = True
                    
                    # Start low-level background thread
                    self.low_level_loop_running = True
                    self.low_level_thread = threading.Thread(target=self._low_level_control_loop, daemon=True)
                    self.low_level_thread.start()
                    LOGGER.info("Transition to low-level control loop complete.")
                
                # Update inputs
                with self.low_level_lock:
                    self.low_level_vx = trans_x
                    self.low_level_yaw_err = yaw_err
                    self.low_level_stairs_active = stairs_action_active
                    self.low_level_hold = hold
                    self.low_level_depth_img = depth_img
                    self.low_level_person_bbox = person_bbox
                return True
            else:
                # Non-stair following. Transition back to high-level if needed
                if self.in_low_level:
                    LOGGER.info("STAIRS CLEARED: Transitioning back to high-level mode...")
                    self.low_level_loop_running = False
                    if self.low_level_thread is not None:
                        self.low_level_thread.join(timeout=1.0)
                        self.low_level_thread = None
                    
                    # Send safety damping command before enabling sport mode service
                    if self.low_level_controller is not None:
                        self.low_level_controller.safety_shutdown()
                    
                    # Select Mode "ai"
                    if self.motion_switcher is not None:
                        self.motion_switcher.SelectMode("ai")
                    
                    self.in_low_level = False
                    
                    time.sleep(1.0)
                    self.sport_client.BalanceStand()
                    time.sleep(0.5)
                    LOGGER.info("Transition to high-level mode complete.")
                
                self.sport_client.Move(trans_x, trans_y, rotation)
                return True
        else:
            self.sport_client.Move(trans_x, trans_y, rotation)
            return True
            
    def _low_level_control_loop(self):
        LOGGER.info("Low-level background thread started.")
        import cv2
        from parkour_depth_mask import mask_person_in_parkour_depth
        
        while self.low_level_loop_running:
            start_time = time.monotonic()
            
            with self.low_level_lock:
                vx = self.low_level_vx
                yaw_err = self.low_level_yaw_err
                stairs_active = self.low_level_stairs_active
                hold = self.low_level_hold
                depth_img = self.low_level_depth_img
                person_bbox = self.low_level_person_bbox
                
            low_state = self.low_level_controller.get_state()
            if low_state is not None:
                # 1. Resize and mask depth frame if available
                if depth_img is not None:
                    try:
                        depth_106x60 = cv2.resize(depth_img, (106, 60), interpolation=cv2.INTER_LINEAR)
                        masked_depth, _, _ = mask_person_in_parkour_depth(
                            depth_106x60, person_bbox, fill_mode="terrain"
                        )
                        self.policy.submit_depth(masked_depth)
                    except Exception as e:
                        LOGGER.error(f"Error preprocessing depth frame: {e}")
                
                # 2. Build mock articulation
                articulation = PhysicalGo2Articulation(low_state)
                
                # 3. Step locomotion policy at 50Hz
                dt = 0.02
                self.policy.step(
                    articulation,
                    cmd=[vx, 0.0, 0.0],
                    dt=dt,
                    delta_yaw=yaw_err,
                    stairs_active=stairs_active,
                    hold=hold
                )
                
                # 4. Publish joint commands
                targets = self.policy.last_targets_isaac
                self.low_level_controller.send_joint_commands(targets, kp=40.0, kd=1.0)
                
            elapsed = time.monotonic() - start_time
            sleep_time = max(0.001, 0.02 - elapsed)
            time.sleep(sleep_time)
            
        LOGGER.info("Low-level background thread stopped.")
    
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
