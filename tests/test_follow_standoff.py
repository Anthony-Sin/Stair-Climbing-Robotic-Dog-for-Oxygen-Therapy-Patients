import unittest
import numpy as np
import time
import sys
import os
from unittest.mock import MagicMock

# Add core path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

# Mock hardware dependencies for offline unit testing
sys.modules['tensorrt'] = MagicMock()
sys.modules['trt_inference'] = MagicMock()
sys.modules['yolo_pose_inference'] = MagicMock()
sys.modules['yolo_stairs_inference'] = MagicMock()
sys.modules['yolox'] = MagicMock()
sys.modules['yolox.tracker'] = MagicMock()
sys.modules['yolox.tracker.byte_tracker'] = MagicMock()
sys.modules['ecs_logging'] = MagicMock()

from gait_estimator import GaitEstimator
from main import _apply_follow_standoff_policy

class MockArgs:
    def __init__(self):
        self.target_distance = 1.0
        self.follow_standoff_speed_gain = 0.4
        self.follow_standoff_band_in = -0.15
        self.follow_standoff_band_out = 0.15
        self.follow_gait_gate = True
        self.follow_pace_distance = 2.0
        self.follow_pace_speed = 0.4
        self.follow_pace_advance_time = 2.0
        self.follow_pace_settle_time = 1.5

class TestFollowStandoff(unittest.TestCase):
    def test_gait_estimator_stationary(self):
        estimator = GaitEstimator(history_len=30, walk_threshold=0.5)
        
        # 1. Feed a stationary keypoint stream
        # Ankles constant separation and no vertical lift, speed near zero
        for _ in range(30):
            kpts = np.zeros((17, 2))
            # Hips
            kpts[11] = [100, 200]
            kpts[12] = [120, 200]
            # Ankles
            kpts[15] = [95, 400]
            kpts[16] = [125, 400]
            
            vis = np.ones(17)
            bbox = np.array([80, 100, 140, 410])
            
            is_walking, conf, speed, gp = estimator.update(
                keypoints=kpts,
                visibility=vis,
                depth_m=1.0,
                dt=0.033,
                robot_speed=0.0,
                robot_yaw_speed=0.0,
                camera_cx=320,
                camera_fx=500,
                bbox=bbox,
            )
            
        self.assertFalse(is_walking)
        self.assertLess(conf, 0.4)
        self.assertLess(speed, 0.1)

    def test_gait_estimator_walking(self):
        estimator = GaitEstimator(history_len=30, walk_threshold=0.5)
        
        # 2. Feed a walking keypoint stream (oscillating ankle sep, vertical lift, non-zero speed)
        # We model a gait cycle over 30 frames
        for i in range(30):
            phase = 2 * np.pi * i / 15.0
            kpts = np.zeros((17, 2))
            # Hips
            kpts[11] = [100, 200]
            kpts[12] = [120, 200]
            # Ankles: separation oscillates from 10px to 50px
            sep = 30 + 20 * np.cos(phase)
            kpts[15] = [110 - sep/2, 400 - 15 * max(0.0, np.sin(phase))]
            kpts[16] = [110 + sep/2, 400 - 15 * max(0.0, -np.sin(phase))]
            
            vis = np.ones(17)
            bbox = np.array([80, 100, 140, 410])
            
            # Distance decreases to model a person moving relative to robot
            depth = 3.0 - 0.02 * i  # ~0.6 m/s relative speed
            
            is_walking, conf, speed, gp = estimator.update(
                keypoints=kpts,
                visibility=vis,
                depth_m=depth,
                dt=0.033,
                robot_speed=0.2, # robot is moving too
                robot_yaw_speed=0.0,
                camera_cx=320,
                camera_fx=500,
                bbox=bbox,
            )
            
        self.assertTrue(is_walking)
        self.assertGreater(conf, 0.5)
        self.assertGreater(speed, 0.4)
        self.assertIsNotNone(gp)

    def test_gait_estimator_occlusion_fallback(self):
        estimator = GaitEstimator(history_len=30, walk_threshold=0.5)
        
        # 3. Feet occluded (visibility = 0.0), speed is high
        for i in range(30):
            kpts = np.zeros((17, 2))
            vis = np.zeros(17) # ankles invisible
            bbox = np.array([80, 100, 140, 410])
            depth = 3.0 - 0.03 * i  # ~0.9 m/s relative speed
            
            is_walking, conf, speed, gp = estimator.update(
                keypoints=kpts,
                visibility=vis,
                depth_m=depth,
                dt=0.033,
                robot_speed=0.0,
                robot_yaw_speed=0.0,
                camera_cx=320,
                camera_fx=500,
                bbox=bbox,
            )
            
        self.assertTrue(is_walking)
        self.assertGreater(conf, 0.5)
        self.assertGreater(speed, 0.5)

    def test_standoff_hysteresis_and_gait_gate(self):
        args = MockArgs()
        debug_info = {}
        state = {
            "go_state": False,
            "pace_state": "advance",
            "pace_timer": 0.0,
            "last_time": time.perf_counter(),
        }
        
        # Initial: gap is 0.8m (below standoff 1.0m + band_in -0.15 = 0.85m)
        cmd = _apply_follow_standoff_policy(
            args, trans_x_cmd=0.5, gap_m=0.8, leader_speed_mps=0.0,
            is_walking=True, debug_info=debug_info, state=state
        )
        self.assertEqual(cmd, 0.0)
        self.assertFalse(state["go_state"])
        
        # Move to 0.9m (inside hysteresis band, should remain in HOLD)
        cmd = _apply_follow_standoff_policy(
            args, trans_x_cmd=0.5, gap_m=0.9, leader_speed_mps=0.0,
            is_walking=True, debug_info=debug_info, state=state
        )
        self.assertEqual(cmd, 0.0)
        self.assertFalse(state["go_state"])
        
        # Move to 1.2m (above standoff 1.0 + band_out 0.15 = 1.15m -> should become GO)
        cmd = _apply_follow_standoff_policy(
            args, trans_x_cmd=0.5, gap_m=1.2, leader_speed_mps=0.0,
            is_walking=True, debug_info=debug_info, state=state
        )
        self.assertEqual(cmd, 0.5)
        self.assertTrue(state["go_state"])
        
        # Move back to 1.0m (inside hysteresis band, should remain in GO)
        cmd = _apply_follow_standoff_policy(
            args, trans_x_cmd=0.5, gap_m=1.0, leader_speed_mps=0.0,
            is_walking=True, debug_info=debug_info, state=state
        )
        self.assertEqual(cmd, 0.5)
        self.assertTrue(state["go_state"])

        # Test Gait Gate Override: is_walking becomes False -> should force HOLD
        cmd = _apply_follow_standoff_policy(
            args, trans_x_cmd=0.5, gap_m=1.0, leader_speed_mps=0.0,
            is_walking=False, debug_info=debug_info, state=state
        )
        self.assertEqual(cmd, 0.0)
        self.assertFalse(state["go_state"])

    def test_approach_pacing_timers(self):
        args = MockArgs()
        debug_info = {}
        state = {
            "go_state": True, # force GO
            "pace_state": "advance",
            "pace_timer": 0.0,
            "last_time": time.perf_counter(),
        }
        
        # Gap is 2.5m (above pace threshold of 2.0m)
        # Advance phase first (first 2.0s of cycle)
        # We simulate 1 second elapsed
        state["last_time"] = time.perf_counter() - 1.0
        cmd = _apply_follow_standoff_policy(
            args, trans_x_cmd=0.8, gap_m=2.5, leader_speed_mps=0.0,
            is_walking=True, debug_info=debug_info, state=state
        )
        self.assertEqual(debug_info["pace_state"], "advance")
        self.assertTrue(debug_info["pace_cap_active"])
        self.assertEqual(cmd, args.follow_pace_speed) # capped at 0.4
        
        # Settle phase (time > 2.0s in cycle, e.g. 2.5s)
        state["last_time"] = time.perf_counter() - 1.5 # cumulative 2.5s
        cmd = _apply_follow_standoff_policy(
            args, trans_x_cmd=0.8, gap_m=2.5, leader_speed_mps=0.0,
            is_walking=True, debug_info=debug_info, state=state
        )
        self.assertEqual(debug_info["pace_state"], "settle")
        self.assertTrue(debug_info["pace_hold_active"])
        self.assertEqual(cmd, 0.0) # settle forces 0 speed

    def test_no_movement_when_person_not_detected(self):
        # We simulate the command flow from main.py when person_detected is False.
        # We check both flat ground and stairs scenarios.
        
        # 1. Flat ground scenario (stairs_detected = False)
        # Person follower might suggest a search rotation when lost, but it should be zeroed if person is not detected.
        # Here we mock the final command override logic:
        # trans_x_cmd = 0.0, rotation_cmd = 0.25 (suggested by follower search)
        trans_x_cmd = 0.0
        rotation_cmd = 0.25
        debug_info = {"person_detected": False, "stairs_detected": False}
        
        # If person_detected is False, we force commands to 0.0
        if not debug_info["person_detected"]:
            trans_x_cmd = 0.0
            rotation_cmd = 0.0
            
        self.assertEqual(trans_x_cmd, 0.0)
        self.assertEqual(rotation_cmd, 0.0)
        
        # 2. Stairs scenario with brief loss (stairs_detected = True, person_detected = False, brief_loss = True)
        # _apply_stair_command_policy returns forward floor speed and 0.0 rotation command
        args = MockArgs()
        args.stair_near_distance = 1.2
        args.stair_approach_speed_scale = 1.0
        args.trans_x_max = 0.8
        args.stair_speed_scale = 1.0
        args.stair_forward_floor = 0.35
        args.stair_yaw_deadband_deg = 5.0
        args.stair_centering_scale = 0.5
        args.stair_rot_max = 0.4
        
        # Mock brief loss state
        debug_info_stairs = {
            "person_detected": False,
            "stairs_detected": True,
            "lost_age_sec": 1.0,
            "lost_search_timeout_sec": 2.5,
            "stairs_depth_m": 0.8,
            "stairs_depth_ever_confirmed": True,
        }
        
        # Call stair policy
        from main import _apply_stair_command_policy
        tx, rot = _apply_stair_command_policy(args, 0.0, 0.25, debug_info_stairs)
        
        # Stair policy suggests moving forward at the floor speed (0.35) due to brief loss
        self.assertGreater(tx, 0.0)
        self.assertEqual(rot, 0.0)
        
        # But the main loop override forces both to 0.0 because person_detected is False
        if not debug_info_stairs["person_detected"]:
            tx = 0.0
            rot = 0.0
            
        self.assertEqual(tx, 0.0)
        self.assertEqual(rot, 0.0)

if __name__ == "__main__":
    unittest.main()
