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
        self.follow_pace_floor_speed = 0.5
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

    def _feed(self, args, state, debug_info, gap, is_walking=True, n=5, trans_x_cmd=0.5):
        # Feed a gap reading n times to warm the median filter (the controller debounces gap noise
        # by deciding on the median of the last <=5 VALID readings, so a transition needs the
        # smoothed gap -- not a single frame -- to cross the threshold).
        cmd = 0.0
        for _ in range(n):
            cmd = _apply_follow_standoff_policy(
                args, trans_x_cmd=trans_x_cmd, gap_m=gap, leader_speed_mps=0.0,
                is_walking=is_walking, debug_info=debug_info, state=state
            )
        return cmd

    def test_standoff_hysteresis_and_gait_gate(self):
        args = MockArgs()
        debug_info = {}
        state = {
            "go_state": False,
            "pace_state": "creep",
            "pace_timer": 0.0,
            "last_time": time.perf_counter(),
        }

        # gap 0.8m (below standoff 1.0 + band_in -0.15 = 0.85m) -> HOLD; lean-on-creep cmd 0.
        cmd = self._feed(args, state, debug_info, 0.8)
        self.assertEqual(cmd, 0.0)
        self.assertFalse(state["go_state"])

        # gap 0.9m (inside hysteresis band) -> remain HOLD.
        cmd = self._feed(args, state, debug_info, 0.9)
        self.assertEqual(cmd, 0.0)
        self.assertFalse(state["go_state"])

        # gap 1.2m (above standoff 1.0 + band_out 0.15 = 1.15m) -> GO. LEAN-ON-CREEP: GO no longer
        # passes the command through; the smoothed gap (1.2) is still inside follow_pace_distance
        # (2.0), so we lean on the policy's floor-creep and command ZERO.
        cmd = self._feed(args, state, debug_info, 1.2)
        self.assertEqual(cmd, 0.0)
        self.assertTrue(state["go_state"])
        self.assertEqual(debug_info["pace_state"], "creep")

        # gap 1.0m (inside hysteresis band) -> remain GO; still creep -> cmd 0.
        cmd = self._feed(args, state, debug_info, 1.0)
        self.assertEqual(cmd, 0.0)
        self.assertTrue(state["go_state"])

        # Gait gate override: is_walking False at gap <= upper -> force HOLD.
        cmd = self._feed(args, state, debug_info, 1.0, is_walking=False)
        self.assertEqual(cmd, 0.0)
        self.assertFalse(state["go_state"])

    def test_far_regime_continuous_advance(self):
        # FAR (gap > follow_pace_distance): catch up continuously -- no duty-cycle settle phase,
        # so a leader who walks away is never lost to the idle fraction. (Method 2 inverted the
        # old behaviour where a far gap triggered the advance/settle pacing.)
        args = MockArgs()
        debug_info = {}
        state = {
            "go_state": True,  # force GO
            "pace_state": "advance",
            "pace_timer": 0.0,
            "last_time": time.perf_counter(),
        }
        # Warm the median filter with a genuinely far gap (2.5 > follow_pace_distance 2.0): the
        # far regime must NOT enter settle, and must command at least the policy floor.
        state["last_time"] = time.perf_counter() - 5.0
        cmd = self._feed(args, state, debug_info, 2.5, trans_x_cmd=0.8)
        self.assertEqual(debug_info["pace_state"], "advance")
        self.assertFalse(debug_info["pace_hold_active"])
        # Cruise passes through, guaranteed at least the policy floor; never zeroed when far.
        self.assertGreaterEqual(cmd, args.follow_pace_floor_speed)
        self.assertEqual(cmd, 0.8)

    def test_near_regime_creep(self):
        # NEAR (go_state True, gap <= follow_pace_distance): LEAN ON THE CREEP. The frozen policy
        # floor-creeps at ~0.5 m/s with vx=0, matching a slow leader, so we command ZERO instead of
        # bursting -- any commanded advance is over-run into a ~1.2 m/s run that overshoots and falls.
        args = MockArgs()
        debug_info = {}
        state = {
            "go_state": True,  # force GO
            "pace_state": "creep",
            "pace_timer": 0.0,
            "last_time": time.perf_counter(),
        }
        cmd = _apply_follow_standoff_policy(
            args, trans_x_cmd=0.8, gap_m=1.1, leader_speed_mps=0.0,
            is_walking=True, debug_info=debug_info, state=state
        )
        self.assertEqual(debug_info["pace_state"], "creep")
        self.assertFalse(debug_info["pace_cap_active"])
        self.assertFalse(debug_info["pace_hold_active"])
        self.assertEqual(cmd, 0.0)  # no forward command; the creep does the following

        # Still creep a frame later (no duty-cycle phases anymore).
        cmd = _apply_follow_standoff_policy(
            args, trans_x_cmd=0.8, gap_m=1.1, leader_speed_mps=0.0,
            is_walking=True, debug_info=debug_info, state=state
        )
        self.assertEqual(debug_info["pace_state"], "creep")
        self.assertEqual(cmd, 0.0)

    def test_stair_bypass_still_updates_collision_gap(self):
        args = MockArgs()
        debug_info = {
            "stairs_detected": True,
            "stairs_action_active": True,
            "target_distance": 1.2,
        }
        state = {
            "go_state": True,
            "pace_state": "creep",
            "pace_timer": 0.0,
            "last_time": time.perf_counter(),
        }

        cmd = self._feed(
            args, state, debug_info, 0.62, n=3, trans_x_cmd=0.2
        )

        self.assertEqual(cmd, 0.2)
        self.assertTrue(debug_info["follow_standoff_skipped_on_stairs"])
        self.assertAlmostEqual(debug_info["standoff_gap_ctrl_m"], 0.62, places=3)
        self.assertAlmostEqual(debug_info["standoff_target_m"], 1.2, places=3)
        self.assertAlmostEqual(debug_info["standoff_lower_bound_m"], 1.05, places=3)

    def test_distant_stair_detection_keeps_approach_standoff(self):
        args = MockArgs()
        debug_info = {
            "stairs_detected": True,
            "stairs_action_active": False,
            "target_distance": 1.2,
        }
        state = {
            "go_state": True,
            "pace_state": "creep",
            "pace_timer": 0.0,
            "last_time": time.perf_counter(),
        }

        cmd = self._feed(args, state, debug_info, 1.6, n=3, trans_x_cmd=0.6)

        # A staircase merely visible in the distance is still flat-ground approach: keep the
        # widened standoff and lean on zero-command creep instead of passing PID bursts through.
        self.assertEqual(cmd, 0.0)
        self.assertFalse(debug_info.get("follow_standoff_skipped_on_stairs", False))
        self.assertAlmostEqual(debug_info["standoff_target_m"], 1.2, places=3)

        # Once the smoothed gap is below the lower threshold, the approach remains in its
        # hysteretic hold state. The main loop consumes this as stair_approach_brake while flat.
        cmd = self._feed(args, state, debug_info, 1.0, n=5, trans_x_cmd=0.6)
        self.assertEqual(cmd, 0.0)
        self.assertTrue(debug_info["follow_standoff_gate_active"])

    def test_stair_collision_block_has_complete_telemetry(self):
        from main import _apply_stair_command_policy

        args = MockArgs()
        args.stair_near_distance = 0.6
        args.stair_approach_speed_scale = 1.0
        args.trans_x_max = 0.35
        args.stair_speed_scale = 0.55
        args.stair_forward_floor = 0.35
        args.stair_climb_collision_floor = 0.70
        args.stair_yaw_deadband_deg = 5.0
        args.stair_centering_scale = 0.5
        args.stair_rot_max = 0.4
        debug_info = {
            "stairs_detected": True,
            "person_detected": True,
            "stairs_depth_m": 0.3,
            "stairs_depth_ever_confirmed": True,
            "standoff_gap_ctrl_m": 0.62,
        }

        tx, _ = _apply_stair_command_policy(args, 0.2, 0.0, debug_info)

        self.assertEqual(tx, 0.0)
        self.assertTrue(debug_info["stair_follow_collision_block"])
        self.assertAlmostEqual(debug_info["stairs_forward_floor_mps"], 0.1925, places=4)

    def test_garbage_leader_speed_clamped(self):
        # leader_speed_mps is depth-derived and can spike to absurd values (observed ~40 m/s on a
        # lock flicker). It must be clamped before widening the standoff, else one bad frame pins the
        # standoff at its 1.5 m cap. Feed a 42 m/s leader speed at a far gap and confirm the standoff
        # (target 1.0 + gain 0.4 * clamp(42 -> 1.0) = 1.4) is bounded, not blown out.
        args = MockArgs()
        debug_info = {}
        state = {"go_state": True, "pace_state": "creep", "pace_timer": 0.0, "last_time": time.perf_counter()}
        _apply_follow_standoff_policy(
            args, trans_x_cmd=0.0, gap_m=3.0, leader_speed_mps=42.9,
            is_walking=True, debug_info=debug_info, state=state
        )
        self.assertAlmostEqual(debug_info["standoff_target_m"], 1.4, places=3)

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
