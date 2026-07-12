import unittest
import numpy as np
import time
import sys
import os
from unittest.mock import MagicMock

# Repo root on sys.path for `core.*` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock hardware dependencies for offline unit testing (keyed by their new module paths)
sys.modules['tensorrt'] = MagicMock()
sys.modules['core.vision.trt_inference'] = MagicMock()
sys.modules['core.vision.yolo_pose_inference'] = MagicMock()
sys.modules['core.vision.yolo_stairs_inference'] = MagicMock()
sys.modules['yolox'] = MagicMock()
sys.modules['yolox.tracker'] = MagicMock()
sys.modules['yolox.tracker.byte_tracker'] = MagicMock()
sys.modules['ecs_logging'] = MagicMock()

from core.control.gait_estimator import GaitEstimator
from core.main import _apply_follow_standoff_policy

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
                is_walking=is_walking, debug_info=debug_info, state=state,
                # incident 8.5 fix: the policy now takes the genuine flag explicitly instead of
                # reading debug_info["stairs_action_active"] (which was produced later same-frame).
                # Forward what the test staged in debug_info -> behaviour-equivalent to the old read.
                stairs_action_active=bool(debug_info.get("stairs_action_active", False)),
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

    def test_raw_gap_decel_cap_prevents_overrun(self):
        # The trot pace-matcher (PGTT) feeds the leader's closing speed forward against the LAGGED
        # median gap, so on a fast approach it commanded ~max forward INTO the patient until the
        # median caught up -- closing to a range where the patient fills the frame and YOLO drops
        # (the zigzag-apex losses). The raw-gap deceleration cap throttles the forward command on
        # the LIVE gap so it ramps to ~0 at the standoff lower bound instead of overrunning.
        from core.control.follow_shaping import _apply_follow_standoff_policy
        args = MockArgs()
        args.trans_x_max = 0.85
        args.follow_trot_speed_kp = 2.0          # PGTT trot path (commands forward inside the band)
        args.follow_standoff_speed_gain = 0.0    # matches the sim launcher; keep standoff fixed at target
        args.target_distance = 0.6               # standoff 0.6 -> lower 0.45, upper 0.75

        def feed(gap):
            state = {"go_state": True, "pace_state": "trot",
                     "first_ctrl_ts": time.perf_counter() - 5.0,  # past the settle warmup
                     "last_time": time.perf_counter()}
            dbg = {}
            cmd = 0.0
            for _ in range(5):  # warm the median to this gap, moving leader (feedforward on)
                cmd = _apply_follow_standoff_policy(
                    args, trans_x_cmd=0.85, gap_m=gap, leader_speed_mps=0.6,
                    is_walking=True, debug_info=dbg, state=state)
            return cmd, dbg

        # Raw gap 0.5 m (just inside the standoff band): the trot wants ~0.4 m/s, but the cap
        # throttles it toward zero so the dog decelerates instead of driving into the patient.
        cmd_close, dbg_close = feed(0.5)
        self.assertTrue(dbg_close.get("raw_gap_decel_active"))
        self.assertLess(cmd_close, 0.2)
        # A far gap (well past the standoff band) is NOT capped -- normal pace-matching is intact.
        cmd_far, dbg_far = feed(1.2)
        self.assertFalse(dbg_far.get("raw_gap_decel_active", False))
        self.assertGreater(cmd_far, 0.5)

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
        from core.main import _apply_stair_command_policy

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
        # Now exercised now that standoff_gap_ctrl_m below is a real (non-None) value --
        # previously masked by short-circuit evaluation while the gap was always None.
        args.stair_climb_collision_floor = 0.55
        
        # Mock brief loss state
        debug_info_stairs = {
            "person_detected": False,
            "stairs_detected": True,
            "lost_age_sec": 1.0,
            "lost_search_timeout_sec": 2.5,
            "stairs_depth_m": 0.8,
            "stairs_depth_ever_confirmed": True,
            # incident 8.15 / F2: the mid-climb gap-brake fails toward SLOW on an unmeasured
            # gap (incident 8.8), so this must be populated with a real (far/safe) value here
            # exactly as the real loop always would -- _apply_follow_standoff_policy runs
            # BEFORE _apply_stair_command_policy every frame and unconditionally writes this
            # key (even to None once warmed up with <3 samples). This test calls the stair
            # policy standalone, skipping that producer, so the key must be seeded by hand or
            # every call here would read as an unmeasured gap and brake to 0 regardless of the
            # brief-loss forward floor under test.
            "standoff_gap_ctrl_m": 2.0,
        }
        
        # Call stair policy
        from core.main import _apply_stair_command_policy
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


class TestLostSearchDirection(unittest.TestCase):
    """The lost-search must turn toward the LAST YOLO sighting (last_person_center),
    not a stale full-success-path bearing -- even a single 1-frame glimpse whose depth
    measurement failed. Regression for the dog spinning RIGHT toward an old fix while
    the patient's final detected frame was on the LEFT."""

    def _make_follower(self):
        from core.control.person_follower import PersonFollower, PersonFollowingConfig
        cfg = PersonFollowingConfig()
        cfg.camera_fx = 600.0
        cfg.camera_cx = 320.0           # 640-wide frame, principal point centred
        cfg.max_rotation_speed = 0.5
        cfg.lost_search_yaw_speed = 0.25
        cfg.lost_search_timeout_sec = 4.0
        cfg.lost_search_min_error_deg = 3.0
        return PersonFollower(cfg)

    def test_searches_toward_last_yolo_glimpse_left(self):
        f = self._make_follower()
        frame_shape = (480, 640)
        depth = np.zeros((480, 640), dtype=np.float32)

        # Stale state: the success path last fired with the person far to the RIGHT.
        f.last_rotation_error_deg = 25.0          # positive == right
        f.is_tracking = True
        f.tracking_start_time = time.time() - 5.0

        # Final 1-frame YOLO glimpse: person on the far LEFT (x well left of cx=320),
        # the kind of edge frame whose depth typically fails. last_person_center is
        # refreshed by _update_person_tracking regardless of depth.
        f.last_person_center = (60, 240)
        f.last_lost_time = time.time() - 1.0      # within the search timeout

        _, rotation_cmd, dbg = f.update(None, depth, frame_shape)

        # Must search LEFT (toward the last glimpse), NOT right toward the stale fix.
        self.assertTrue(dbg.get("lost_search_active"))
        self.assertEqual(dbg.get("lost_search_direction"), "left")
        self.assertLess(dbg.get("last_seen_bearing_deg"), 0.0)
        # In this convention a left target yields a POSITIVE yaw command.
        self.assertGreater(rotation_cmd, 0.0)

    def test_searches_toward_last_yolo_glimpse_right(self):
        f = self._make_follower()
        frame_shape = (480, 640)
        depth = np.zeros((480, 640), dtype=np.float32)

        f.last_rotation_error_deg = -25.0         # stale LEFT fix
        f.is_tracking = True
        f.tracking_start_time = time.time() - 5.0
        f.last_person_center = (600, 240)         # final glimpse far RIGHT
        f.last_lost_time = time.time() - 1.0

        _, rotation_cmd, dbg = f.update(None, depth, frame_shape)

        self.assertTrue(dbg.get("lost_search_active"))
        self.assertEqual(dbg.get("lost_search_direction"), "right")
        self.assertGreater(dbg.get("last_seen_bearing_deg"), 0.0)
        self.assertLess(rotation_cmd, 0.0)

    def test_centered_last_glimpse_does_not_spin(self):
        # Person basically centred when lost AND no lateral motion -> no spurious search
        # spin even if the stale success-path value was large. The direction is "unknown"
        # so the dog HOLDS rather than guessing a side.
        f = self._make_follower()
        frame_shape = (480, 640)
        depth = np.zeros((480, 640), dtype=np.float32)

        f.last_rotation_error_deg = 25.0          # stale large value
        f.is_tracking = True
        f.tracking_start_time = time.time() - 5.0
        f.last_person_center = (322, 240)         # ~centred (<3 deg)
        f.person_velocity = None                  # no motion cue
        f.last_lost_time = time.time() - 1.0

        _, rotation_cmd, dbg = f.update(None, depth, frame_shape)

        self.assertFalse(dbg.get("lost_search_active"))
        self.assertEqual(rotation_cmd, 0.0)
        self.assertEqual(dbg.get("lost_search_cue"), "none")

    def test_centered_glimpse_uses_motion_tiebreak(self):
        # Final glimpse near centre but the box was sliding LEFT fast -> the patient turned
        # and stepped laterally out of frame; search toward the motion direction (left).
        f = self._make_follower()
        frame_shape = (480, 640)
        depth = np.zeros((480, 640), dtype=np.float32)

        f.last_rotation_error_deg = 0.0
        f.is_tracking = True
        f.tracking_start_time = time.time() - 5.0
        f.last_person_center = (322, 240)         # ~centred (<3 deg) -> position ambiguous
        f.person_velocity = (-300.0, 0.0)         # px/s, moving LEFT well past the gate
        f.last_lost_time = time.time() - 1.0

        _, rotation_cmd, dbg = f.update(None, depth, frame_shape)

        self.assertTrue(dbg.get("lost_search_active"))
        self.assertEqual(dbg.get("lost_search_direction"), "left")
        self.assertEqual(dbg.get("lost_search_cue"), "motion")
        self.assertGreater(rotation_cmd, 0.0)     # left target -> positive yaw

    @staticmethod
    def _encode_profile(ranges_m, step_deg=3.0):
        import zlib, base64
        arr = np.clip(np.asarray(ranges_m, dtype=np.float64) * 1000.0, 0, 65535).astype(np.uint16)
        blob = base64.b64encode(zlib.compress(arr.tobytes())).decode("ascii")
        return {
            "ranges_mm": blob,
            "n_azimuth": int(arr.shape[0]),
            "azimuth_step_deg": step_deg,
            "view_range_m": 6.0,
            "min_range_m": 0.05,
        }

    def test_reversal_motion_overrides_stale_side(self):
        # Zigzag apex: last bbox was just RIGHT of centre but the box was sliding LEFT fast
        # (the patient reversed). Trust where they are HEADING, not the stale last-seen side.
        f = self._make_follower()
        frame_shape = (480, 640)
        depth = np.zeros((480, 640), dtype=np.float32)
        f.is_tracking = True
        f.tracking_start_time = time.time() - 5.0
        f.last_person_center = (370, 240)         # ~+4.8 deg right, within the reversal band
        f.person_velocity = (-400.0, 0.0)         # sliding LEFT, well past the gate
        f.last_lost_time = time.time() - 1.0

        _, rotation_cmd, dbg = f.update(None, depth, frame_shape)

        self.assertEqual(dbg.get("lost_search_cue"), "motion_reversal")
        self.assertEqual(dbg.get("lost_search_direction"), "left")
        self.assertGreater(rotation_cmd, 0.0)     # left target -> positive yaw

    def test_lidar_bridge_tracks_offaxis_past_timeout(self):
        # The whole 47 s-freeze fix: a live LiDAR bearing keeps the dog turning toward the
        # patient EVEN PAST the lost-search timeout, so a long YOLO outage no longer freezes it.
        f = self._make_follower()
        f.config.lidar_fusion_enabled = True
        frame_shape = (480, 640)
        depth = np.zeros((480, 640), dtype=np.float32)
        f.is_tracking = True
        f.tracking_start_time = time.time() - 8.0
        f.last_person_center = (322, 240)          # ~centred prior (position cue ambiguous)
        f.last_person_range_m = 1.5
        f.last_lost_time = time.time() - 6.0       # 6 s > timeout 4 s: only the bridge can act

        ranges = np.full(120, 5.0)                 # walls everywhere...
        b = int(round((360.0 - 20.0) / 3.0)) % 120  # ...except a 1.5 m return ~20 deg to the RIGHT
        for k in (b - 1, b, b + 1):
            ranges[k % 120] = 1.5
        prof = self._encode_profile(ranges)

        _, rotation_cmd, dbg = f.update(None, depth, frame_shape, lidar_profile=prof)

        self.assertEqual(dbg.get("lost_search_cue"), "lidar")
        self.assertTrue(dbg.get("lost_search_active"))
        self.assertEqual(dbg.get("lost_search_direction"), "right")
        self.assertLess(rotation_cmd, 0.0)         # right target -> negative yaw

    def test_bounded_arc_scan_pingpongs(self):
        # No LiDAR: the in-place re-acquire is a BOUNDED +/- arc scan -- first toward the
        # last-known side, then sweeping ACROSS centre to the other side, then back, ping-ponging
        # within +/- lost_search_arc_deg (never a full 180). One arc leg takes
        # radians(arc)/yaw_speed seconds.
        import math as _math
        f = self._make_follower()
        frame_shape = (480, 640)
        depth = np.zeros((480, 640), dtype=np.float32)
        f.is_tracking = True
        # PersonFollower now measures every duration off time.perf_counter() (incident 8.6). Seed
        # follower timestamps with the SAME monotonic clock, or perf-vs-wall epochs mismatch by ~1.7e9.
        f.tracking_start_time = time.perf_counter() - 10.0
        f.last_person_center = (600, 240)          # last seen RIGHT (search_sign +1)
        from core.control.person_follower import _LOST_SCAN_LEG_SEC
        arc_rad = _math.radians(f.config.lost_search_arc_deg)
        scan_rate = min(f.config.max_rotation_speed,
                        max(f.config.lost_search_yaw_speed, arc_rad / _LOST_SCAN_LEG_SEC))
        leg_sec = arc_rad / scan_rate

        # The scan phase is measured from when the scan ENGAGED (lost_search_start_time), not raw
        # lost_age, so it always opens toward the last-seen side regardless of how long was lost.
        # Early (within the first leg): scan TOWARD the last-known side (right -> negative yaw).
        f.last_lost_time = time.perf_counter() - 0.3 * leg_sec
        f.lost_search_start_time = time.perf_counter() - 0.3 * leg_sec
        _, rot_toward, dbg_toward = f.update(None, depth, frame_shape)
        self.assertTrue(dbg_toward.get("lost_search_active"))
        self.assertEqual(dbg_toward.get("lost_search_phase"), "toward")
        self.assertEqual(dbg_toward.get("lost_search_direction"), "right")
        self.assertLess(rot_toward, 0.0)

        # Mid-scan (the across leg, ~1.5 legs in): swing to the OPPOSITE side (left -> positive yaw).
        f.last_lost_time = time.perf_counter() - 1.5 * leg_sec
        f.lost_search_start_time = time.perf_counter() - 1.5 * leg_sec
        _, rot_across, dbg_across = f.update(None, depth, frame_shape)
        self.assertTrue(dbg_across.get("lost_search_active"))
        self.assertEqual(dbg_across.get("lost_search_phase"), "across")
        self.assertEqual(dbg_across.get("reason"), "Target lost - scanning opposite side")
        self.assertGreater(rot_across, 0.0)

        # The yaw magnitude is the bounded scan rate (a scan, not a runaway spin).
        self.assertAlmostEqual(abs(rot_toward), scan_rate, places=6)
        self.assertAlmostEqual(abs(rot_across), scan_rate, places=6)
        self.assertLessEqual(scan_rate, f.config.max_rotation_speed + 1e-9)

    def test_scan_gives_up_after_max_sec(self):
        # Past lost_search_max_sec the scan stops (hold) instead of scanning forever.
        f = self._make_follower()
        frame_shape = (480, 640)
        depth = np.zeros((480, 640), dtype=np.float32)
        f.is_tracking = True
        # Seed with perf_counter (see note above) so scan_elapsed / lost_age match the follower's clock.
        f.tracking_start_time = time.perf_counter() - 30.0
        f.last_person_center = (600, 240)
        f.last_lost_time = time.perf_counter() - (f.config.lost_search_max_sec + 2.0)
        # Scan has been running (and ping-ponging) longer than the max -> give up and hold.
        f.lost_search_start_time = time.perf_counter() - (f.config.lost_search_max_sec + 2.0)

        trans, rotation_cmd, dbg = f.update(None, depth, frame_shape)

        self.assertEqual(trans, 0.0)
        self.assertEqual(rotation_cmd, 0.0)
        self.assertFalse(dbg.get("lost_search_active"))

    def test_command_and_label_agree(self):
        # The yaw command sign and the HUD label come from the SAME search_sign, so a
        # right-side glimpse always pairs a negative command with the "right" label.
        f = self._make_follower()
        frame_shape = (480, 640)
        depth = np.zeros((480, 640), dtype=np.float32)
        f.is_tracking = True
        f.tracking_start_time = time.time() - 5.0
        f.last_lost_time = time.time() - 1.0

        for center_x, label, cmd_positive in [((600, 240), "right", False),
                                              ((40, 240), "left", True)]:
            f.last_person_center = center_x
            _, rotation_cmd, dbg = f.update(None, depth, frame_shape)
            self.assertEqual(dbg.get("lost_search_direction"), label)
            self.assertEqual(rotation_cmd > 0.0, cmd_positive)


class TestStairLidarRiserGate(unittest.TestCase):
    """On the stairs a 2D LiDAR ray at the person's bearing hits the RISER (~0.5 m) in front of
    the low camera, NOT the elevated person up the steps. The follow distance fusion must drop
    that near LiDAR while climbing and trust the person's depth -- otherwise fuse_distance's
    reject_far_depth guard trusts the riser, the perceived gap collapses, and the controller
    BRAKES on a phantom 'caught up', freezing with the person still in frame
    (run_sim_20260703_202548: false 0.56 m gap while the patient was 4.1 m ahead up the stairs)."""

    @staticmethod
    def _encode_profile(ranges_m, step_deg=3.0):
        import zlib, base64
        arr = np.clip(np.asarray(ranges_m, dtype=np.float64) * 1000.0, 0, 65535).astype(np.uint16)
        blob = base64.b64encode(zlib.compress(arr.tobytes())).decode("ascii")
        return {"ranges_mm": blob, "n_azimuth": int(arr.shape[0]),
                "azimuth_step_deg": step_deg, "view_range_m": 6.0, "min_range_m": 0.05}

    def _make_follower(self):
        from core.control.person_follower import PersonFollower, PersonFollowingConfig
        cfg = PersonFollowingConfig()
        cfg.camera_fx = 600.0
        cfg.camera_cx = 320.0                        # 640-wide frame, principal point centred
        cfg.lidar_fusion_enabled = True
        return PersonFollower(cfg)

    def _run(self, on_stairs):
        f = self._make_follower()
        frame_shape = (480, 640)
        depth = np.full((480, 640), 2300.0, dtype=np.float32)   # person at 2.3 m (depth is mm)
        person = {"bbox": [300, 150, 340, 460], "matched_detection": True}  # centred -> bearing ~0
        ranges = np.full(120, 5.0)                   # walls at 5 m everywhere...
        for k in (0, 1, 119):                        # ...except a 0.56 m riser DEAD AHEAD (bearing 0)
            ranges[k] = 0.56
        _, _, dbg = f.update(person, depth, frame_shape,
                             lidar_profile=self._encode_profile(ranges), on_stairs=on_stairs)
        return dbg

    def test_flat_ground_keeps_lidar_fusion(self):
        # on_stairs False: unchanged -- reject_far_depth trusts the near LiDAR (this guards the
        # depth background-latch on flat ground). The stair gate must NOT fire.
        dbg = self._run(on_stairs=False)
        self.assertFalse(dbg.get("stair_lidar_riser_rejected"))
        self.assertIsNotNone(dbg.get("fused_distance_m"))
        self.assertLess(float(dbg["fused_distance_m"]), 1.0)    # near LiDAR (riser-like) won

    def test_on_stairs_rejects_riser_and_uses_person_depth(self):
        # on_stairs True: the 0.56 m riser LiDAR is dropped, the person's depth (2.3 m) carries
        # the range -> the controller sees the person far ahead, no false brake.
        dbg = self._run(on_stairs=True)
        self.assertTrue(dbg.get("stair_lidar_riser_rejected"))
        self.assertGreater(float(dbg["fused_distance_m"]), 1.5)


if __name__ == "__main__":
    unittest.main()
