"""ClimbFSM flat-loss-glide turn-guard.

The straight glide (FLAT_LOSS_GLIDE) is only correct when the patient was lost heading
roughly dead-ahead. When they turned off-axis (a zigzag apex) gliding straight drives away
from them AND -- because the dispatch hard-zeroes yaw during the glide -- it suppresses the
recovery turn. The guard makes the glide YIELD to the recovery yaw in those cases.

Pure-Python: ClimbFSM has no Isaac/torch deps. Run via pytest or directly.
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.control.climb_fsm import ClimbFSM, GLIDE_MAX_BEARING_DEG


class _Args:
    """Minimal args surface ClimbFSM.update() reads (values that keep all stair latches off)."""
    stair_seen_persist_sec = 1.0
    stair_hold_suppress_sec = 1.0
    obstacle_slow_distance = 0.5
    stair_target_distance = 0.6
    stair_depth_engage_distance = 0.45
    stair_climb_max_sec = 20.0
    stair_near_distance = 0.45
    stair_forward_floor = 0.16
    stair_loss_forward_floor = 0.35
    trans_x_max = 0.85
    stair_speed_scale = 0.45
    stair_climb_commit_distance = 0.4
    follow_loss_glide_sec = 4.0


def _glide_eligible(last_seen_bearing_deg, recovery_yaw_active):
    fsm = ClimbFSM(_Args())
    out = fsm.update(
        time.perf_counter(),
        stairs_detected=False,
        stairs_action_active=False,
        stairs_depth_m=None,
        last_stairs_depth_m=None,
        stairs_depth_ever_confirmed=False,
        person_detected=False,          # patient lost
        depth_distance_m=None,
        front_near_m=2.0,               # clear path ahead (> 0.9)
        standoff_gap_ctrl_m=None,
        lost_age_sec=1.0,               # within follow_loss_glide_sec
        motion_allowed=False,
        last_seen_bearing_deg=last_seen_bearing_deg,
        recovery_yaw_active=recovery_yaw_active,
    )
    return bool(out["flat_loss_glide_eligible"]), out["fsm_state"]


class TestFlatLossGlideGuard(unittest.TestCase):
    def test_glides_when_lost_straight_ahead(self):
        eligible, state = _glide_eligible(last_seen_bearing_deg=2.0, recovery_yaw_active=False)
        self.assertTrue(eligible)
        self.assertEqual(state, "FLAT_LOSS_GLIDE")

    def test_no_glide_when_lost_off_axis(self):
        # Patient clearly turned (30 deg >> 8 deg) -> do NOT glide straight; fall through to STOP
        # so the recovery yaw owns the frame.
        eligible, state = _glide_eligible(last_seen_bearing_deg=30.0, recovery_yaw_active=False)
        self.assertFalse(eligible)
        self.assertEqual(state, "STOP")

    def test_no_glide_when_recovery_yaw_active(self):
        # Even near-centre, if the follower is already turning to re-acquire, the glide yields.
        eligible, _ = _glide_eligible(last_seen_bearing_deg=2.0, recovery_yaw_active=True)
        self.assertFalse(eligible)

    def test_glides_when_bearing_unknown(self):
        # No bearing info (None) -> preserve the original straight-loss behaviour.
        eligible, _ = _glide_eligible(last_seen_bearing_deg=None, recovery_yaw_active=False)
        self.assertTrue(eligible)

    def test_threshold_value(self):
        self.assertEqual(GLIDE_MAX_BEARING_DEG, 8.0)


if __name__ == "__main__":
    unittest.main()
