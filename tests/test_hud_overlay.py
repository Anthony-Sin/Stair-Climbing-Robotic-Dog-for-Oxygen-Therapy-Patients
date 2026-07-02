"""Smoke tests for the Tac-Amber on-frame HUD (core/hud/).

Pure host render (numpy + cv2 + PIL) — no Isaac/GPU. Verifies the overlay draws
without error for every tracking state, that the group backings actually tint
the frame (legibility fix), that the status pill grows with its label, and that
the re-added depth view tolerates a missing/odd depth image.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hud.visualization import draw_frame_overlays          # noqa: E402
from core.hud import hud_primitives as hp                       # noqa: E402
from core.hud.hud_panels import draw_depth_view                 # noqa: E402


def _frame():
    return np.full((720, 1280, 3), 120, dtype=np.uint8)


def _debug(lock, fell=False, depth=True):
    dep = None
    if depth:
        dep = np.tile(np.linspace(2500, 600, 480).astype(np.uint16)[:, None], (1, 640))
    return {
        "matched_visual_lock": lock,
        "center_x": 640, "depth_distance_m": 0.52, "rotation_error_deg": -6.4,
        "distance_confidence": 0.93, "distance_source": "lidar_depth_fused",
        "stairs_detected": True, "stairs_conf": 0.88, "stairs_bbox": [520, 300, 760, 560],
        "depth_img": dep,
        "stair_demo": {
            "robot": {"fell": fell, "fall_type": "collapsed_low" if fell else "none",
                      "roll_deg": 2.1, "pitch_deg": 8.4, "height_m": 0.31},
            "locomotion": {"policy": "pgtt_level17", "mode": "CLIMB",
                           "gait_pattern": "diagonal_trot", "foot_clearance_m": 0.06,
                           "commanded_speed_mps": 0.30, "body_height_target_m": 0.30,
                           "leg_commands": {}},
        },
    }


class TestOverlayStates(unittest.TestCase):
    def test_all_states_render_and_modify_frame(self):
        for lock, reac, fell in [(True, False, False), (False, True, False),
                                 (False, False, False), (False, False, True)]:
            f = _frame()
            before = f.copy()
            draw_frame_overlays(f, _debug(lock, fell), False, reac, "single",
                                frame_meta={"success": True, "swing_legs": ["FL", "RR"]},
                                trans_x_cmd=0.30, rotation_cmd=-0.12, proc_fps=6.7, view_fps=5.9)
            self.assertFalse(np.array_equal(f, before), "overlay drew nothing")

    def test_runs_without_depth_image(self):
        f = _frame()
        draw_frame_overlays(f, _debug(True, depth=False), False, False, "single",
                            frame_meta={"success": True})  # must not raise


class TestCentredTags(unittest.TestCase):
    def test_stairs_only_no_person_renders(self):
        # center_x None → no PERSON tag, but the centred STAIRS tag must still draw
        f = _frame()
        d = _debug(False)
        d["center_x"] = None
        d["depth_distance_m"] = None
        before = f.copy()
        draw_frame_overlays(f, d, False, True, "single", frame_meta={"success": True})
        self.assertFalse(np.array_equal(f, before), "stairs-only overlay drew nothing")


class TestFocalTag(unittest.TestCase):
    def test_focal_tag_draws_and_returns_bottom(self):
        f = _frame()
        layer = hp.TextLayer()
        before = f.copy()
        bottom = hp.focal_tag(f, layer, 640, 300,
                              [("TARGET", hp.AMBER), ("0.62m", hp.BRIGHT_SLATE)], size=15)
        layer.flush(f)
        self.assertGreater(bottom, 300, "tag bottom must be below its top")
        self.assertFalse(np.array_equal(f, before), "focal tag drew nothing")


class TestDisplayKit(unittest.TestCase):
    def test_decode_text_mid_and_full(self):
        f = _frame()
        layer = hp.TextLayer()
        hp.decode_text(layer, 100, 100, "PERSON #01", hp.ACCENT, progress=0.4, size=16, seed=7)
        hp.decode_text(layer, 100, 140, "PERSON #01", hp.ACCENT, progress=1.0, size=16)
        before = f.copy()
        layer.flush(f)
        self.assertFalse(np.array_equal(f, before), "decode_text drew nothing")

    def test_label_stack_and_status_bar_return_y(self):
        f = _frame()
        layer = hp.TextLayer()
        y1 = hp.label_stack(layer, 20, 40, "INPUT DATA", "PERSON #01", progress=0.6, seed=3)
        self.assertGreater(y1, 40)
        yb = hp.status_bar(f, layer, 20, y1, 200, "LOCKED", "locked", progress=0.5)
        self.assertGreater(yb, y1)
        layer.flush(f)

    def test_scan_frame_animates(self):
        f = _frame()
        a = f.copy()
        hp.scan_frame(f, 640, 360, 280, 280, progress=0.3, state="scanning")
        self.assertFalse(np.array_equal(f, a), "scan_frame drew nothing")


class TestGroupBacking(unittest.TestCase):
    def test_backing_darkens_region(self):
        f = _frame()
        region = f[40:120, 10:200].copy()
        hp.group_backing(f, 10, 40, 200, 120, alpha=0.5)
        # a dark scrim must reduce mean brightness of the covered region
        self.assertLess(f[40:120, 10:200].mean(), region.mean())


class TestStatusPill(unittest.TestCase):
    def test_pill_right_edge_grows_with_text(self):
        f = _frame()
        layer = hp.TextLayer()
        short = hp.status_pill(f, layer, 10, 10, "LOST", "lost", size=12)
        wide = hp.status_pill(f, layer, 10, 60, "SEARCHING", "scanning", size=12)
        self.assertGreater(wide, short)

    def test_pill_states_do_not_crash(self):
        f = _frame()
        layer = hp.TextLayer()
        for word, state in [("LOCKED", "locked"), ("SEARCHING", "scanning"), ("LOST", "lost")]:
            x1 = hp.status_pill(f, layer, 10, 10, word, state, size=12)
            self.assertGreater(x1, 10)
        layer.flush(f)


class TestDepthView(unittest.TestCase):
    def test_depth_view_handles_none(self):
        f = _frame()
        layer = hp.TextLayer()
        draw_depth_view(f, layer, 10, 200, 176, 120, None, None, False, 0.0)
        layer.flush(f)  # must not raise

    def test_depth_view_blits_image(self):
        f = _frame()
        layer = hp.TextLayer()
        dep = np.tile(np.linspace(2500, 600, 480).astype(np.uint16)[:, None], (1, 640))
        before = f[201:319, 11:185].copy()
        draw_depth_view(f, layer, 10, 200, 176, 120, dep, [520, 300, 760, 560], True, 0.88)
        self.assertFalse(np.array_equal(f[201:319, 11:185], before), "depth tile not blitted")


if __name__ == "__main__":
    unittest.main()
