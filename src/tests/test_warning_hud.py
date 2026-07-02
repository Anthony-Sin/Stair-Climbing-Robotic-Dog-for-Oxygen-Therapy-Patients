"""Headless smoke + behaviour tests for the WARNING target-acquisition HUD
(``core/hud/warning_kit.py`` + ``examples/warning_hud.py``).

Pure host render (numpy + cv2 + PIL) — no GL, no camera, no audio device.  Covers:
the HUD renders for every tracking state; info cards actually spawn/collapse with
the scene (DEPTH while tracking, ALERT when lost, CONTACT-02 with 2+ targets); the
range estimate is monotonic; and the presence-driven state machine walks
BOOT→LOCK→LOST→REACQUIRE→LOCK.
"""
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "examples"))

from core.hud.warning_kit import Target, Telemetry, WarningHud, CardStack, CardSpec  # noqa: E402
from warning_hud import estimate_range, StateMachine, SimSource, Demo  # noqa: E402

W, H = 1280, 720


def _frame():
    return np.full((H, W, 3), 90, dtype=np.uint8)


def _prime(hud, tel, t=4.0, ticks=30):
    out = None
    for i in range(ticks):
        out = hud.render(_frame(), tel, t + i * 0.033, 0.033)
    return out


class RenderStates(unittest.TestCase):
    def test_every_state_renders(self):
        p = Target(0.52, 0.5, 0.16, 0.52, 0.9, 1.4, 1, True)
        states = {
            "BOOTING": Telemetry("BOOTING", [], boot=0.3),
            "ACQUIRING": Telemetry("ACQUIRING", [p], boot=1.0, signal=0.5),
            "LOCKED": Telemetry("LOCKED", [p], boot=1.0, signal=0.9),
            "TARGET LOST": Telemetry("TARGET LOST", [], boot=1.0, signal=0.1, lost_for=1.5),
            "REACQUIRE": Telemetry("REACQUIRE", [], boot=1.0, signal=0.4),
        }
        for name, tel in states.items():
            hud = WarningHud((W, H))
            out = _prime(hud, tel)
            self.assertEqual(out.shape, (H, W, 3), name)
            self.assertEqual(out.dtype, np.uint8, name)
            self.assertGreater(int(out.max()), 40, name)  # something drew

    def test_resizes_mismatched_frame(self):
        hud = WarningHud((W, H))
        small = np.full((360, 640, 3), 100, np.uint8)
        out = hud.render(small, Telemetry("LOCKED", [Target(0.5, 0.5, 0.1, 0.3)], boot=1.0), 3.0, 0.03)
        self.assertEqual(out.shape, (H, W, 3))


class DynamicCards(unittest.TestCase):
    def _active_keys(self, hud, tel):
        left, right = hud._card_specs(tel)
        return {c.key for c in left + right}

    def test_detail_card_only_when_surfaced(self):
        hud = WarningHud((W, H))
        p = Target(0.5, 0.5, 0.16, 0.5, 0.9, 1.4, 1, True)
        self.assertIn("detail", self._active_keys(hud, Telemetry("LOCKED", [p], boot=1.0, show_detail=True)))
        # once the "at a glance" hold expires, the detail card is gone
        self.assertNotIn("detail", self._active_keys(hud, Telemetry("LOCKED", [p], boot=1.0, show_detail=False)))

    def test_alert_only_when_lost(self):
        hud = WarningHud((W, H))
        p = Target(0.5, 0.5, 0.16, 0.5, 0.9, 1.4, 1, True)
        self.assertNotIn("alert", self._active_keys(hud, Telemetry("LOCKED", [p], boot=1.0)))
        self.assertIn("alert", self._active_keys(hud, Telemetry("TARGET LOST", [], boot=1.0, lost_for=1.0)))

    def test_contact2_only_with_two_targets(self):
        hud = WarningHud((W, H))
        p = Target(0.5, 0.5, 0.16, 0.5, 0.9, 1.4, 1, True)
        s = Target(0.3, 0.45, 0.1, 0.34, 0.6, 2.4, 2)
        self.assertNotIn("contact2", self._active_keys(hud, Telemetry("LOCKED", [p], boot=1.0)))
        self.assertIn("contact2", self._active_keys(hud, Telemetry("LOCKED", [p, s], boot=1.0)))

    def test_stack_spawns_then_collapses(self):
        st = CardStack("R")
        spec = CardSpec("detail", "TARGET", rows=[("A", "B", None)])
        for _ in range(20):
            st.update([spec], 0.05)
        self.assertIn("detail", st._cards)
        self.assertGreaterEqual(st._cards["detail"].anim, 0.99)
        for _ in range(20):                       # remove -> must fully collapse & drop
            st.update([], 0.05)
        self.assertNotIn("detail", st._cards)


class InstrumentsAndToasts(unittest.TestCase):
    def test_depth_and_radar_render_when_sensors_present(self):
        hud = WarningHud((W, H))
        depth = np.tile(np.linspace(400, 5000, 96).astype(np.uint16)[:, None], (1, 128))
        lidar = {"ranges_m": [3.0] * 121, "view_range_m": 6.0}
        tel = Telemetry("LOCKED", [Target(0.5, 0.5, 0.16, 0.5, 0.9, 1.4, 1, True)],
                        boot=1.0, depth=depth, stairs=(True, 0.8, (0.3, 0.45, 0.74, 0.95)),
                        lidar=lidar)
        out = _prime(hud, tel)
        self.assertEqual(out.shape, (H, W, 3))

    def test_depth_color_maps_near_warm_far_cool(self):
        near = WarningHud._depth_color(np.full((4, 4), 400, np.uint16))   # BGR
        far = WarningHud._depth_color(np.full((4, 4), 5000, np.uint16))
        self.assertGreater(int(near[..., 2].mean()), int(far[..., 2].mean()))   # near redder
        self.assertGreater(int(far[..., 0].mean()), int(near[..., 0].mean()))   # far bluer

    def test_toasts_spawn_and_expire(self):
        hud = WarningHud((W, H))
        frame = _frame()
        base = dict(boot=1.0)
        # spawn a toast, tick it up, then drop it and let it collapse
        for _ in range(10):
            hud._draw_toasts(frame, Telemetry("LOCKED", **base, toasts=[("HELLO", "info")]), 1.0, 0.05)
        self.assertIn("HELLO", hud._toasts)
        for _ in range(12):
            hud._draw_toasts(frame, Telemetry("LOCKED", **base, toasts=[]), 1.0, 0.05)
        self.assertNotIn("HELLO", hud._toasts)


class Telemetry_Honesty(unittest.TestCase):
    def test_range_monotonic(self):
        near = estimate_range(0.60)   # big bbox -> close
        far = estimate_range(0.20)    # small bbox -> far
        self.assertIsNotNone(near)
        self.assertLess(near, far)
        self.assertIsNone(estimate_range(0.0))

    def test_sim_confidence_varies_and_bounded(self):
        src = SimSource()
        scores = []
        for t in (2.5, 3.5, 4.6, 5.5, 6.5):       # inside the "present" window
            _, tgs, _ = src.read(t)
            scores += [tg.score for tg in tgs]
        self.assertTrue(scores)
        self.assertTrue(all(0.3 <= s <= 1.0 for s in scores))
        self.assertGreater(max(scores) - min(scores), 0.02)   # not a flat fake value


class StateWalk(unittest.TestCase):
    def test_presence_drives_full_sequence(self):
        sm = StateMachine(lock_frames=3, lost_frames=3)
        seq = []
        cues = []

        def present_at(t):                        # gone in a middle window
            return not (5.0 <= t < 6.0) and t >= 2.2

        t = 0.0
        while t < 8.0:
            sm.update(present_at(t), t, cues)
            seq.append(sm.state)
            t += 0.05
        self.assertEqual(seq[0], "BOOTING")
        self.assertIn("LOCKED", seq)
        self.assertIn("TARGET LOST", seq)
        self.assertIn("REACQUIRE", seq)
        self.assertEqual(sm.state, "LOCKED")       # relocked by the end
        names = [c[1] for c in cues]
        self.assertIn("lock", names)
        self.assertIn("alert", names)

    def test_demo_step_runs_without_audio(self):
        demo = Demo(SimSource(), WarningHud((W, H)), with_audio=False)
        out = None
        for i in range(60):
            out = demo.step(i / 30.0, 1 / 30.0)
        self.assertEqual(out.shape, (H, W, 3))
        self.assertTrue(demo.cues)                 # emitted at least boot cues


if __name__ == "__main__":
    unittest.main()
