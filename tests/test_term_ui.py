"""Host tests for the btop-style terminal UI toolkit (core.telemetry.term_ui).

Pure-Python, no GPU/Isaac. Verifies the two properties that keep the redesign
from breaking anything: (1) when color is disabled the output is plain ASCII
with zero escape codes, and (2) every component lines up to the requested
visible width regardless of ANSI styling.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.telemetry import term_ui as tu  # noqa: E402


PLAIN = tu.Theme(level=tu.NONE, unicode=False)
PLAIN_UNI = tu.Theme(level=tu.NONE, unicode=True)
COLOR = tu.Theme(level=tu.TRUECOLOR, unicode=True)


class TestColorDisabled(unittest.TestCase):
    def test_paint_is_noop_without_color(self):
        self.assertEqual(PLAIN.paint("hi", fg="green", bold=True), "hi")

    def test_no_escape_codes_anywhere_when_disabled(self):
        lines = tu.panel("config", [tu.kv("mode", "sim", PLAIN), tu.meter(0.6, 10, PLAIN)],
                         width=40, theme=PLAIN)
        blob = "\n".join(lines)
        blob += tu.sparkline([1, 2, 3, 4], PLAIN)
        blob += tu.status_line("12:00:00", "isaac", "ready", "world boot", PLAIN)
        self.assertNotIn("\x1b", blob)

    def test_ascii_box_fallback(self):
        lines = tu.panel("x", ["body"], width=20, theme=PLAIN)
        self.assertTrue(lines[0].startswith("+"))
        self.assertNotIn("╭", "\n".join(lines))

    def test_color_enabled_emits_escape(self):
        self.assertIn("\x1b[", COLOR.paint("hi", fg="green"))


class TestWidthAlignment(unittest.TestCase):
    def test_visible_len_ignores_ansi(self):
        self.assertEqual(tu.visible_len(COLOR.paint("hello", fg="red")), 5)

    def test_panel_lines_exact_width_plain(self):
        for w in (20, 31, 48, 60):
            lines = tu.panel("title", ["a", tu.kv("k", "v", PLAIN), ""], width=w, theme=PLAIN)
            for ln in lines:
                self.assertEqual(tu.visible_len(ln), w, f"width {w}: {ln!r}")

    def test_panel_lines_exact_width_colored(self):
        for w in (24, 40, 55):
            rows = [tu.kv("network", "127.0.0.1:52002", COLOR),
                    tu.meter(0.42, 18, COLOR)]
            lines = tu.panel("config", rows, width=w, accent="net", theme=COLOR)
            for ln in lines:
                self.assertEqual(tu.visible_len(ln), w, f"width {w}: {ln!r}")

    def test_panel_grows_to_fit_long_title(self):
        lines = tu.panel("a-very-long-title-here", ["x"], width=10, theme=PLAIN)
        self.assertGreaterEqual(tu.visible_len(lines[0]), len("a-very-long-title-here") + 8)

    def test_footer_line_width(self):
        lines = tu.panel("t", ["body"], width=30, theme=COLOR, footer="press q to quit")
        self.assertEqual(tu.visible_len(lines[-1]), 30)

    def test_pad_alignments(self):
        self.assertEqual(tu.pad("ab", 5), "ab   ")
        self.assertEqual(tu.pad("ab", 5, "right"), "   ab")
        self.assertEqual(tu.pad("ab", 5, "center"), " ab  ")

    def test_truncate_marks_overflow(self):
        self.assertEqual(tu.visible_len(tu.truncate("abcdefgh", 4)), 4)
        self.assertTrue(tu.truncate("abcdefgh", 4).endswith("…"))


class TestMeterAndSparkline(unittest.TestCase):
    def test_meter_fill_count(self):
        # plain unicode theme: 60% of 10 cells -> 6 filled '■'
        m = tu.meter(0.6, 10, PLAIN_UNI)
        self.assertEqual(m.count("■"), 6)
        self.assertEqual(m.count("░"), 4)

    def test_meter_clamps(self):
        self.assertEqual(tu.visible_len(tu.meter(5.0, 8, PLAIN_UNI)), 8)
        self.assertEqual(tu.visible_len(tu.meter(-1.0, 8, PLAIN_UNI)), 8)

    def test_meter_ascii_mode(self):
        m = tu.meter(0.5, 8, PLAIN)  # non-unicode -> '#'/'.'
        self.assertEqual(m.count("#"), 4)
        self.assertEqual(m.count("."), 4)

    def test_sparkline_length_matches_samples(self):
        s = tu.sparkline([0, 1, 2, 3, 4], PLAIN_UNI)
        self.assertEqual(tu.visible_len(s), 5)

    def test_sparkline_windowing(self):
        s = tu.sparkline(list(range(100)), PLAIN_UNI, width=10)
        self.assertEqual(tu.visible_len(s), 10)

    def test_sparkline_empty(self):
        self.assertEqual(tu.sparkline([], COLOR), "")


class TestLayout(unittest.TestCase):
    def test_hjoin_rows_equal_width(self):
        a = tu.panel("a", ["x", "yy"], 20, theme=COLOR)
        b = tu.panel("b", ["one", "two", "three"], 30, theme=COLOR)
        joined = tu.hjoin([a, b], gap=2)
        widths = {tu.visible_len(ln) for ln in joined}
        self.assertEqual(len(widths), 1, f"ragged widths: {widths}")
        self.assertEqual(joined and tu.visible_len(joined[0]), 20 + 2 + 30)

    def test_hjoin_pads_shorter_block(self):
        a = tu.panel("a", ["x"], 10, theme=PLAIN)          # 3 lines
        b = tu.panel("b", ["1", "2", "3", "4"], 10, theme=PLAIN)  # 6 lines
        joined = tu.hjoin([a, b], gap=1)
        self.assertEqual(len(joined), 6)

    def test_hjoin_empty(self):
        self.assertEqual(tu.hjoin([]), [])

    def test_fit_height_pads_and_truncates(self):
        self.assertEqual(len(tu.fit_height(["a", "b"], 5)), 5)
        self.assertEqual(len(tu.fit_height(["a", "b", "c", "d"], 2)), 2)

    def test_bar_exact_width(self):
        for w in (20, 40, 80):
            self.assertEqual(tu.visible_len(tu.bar("left", "right", w, COLOR)), w)
            self.assertEqual(tu.visible_len(tu.bar("left", "right", w, PLAIN)), w)

    def test_bar_truncates_long_left(self):
        b = tu.bar("x" * 100, "end", 20, PLAIN)
        self.assertEqual(tu.visible_len(b), 20)
        self.assertTrue(b.rstrip().endswith("end"))


class TestGradient(unittest.TestCase):
    def test_gradient_endpoints(self):
        self.assertEqual(tu.gradient_rgb(0.0), (0x77, 0xCA, 0x9B))
        self.assertEqual(tu.gradient_rgb(1.0), (0xDC, 0x4C, 0x4C))
        self.assertEqual(tu.gradient_rgb(0.5), (0xCB, 0xC0, 0x6C))

    def test_gradient_monotone_red_rises(self):
        self.assertLess(tu.gradient_rgb(0.1)[0], tu.gradient_rgb(0.9)[0])


class TestColorDetection(unittest.TestCase):
    def test_no_color_env_disables(self):
        old = os.environ.get("NO_COLOR")
        os.environ["NO_COLOR"] = "1"
        try:
            self.assertEqual(tu.detect_color_level(force=None), tu.NONE)
        finally:
            if old is None:
                del os.environ["NO_COLOR"]
            else:
                os.environ["NO_COLOR"] = old

    def test_force_flags(self):
        self.assertEqual(tu.detect_color_level(force=True), tu.TRUECOLOR)
        self.assertEqual(tu.detect_color_level(force=False), tu.NONE)

    def test_downconvert_256_and_16(self):
        # red maps to a high-red 256 index and ansi16 red(9)
        self.assertEqual(tu._rgb_to_16(0xFF, 0, 0), 9)
        idx = tu._rgb_to_256(0xFF, 0, 0)
        self.assertTrue(16 <= idx <= 231)


class TestStatusGlyphs(unittest.TestCase):
    def test_known_states_have_glyphs(self):
        for state in ("start", "ready", "failed", "warning", "skipped"):
            self.assertTrue(tu.state_glyph(state, COLOR))

    def test_ascii_glyph_fallback_no_unicode(self):
        g = tu.state_glyph("ready", PLAIN)
        self.assertNotIn("✔", g)


if __name__ == "__main__":
    unittest.main()
