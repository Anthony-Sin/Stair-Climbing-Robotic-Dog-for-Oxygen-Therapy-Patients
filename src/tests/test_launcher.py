"""Host tests for the btop launcher's command-building (launcher.py).

Pure-Python: verifies that interactive flag choices map to the exact normal
commands (sim\\run_sim.bat ... / ./real/run_real.sh ...), that defaults are
omitted, and that dependent options hide correctly. No GPU/Isaac/TTY.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import launcher as L  # noqa: E402
from core.telemetry import term_ui as tu  # noqa: E402
from launcher_lib import app as A  # noqa: E402

PLAIN = tu.Theme(level=tu.NONE, unicode=True)


def _sim_parse(tokens):
    """Parse console tokens onto a fresh sim config; return (display, errors)."""
    cfg = L._sim_config()
    errs = L.parse_tokens(cfg, tokens)
    display, *_ = L.build_command(cfg)
    return display, errs


class _FakeScreen:
    """Stand-in for tu.Screen: captures rendered frames, no real terminal."""
    def __init__(self, *a, **k):
        self.frames = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def render(self, lines):
        self.frames.append(list(lines))


class _FakeKeys:
    """Feeds a scripted key sequence, then 'quit' so a loop can never hang."""
    def __init__(self, seq):
        self.seq = list(seq)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.seq.pop(0) if self.seq else "quit"


class TestSimCommand(unittest.TestCase):
    def test_default_is_bare(self):
        d, _argv, _cwd, _env, _sup = L.build_command(L._sim_config())
        self.assertEqual(d, "sim\\run_sim.bat")

    def test_tuned_flags_in_order(self):
        c = L._sim_config()
        c.get("policy").value = "parkour"
        c.get("headless").value = True
        c.get("max_run").value = 120
        c.get("o2").value = True
        d, _a, _c, _e, _s = L.build_command(c)
        self.assertIn("--locomotion-policy parkour", d)
        self.assertIn("--headless", d)
        self.assertIn("--with-o2-payload", d)
        self.assertIn("--max-run-time-sec 120", d)

    def test_default_numeric_is_omitted(self):
        c = L._sim_config()
        # max_run stays at its 900 default -> no flag
        d, *_ = L.build_command(c)
        self.assertNotIn("--max-run-time-sec", d)

    def test_pgtt_level_hidden_for_parkour(self):
        c = L._sim_config()
        self.assertTrue(any(o.key == "pgtt_level" for o in c.visible()))
        c.get("policy").value = "parkour"
        self.assertFalse(any(o.key == "pgtt_level" for o in c.visible()))

    def test_vision_preview_sets_show_recordings(self):
        c = L._sim_config()
        c.get("vision_preview").value = True
        _d, _a, _c, env, _s = L.build_command(c)
        self.assertEqual(env.get("SHOW_RECORDINGS"), "1")

    def test_no_pause_env_set(self):
        _d, _a, _c, env, _s = L.build_command(L._sim_config())
        self.assertEqual(env.get("NO_PAUSE"), "1")


class TestRealCommand(unittest.TestCase):
    def test_default_is_bare(self):
        d, argv, _c, _e, _s = L.build_command(L._real_config())
        self.assertEqual(d, "./real/run_real.sh")
        self.assertEqual(argv[0], "bash")

    def test_lidar_and_record(self):
        r = L._real_config()
        r.get("heightscan").value = "lidar"
        r.get("record").value = True
        d, _a, _c, _e, _s = L.build_command(r)
        self.assertIn("--lidar", d)
        self.assertIn("--record", d)


class TestOptionCycling(unittest.TestCase):
    def test_bool_toggle(self):
        o = L.Option("h", "headless", "bool", False, flag="--headless")
        o.cycle(1)
        self.assertTrue(o.value)
        self.assertEqual(o.to_flags(), ["--headless"])
        o.cycle(-1)
        self.assertEqual(o.to_flags(), [])

    def test_choice_wraps(self):
        o = L.Option("p", "policy", "choice", "pgtt", flag="--p", choices=["pgtt", "parkour"])
        o.cycle(1)
        self.assertEqual(o.value, "parkour")
        o.cycle(1)
        self.assertEqual(o.value, "pgtt")  # wrap

    def test_numeric_clamps(self):
        o = L.Option("n", "n", "int", 0, flag="--n", step=1, minimum=0, maximum=3)
        o.cycle(-1)
        self.assertEqual(o.value, 0)  # clamp at min
        for _ in range(10):
            o.cycle(1)
        self.assertEqual(o.value, 3)  # clamp at max

    def test_no_x_bool_emits_when_false(self):
        o = L.Option("s", "stand", "bool", True, flag="--no-stand-up", bool_true_flag=False)
        self.assertEqual(o.to_flags(), [])   # value True -> default behavior, no flag
        o.cycle(1)
        self.assertEqual(o.to_flags(), ["--no-stand-up"])  # value False -> emit --no-x


class TestRenderSmoke(unittest.TestCase):
    def test_render_start_no_crash_plain(self):
        lines = L.render_start(L._sim_config(), 0, PLAIN)
        self.assertTrue(any("run_sim.bat" in ln for ln in lines))

    def test_render_dashboard_no_crash(self):
        stages = {"setup": {"ts": "12:00:00", "state": "ready", "msg": "ok"}}
        lines = L.render_dashboard(L._sim_config(), stages, ["log line"], 0.0, PLAIN, None, [0, 1, 2])
        self.assertTrue(lines)

    def test_stage_regex_parses_launcher_line(self):
        m = L._STAGE_RE.match("[12:34:56] isaac        start     world boot")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(2), "isaac")
        self.assertEqual(m.group(3), "start")

    def test_telemetry_rows_handle_missing(self):
        self.assertTrue(L._telemetry_rows(None, None, PLAIN, 40))  # no data -> placeholder
        sim = {"x": 2.3, "h": 0.31, "pitch": 8.2, "roll": -3.1, "policy_cmd": [0.4, 0.0, 0.05]}
        rows = L._telemetry_rows(sim, [1, 2, 3], PLAIN, 40)
        blob = "\n".join(rows)
        self.assertIn("2.3", blob)
        self.assertIn("0.31", blob)

    def test_dashboard_with_telemetry_renders(self):
        stages = {"docker": {"ts": "12:01:05", "state": "running", "msg": "up"}}
        sim = {"x": 1.0, "h": 0.3, "pitch": 5.0, "roll": 1.0, "policy_cmd": [0.3, 0, 0]}
        lines = L.render_dashboard(L._sim_config(), stages, ["log"], 0.0, PLAIN, None,
                                   [0, 1], telemetry=sim, x_hist=[0, 1], run_dir="log/run_x")
        self.assertTrue(any("robot" in ln for ln in lines))

    def test_read_fall_diag_missing_dir(self):
        self.assertIsNone(L._read_fall_diag(None))
        self.assertIsNone(L._read_fall_diag("/no/such/dir/xyz"))

    def test_mission_view_shows_climb_and_gap(self):
        sim = {"gap_m": 0.52, "person_detected": True, "stairs_detected": True,
               "stairs_action_active": True, "body_vx": 0.38, "tilt_deg": 8.6,
               "handoff": {"handoff_state": "climb", "stair_count": 3, "handoff_climbs_done": 2}}
        blob = "\n".join(L._telemetry_rows(sim, None, PLAIN, 40, view="mission"))
        self.assertIn("CLIMB", blob)
        self.assertIn("step 3", blob)
        self.assertIn("0.52", blob)          # person gap
        self.assertIn("climb-gate", blob)

    def test_tele_views_constant(self):
        self.assertIn("robot", L.TELE_VIEWS)
        self.assertIn("mission", L.TELE_VIEWS)

    def test_console_scroll_indicator(self):
        cfg = L._sim_config()
        tail = [f"line {i}" for i in range(60)]
        live = "\n".join(L.render_dashboard(cfg, {}, tail, 0.0, PLAIN, None, [0], scroll=0))
        scrolled = "\n".join(L.render_dashboard(cfg, {}, tail, 0.0, PLAIN, None, [0], scroll=8))
        self.assertIn("live ·", live)
        self.assertIn("End=live", scrolled)

    def test_dashboard_mission_view_renders(self):
        sim = {"handoff": {"handoff_state": "walk", "stair_count": 0}, "gap_m": 1.0,
               "person_detected": True}
        lines = L.render_dashboard(L._sim_config(), {}, ["x"], 0.0, PLAIN, None, [0],
                                   telemetry=sim, tele_view="mission")
        self.assertTrue(any("mission" in ln for ln in lines))


class TestParseLine(unittest.TestCase):
    def test_bare_word_bool(self):
        d, errs = _sim_parse(["headless"])
        self.assertIn("--headless", d)
        self.assertEqual(errs, [])

    def test_dashed_form(self):
        d, _ = _sim_parse(["--headless"])
        self.assertIn("--headless", d)

    def test_choice_key_value(self):
        d, errs = _sim_parse(["pgtt-level", "level20"])
        self.assertIn("--pgtt-level level20", d)
        self.assertEqual(errs, [])

    def test_bare_choice_value(self):
        d, _ = _sim_parse(["level20"])          # standalone choice value
        self.assertIn("--pgtt-level level20", d)

    def test_choice_numeric_shorthand(self):
        d, _ = _sim_parse(["level", "20"])       # 20 -> level20
        self.assertIn("--pgtt-level level20", d)

    def test_numeric_value_and_clamp(self):
        d, _ = _sim_parse(["max-run", "300"])
        self.assertIn("--max-run-time-sec 300", d)
        d2, _ = _sim_parse(["max-run", "999999"])   # clamps to maximum 3600
        self.assertIn("--max-run-time-sec 3600", d2)

    def test_multiple_tokens(self):
        d, errs = _sim_parse(["headless", "o2", "pgtt-level", "level20"])
        self.assertIn("--headless", d)
        self.assertIn("--with-o2-payload", d)
        self.assertIn("--pgtt-level level20", d)
        self.assertEqual(errs, [])

    def test_unknown_flag_reports_error(self):
        d, errs = _sim_parse(["frobnicate"])
        self.assertTrue(errs)
        self.assertIn("frobnicate", errs[0])
        self.assertEqual(d, "sim\\run_sim.bat")   # nothing applied

    def test_choice_needs_value(self):
        _d, errs = _sim_parse(["pgtt-level"])
        self.assertTrue(errs)

    def test_real_bare_choice_and_bool(self):
        cfg = L._real_config()
        errs = L.parse_tokens(cfg, ["lidar", "record"])
        d, *_ = L.build_command(cfg)
        self.assertIn("--lidar", d)
        self.assertIn("--record", d)
        self.assertEqual(errs, [])


class TestComplete(unittest.TestCase):
    def test_completes_flag_and_command(self):
        cands = L.complete(L._sim_config(), "he")
        self.assertIn("headless", cands)
        self.assertIn("help", cands)          # command word

    def test_completes_choice_values(self):
        self.assertIn("level20", L.complete(L._sim_config(), "level"))

    def test_target_word(self):
        self.assertIn("sim", L.complete(L._sim_config(), "s"))


class TestPresets(unittest.TestCase):
    def test_both_targets_have_presets(self):
        self.assertTrue(L.presets_for("sim"))
        self.assertTrue(L.presets_for("real"))

    def test_first_sim_preset_is_bare(self):
        cfg = L.preset_config("sim", L.presets_for("sim")[0])
        d, *_ = L.build_command(cfg)
        self.assertEqual(d, "sim\\run_sim.bat")

    def test_headless_preset_applies_flag(self):
        p = next(p for p in L.presets_for("sim") if "headless" in p.tokens)
        d, *_ = L.build_command(L.preset_config("sim", p))
        self.assertIn("--headless", d)

    def test_real_lidar_preset(self):
        p = next(p for p in L.presets_for("real") if "lidar" in p.tokens)
        d, *_ = L.build_command(L.preset_config("real", p))
        self.assertIn("--lidar", d)


class TestMenuLoop(unittest.TestCase):
    """Drive the whole preset menu headlessly with scripted keystrokes."""

    def _run(self, keys, start="sim", dashboard=False):
        fake_keys = _FakeKeys(keys)
        real_bc = L.build_command

        def bc(cfg):                       # force 'supported' so argv reflects flags on any OS
            d, argv, cwd, env, _sup = real_bc(cfg)
            return d, argv, cwd, env, True

        with mock.patch.object(A, "KeyReader", lambda: fake_keys), \
                mock.patch.object(A.tu, "Screen", _FakeScreen), \
                mock.patch.object(A, "build_command", side_effect=bc), \
                mock.patch.object(A, "_await_return", return_value=False), \
                mock.patch.object(A, "run_passthrough", return_value=0) as rp, \
                mock.patch.object(A, "run_with_dashboard", return_value=0) as rd:
            rc = A.interactive(L._sim_config(), L._real_config(), PLAIN, start,
                               demo=False, dashboard=dashboard)
        return rc, rp, rd

    def test_enter_runs_selected_preset(self):
        rc, rp, rd = self._run(["enter"])                 # preset 0 = full demo
        self.assertEqual(rc, 0)
        rp.assert_called_once()
        rd.assert_not_called()
        self.assertIn("run_sim.bat", rp.call_args[0][0][-1])

    def test_navigate_then_run_applies_preset(self):
        _rc, rp, _rd = self._run(["down", "enter"])       # preset 1 = headless
        self.assertIn("--headless", rp.call_args[0][0])

    def test_edit_toggle_then_run(self):
        # e -> flag editor; ↓↓↓ to 'headless render'; → toggles it on; ⏎ runs.
        keys = ["e", "down", "down", "down", "right", "enter"]
        _rc, rp, _rd = self._run(keys)
        self.assertIn("--headless", rp.call_args[0][0])

    def test_edit_esc_returns_to_menu(self):
        # e opens the editor, esc backs out, ⏎ then runs the untouched preset 0.
        _rc, rp, _rd = self._run(["e", "esc", "enter"])
        rp.assert_called_once()
        self.assertNotIn("--headless", rp.call_args[0][0])   # nothing was toggled

    def test_dashboard_flag_uses_dashboard(self):
        _rc, rp, rd = self._run(["enter"], dashboard=True)
        rd.assert_called_once()
        rp.assert_not_called()

    def test_quit_never_launches(self):
        rc, rp, rd = self._run(["q"])
        self.assertEqual(rc, 0)
        rp.assert_not_called()
        rd.assert_not_called()

    def test_tab_switches_to_real(self):
        _rc, rp, _rd = self._run(["tab", "enter"])        # real preset 0 -> bash run_real.sh
        self.assertEqual(rp.call_args[0][0][0], "bash")


if __name__ == "__main__":
    unittest.main()
