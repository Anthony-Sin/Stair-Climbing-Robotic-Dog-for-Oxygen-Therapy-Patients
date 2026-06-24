"""Host-side tests for the stair-sweep presentation pack (sim/analysis/sweep_present.py).

No Isaac / GPU needed. Covers:
  * analyze_climb.analyze_run verdict correctness on synthetic fall_diag streams
    (CLEAN / COLLIDED / FELL / DID NOT REACH) -- the SINGLE source of truth the
    presenter and the live sim share.
  * scan_scene_events reading the stair config out of the `sim` dict (not `event`).
  * build_filtergraph as a PURE string (layout, speed, captions, BEST badge) -- no ffmpeg.
  * _compute_speeds math (sync vs uniform; missing durations).
  * End-to-end run(): CSV/JSON data section + matplotlib graphs + stats card.
  * ffmpeg-gated: a real 2x3 montage renders to a valid mp4 (skipped without ffmpeg).

Run directly (python tests/test_sweep_present.py) or via pytest.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ANALYSIS = os.path.join(_REPO, "sim", "analysis")
if _ANALYSIS not in sys.path:
    sys.path.insert(0, _ANALYSIS)

import analyze_climb  # noqa: E402
import sweep_present as sp  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic run-dir builder (mimics the real structured logger: payload under `sim`)
# ---------------------------------------------------------------------------
def _traj(kind, i, n, vid_seconds):
    f = i / (n - 1)
    if kind == "clean":
        x, h, pitch, tilt = -4.0 + f * 10.5, 0.30, (4.0 if i % 2 else -4.0), 6.0
    elif kind == "collided":
        x, h, pitch, tilt = -4.0 + f * 7.5, (0.30 - (0.16 if f > 0.7 else 0.0)), (-16.0 if f > 0.5 else -4.0), 22.0
    elif kind == "fell":
        x, h, pitch, tilt = -4.0 + f * 6.5, 0.28, -10.0, (6.0 + (80.0 if f > 0.8 else 0.0))
    else:  # no_reach
        x, h, pitch, tilt = -4.0 + f * 5.0, 0.31, 2.0, 5.0
    return {"t": round(f * vid_seconds * 2.0, 3), "x": round(x, 3), "y": 0.05, "h": round(h, 3),
            "roll": 1.0, "pitch": pitch, "yaw": 3.0, "tilt_deg": tilt, "vx": 0.4, "wz": 0.0,
            "gap_m": 1.5, "stairs_action_active": x > 1.5}


def make_run(base, name, height, kind, with_video=False, vid_seconds=2):
    rd = os.path.join(base, name)
    os.makedirs(os.path.join(rd, "debug"), exist_ok=True)
    os.makedirs(os.path.join(rd, "videos"), exist_ok=True)
    lines = [{"event": {"action": "stair_preset_configured"},
              "sim": {"preset": "commercial", "step_count": 14, "step_height_m": height,
                      "step_depth_m": 0.305, "top_height_m": round(14 * height, 3), "handrail": True}}]
    n = 60
    for i in range(n):
        lines.append({"event": {"action": "fall_diag"}, "sim": _traj(kind, i, n, vid_seconds)})
    if kind == "clean":
        lines.append({"event": {"action": "robot_reached_stair_waypoint"},
                      "sim": {"t": round(vid_seconds * 2.0, 3), "x": 6.6, "h": 0.30}})
    with open(os.path.join(rd, "debug", "isaac_env.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")
    if with_video:
        out = os.path.join(rd, "videos", "scene_view.mp4")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"testsrc=size=320x180:rate=30:duration={vid_seconds}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(vid_seconds), out], check=True)
    return rd


# ---------------------------------------------------------------------------
# analyze_run verdicts (shared source of truth)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind,prefix", [
    ("clean", "CLEAN"),
    ("collided", "COLLIDED"),
    ("fell", "FELL"),
    ("no_reach", "DID NOT REACH"),
])
def test_analyze_run_verdicts(kind, prefix):
    with tempfile.TemporaryDirectory() as td:
        rd = make_run(td, "run_sim_x", 0.15, kind)
        res = analyze_climb.analyze_run(rd)
        assert res["stats"] is not None
        assert res["stats"]["verdict"].startswith(prefix), res["stats"]["verdict"]


def test_analyze_run_empty():
    with tempfile.TemporaryDirectory() as td:
        rd = os.path.join(td, "run_sim_empty")
        os.makedirs(os.path.join(rd, "debug"))
        open(os.path.join(rd, "debug", "isaac_env.jsonl"), "w").close()
        res = analyze_climb.analyze_run(rd)
        assert res["stats"] is None and res["rows"] == []


def test_print_report_keeps_grep_contract(capsys):
    # run_stair_sweep.ps1 greps these exact substrings -- lock them in.
    with tempfile.TemporaryDirectory() as td:
        rd = make_run(td, "run_sim_x", 0.15, "collided")
        analyze_climb.print_report(rd, analyze_climb.analyze_run(rd))
    out = capsys.readouterr().out
    assert "VERDICT:" in out
    assert "step-runs past base" in out


# ---------------------------------------------------------------------------
# scene-event parsing
# ---------------------------------------------------------------------------
def test_scan_scene_events_reads_sim_payload():
    with tempfile.TemporaryDirectory() as td:
        rd = make_run(td, "run_sim_x", 0.178, "clean")
        ev = sp.scan_scene_events(os.path.join(rd, "debug", "isaac_env.jsonl"))
        assert ev["step_height_m"] == 0.178
        assert ev["step_count"] == 14
        assert ev["step_depth_m"] == 0.305
        assert ev["waypoint_status"].startswith("PASS")
        assert ev["waypoint_time"] is not None


# ---------------------------------------------------------------------------
# pure montage helpers
# ---------------------------------------------------------------------------
def test_compute_speeds_sync():
    ms, pads = sp._compute_speeds([10.0, 20.0], 20.0, "sync")
    assert ms == [2.0, 1.0]
    assert pads == [0.0, 0.0]


def test_compute_speeds_uniform_preserves_relative():
    ms, pads = sp._compute_speeds([10.0, 20.0], 20.0, "uniform")
    assert ms == [1.0, 1.0]          # single factor (ref = longest)
    assert pads == [10.0, 0.0]       # the shorter clip freezes for the remainder


def test_compute_speeds_missing_duration():
    ms, pads = sp._compute_speeds([None, 10.0], 20.0, "sync")
    assert ms[0] is None and ms[1] == 2.0
    assert pads[0] == 20.0


def test_build_filtergraph_structure():
    tiles = [
        {"kind": "video", "input_index": 0, "x": 0, "y": 0, "m": 0.25, "pad": 0.0,
         "fit": "cover", "caption": "cap0.txt", "best": True, "font": "font.ttf"},
        {"kind": "placeholder", "input_index": None, "x": 960, "y": 0, "caption": "cap1.txt",
         "font": "font.ttf"},
        {"kind": "image", "input_index": 1, "x": 960, "y": 720},
    ]
    fg = sp.build_filtergraph(tiles, 20.0)
    assert fg.startswith("color=c=black:s=1920x1080")
    assert "setpts=PTS*0.250000" in fg          # speed-up applied
    assert "crop=960:360" in fg                  # cover fit
    assert "textfile=cap0.txt" in fg             # caption wired
    assert "text=BEST" in fg and "gold" in fg    # best badge
    assert "color=c=0x141414" in fg              # placeholder source (valid ffmpeg color)
    assert "overlay=x=960:y=720" in fg           # stats-card cell position
    assert fg.rstrip().endswith("[out]")         # final output label


def test_build_filtergraph_letterbox():
    tiles = [{"kind": "video", "input_index": 0, "x": 0, "y": 0, "m": 1.0, "pad": 0.0,
              "fit": "letterbox", "caption": None, "best": False, "font": "font.ttf"}]
    fg = sp.build_filtergraph(tiles, 5.0)
    assert "force_original_aspect_ratio=decrease" in fg and "pad=960:360" in fg


# ---------------------------------------------------------------------------
# end-to-end data section + graphs (no ffmpeg needed)
# ---------------------------------------------------------------------------
def test_run_writes_data_section_and_graphs():
    with tempfile.TemporaryDirectory() as td:
        dirs = [
            make_run(td, "run_sim_1", 0.10, "clean"),
            make_run(td, "run_sim_2", 0.15, "collided"),
            make_run(td, "run_sim_3", 0.198, "no_reach"),
        ]
        out = os.path.join(td, "pres")
        rc = sp.main(["--run-dirs", *dirs, "--heights", "0.1", "0.15", "0.198",
                      "--out", out, "--no-montage", "--no-clips"])
        assert rc == 0
        csv_path = os.path.join(out, "sweep_summary.csv")
        assert os.path.exists(csv_path)
        assert os.path.exists(os.path.join(out, "sweep_summary.json"))
        assert os.path.exists(os.path.join(out, "stats_card.png"))
        for g in ("g1_climb_profile.png", "g2_steps_vs_riser.png", "g3_stability_vs_riser.png",
                  "g4_reach_vs_riser.png", "g5_dashboard.png"):
            assert os.path.exists(os.path.join(out, "graphs", g)), g
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        risers = sorted(float(r["riser_m"]) for r in rows)
        assert risers == [0.1, 0.15, 0.198]
        by = {round(float(r["riser_m"]), 3): r for r in rows}
        assert by[0.1]["verdict_short"] == "CLEAN"
        assert by[0.15]["verdict_short"] == "COLLIDED"
        assert by[0.198]["verdict_short"] == "NO REACH"


def test_manifest_single_episode_collapse():
    # PowerShell ConvertTo-Json collapses a 1-elem array to an object -- the reader must cope.
    with tempfile.TemporaryDirectory() as td:
        rd = make_run(td, "run_sim_1", 0.15, "collided")
        man = {"stamp": "t", "episodes": {"height": 0.15, "label": "x", "run_dir": rd, "waypoint": "?"}}
        mp = os.path.join(td, "manifest.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(man, f)
        out = os.path.join(td, "pres")
        rc = sp.main(["--manifest", mp, "--out", out, "--no-montage", "--no-clips", "--no-graphs"])
        assert rc == 0
        with open(os.path.join(out, "sweep_summary.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and rows[0]["verdict_short"] == "COLLIDED"


# ---------------------------------------------------------------------------
# ffmpeg-gated: real montage render
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="ffmpeg/ffprobe not on PATH")
def test_montage_renders_valid_mp4():
    with tempfile.TemporaryDirectory() as td:
        dirs = [
            make_run(td, "run_sim_1", 0.10, "clean", with_video=True, vid_seconds=2),
            make_run(td, "run_sim_2", 0.15, "collided", with_video=True, vid_seconds=1),
        ]
        out = os.path.join(td, "pres")
        rc = sp.main(["--run-dirs", *dirs, "--heights", "0.1", "0.15",
                      "--out", out, "--montage-seconds", "3", "--no-clips"])
        assert rc == 0
        montage = os.path.join(out, "stair_sweep_montage.mp4")
        assert os.path.exists(montage)
        w = sp._ffprobe(montage, "stream=width")
        d = sp.ffprobe_duration(montage)
        assert w == "1920"
        assert d is not None and 2.5 <= d <= 3.6
        # transient ffmpeg workdir is cleaned up
        assert not os.path.exists(os.path.join(out, "_montage"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
