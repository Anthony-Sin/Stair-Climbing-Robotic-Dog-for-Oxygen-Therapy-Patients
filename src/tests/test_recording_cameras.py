"""Host-side (no Isaac) tests for the smarter recording cameras.

Covers the pure logic of the camera improvements:
  * RecordingWriter backend selection + resolution math (HD ffmpeg vs mp4v cap).
  * The autofit "zoom-to-fit" overview math in CinematicConductor: the bounding
    sphere of {robot, patient, stair span} is solved to a dolly distance + fixed
    lens so EVERY subject projects inside the vertical FOV -- i.e. it never clips.
  * The default-scene camera bundle wiring (autofit overview + cinematic chase).

The stateful per-frame apply/latch (mode="autofit") needs the Isaac USD stage, so
it is exercised by the Isaac run, not here.

Run: python tests/test_recording_cameras.py
"""

import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))

import recording_writer as rw  # noqa: E402
from recording_writer import RecordingWriter, fit_resolution, parse_resolution  # noqa: E402
from recording_cameras import build_default_camera_spec  # noqa: E402
from final_scene.isaac_mount import CinematicDirector  # noqa: E402


# ---------------------------------------------------------------------------
# RecordingWriter: backend selection + resolution policy
# ---------------------------------------------------------------------------
def test_parse_resolution():
    assert parse_resolution("1920x1080") == (1920, 1080)
    assert parse_resolution("1280X720") == (1280, 720)
    assert parse_resolution("bad") == (1280, 720)          # default
    assert parse_resolution("101x51") == (100, 50)          # even-snapped


def test_fit_resolution_within_budget_unchanged():
    # 1280x720 = 921600 px; budget 1280*720 -> unchanged (even).
    assert fit_resolution(1280, 720, 1280 * 720) == (1280, 720)


def test_fit_resolution_downscales_aspect_preserving():
    # 1920x1080 into the mp4v budget (~768x432).
    w, h = fit_resolution(1920, 1080, 768 * 432)
    assert w * h <= 768 * 432
    assert w % 2 == 0 and h % 2 == 0
    assert abs((w / h) - (1920 / 1080)) < 0.02          # 16:9 preserved


def test_backend_order_respects_encoder_choice():
    assert RecordingWriter("p.mp4", 30, encoder="mp4v")._backend_order() == ("mp4v",)
    assert RecordingWriter("p.mp4", 30, encoder="ffmpeg")._backend_order() == ("ffmpeg", "mp4v")
    assert RecordingWriter("p.mp4", 30, encoder="auto")._backend_order() == ("ffmpeg", "mp4v")


def test_resolve_ffmpeg_prefers_path(monkeypatch):
    monkeypatch.setattr(rw.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert rw.resolve_ffmpeg() == "/usr/bin/ffmpeg"


def test_release_before_open_is_safe():
    w = RecordingWriter(os.path.join(REPO, "does_not_matter.mp4"), 30)
    assert w.started is False
    assert w.release() is False          # nothing was written; no file touched


# ---------------------------------------------------------------------------
# Autofit overview: bounding sphere + dolly solve + never-clip guarantee
# ---------------------------------------------------------------------------
def _overview_camera():
    spec = build_default_camera_spec(overview_mode="autofit")
    cam = next(c for c in spec.wall_recording_cameras if c.mode == "autofit")
    return cam


def _half_vfov(cam):
    return math.atan(float(cam.vertical_aperture_mm) / (2.0 * float(cam.focal_length_mm)))


def _angle_off_axis(eye, target, p):
    ax = [target[i] - eye[i] for i in range(3)]
    an = math.sqrt(sum(c * c for c in ax))
    ax = [c / an for c in ax]
    v = [p[i] - eye[i] for i in range(3)]
    vn = math.sqrt(sum(c * c for c in v))
    cos_ang = sum(v[i] * ax[i] for i in range(3)) / vn
    return math.acos(max(-1.0, min(1.0, cos_ang)))


def test_bounding_sphere_center_and_radius():
    pts = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 2.0), (0.0, 2.0, 2.0)]
    center, radius = CinematicDirector._bounding_sphere(pts)
    assert center == (2.0, 1.0, 1.0)
    # half the bbox diagonal (4 x 2 x 2)
    assert abs(radius - 0.5 * math.sqrt(16 + 4 + 4)) < 1e-9


def test_autofit_points_lift_robot_to_torso_height():
    director = CinematicDirector(build_default_camera_spec())
    pts = director._autofit_points((1.0, 0.0, 0.05), (2.0, 0.0, 0.9), terrain_z=0.0,
                                   subject_points=[(3.0, 0.0, 0.5)])
    # robot z floored to terrain + torso min (>0.05), patient + stair point preserved.
    assert pts[0][2] > 0.4
    assert (2.0, 0.0, 0.9) in pts
    assert (3.0, 0.0, 0.5) in pts


def test_autofit_solve_fixed_lens_and_distance_clamp():
    cam = _overview_camera()
    director = CinematicDirector(build_default_camera_spec())
    pts = [(-4.5, 0.0, 0.45), (-3.5, 0.0, 0.9), (2.0, 0.0, 0.0), (6.27, 0.0, 2.1)]
    eye, target, focal, dist = director._autofit_solve(cam, pts)
    assert focal == cam.focal_length_mm                          # fixed lens; dolly only
    assert cam.autofit_min_distance_m <= dist <= cam.autofit_max_distance_m
    # target is the bbox centre
    center, _ = CinematicDirector._bounding_sphere(pts)
    assert all(abs(target[i] - center[i]) < 1e-9 for i in range(3))


def test_autofit_never_clips_subjects():
    """Every subject projects inside the vertical half-FOV from the solved pose."""
    cam = _overview_camera()
    director = CinematicDirector(build_default_camera_spec())
    pts = [(-4.5, 0.0, 0.45), (-3.5, 0.0, 0.9), (2.0, 0.0, 0.0), (6.27, 0.0, 2.1)]
    eye, target, _focal, dist = director._autofit_solve(cam, pts)
    half = _half_vfov(cam)
    for p in pts:
        assert _angle_off_axis(eye, target, p) <= half + 1e-6


def test_autofit_wider_spread_dollies_farther():
    cam = _overview_camera()
    director = CinematicDirector(build_default_camera_spec())
    tight = [(0.0, 0.0, 0.5), (1.0, 0.0, 0.6)]
    wide = [(-4.5, 0.0, 0.5), (6.27, 0.0, 2.1)]
    _, _, _, dist_tight = director._autofit_solve(cam, tight)
    _, _, _, dist_wide = director._autofit_solve(cam, wide)
    assert dist_wide > dist_tight


def test_autofit_huge_spread_clamps_to_max():
    cam = _overview_camera()
    director = CinematicDirector(build_default_camera_spec())
    huge = [(-100.0, -100.0, 0.0), (100.0, 100.0, 100.0)]
    _, _, _, dist = director._autofit_solve(cam, huge)
    assert dist == cam.autofit_max_distance_m


# ---------------------------------------------------------------------------
# Default-scene bundle wiring
# ---------------------------------------------------------------------------
def test_bundle_roles_and_modes():
    spec = build_default_camera_spec(overview_mode="autofit")
    roles = {c.recording_role: c.mode for c in spec.wall_recording_cameras}
    assert roles == {"topdown": "autofit", "scene_view": "chase"}
    assert spec.wall_camera_parent_path.startswith("/World/")


def test_bundle_fixed_overview_sets_static():
    spec = build_default_camera_spec(overview_mode="fixed")
    overview = next(c for c in spec.wall_recording_cameras if c.recording_role == "topdown")
    assert overview.autofit_static is True
    spec2 = build_default_camera_spec(overview_mode="autofit")
    overview2 = next(c for c in spec2.wall_recording_cameras if c.recording_role == "topdown")
    assert overview2.autofit_static is False


def test_bundle_chase_uses_view_camera_params():
    spec = build_default_camera_spec(chase_distance_m=4.0, chase_height_m=2.0, chase_side_m=0.5)
    chase = next(c for c in spec.wall_recording_cameras if c.recording_role == "scene_view")
    assert chase.chase_distance_m == 4.0
    assert chase.chase_height_m == 2.0
    assert chase.chase_side_m == 0.5


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in fns:
        try:
            # crude monkeypatch shim for the lone fixture-using test
            if "monkeypatch" in fn.__code__.co_varnames:
                class _MP:
                    def setattr(self, obj, name, val):
                        setattr(obj, name, val)
                fn(_MP())
            else:
                fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
