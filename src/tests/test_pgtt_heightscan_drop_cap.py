"""Tests for the PGTT heightscan drop-off clamp (incident 8.15/8.16, F3).

Host-safe (no Isaac/torch deps) -- exercises ``build_heightscan`` directly, the numpy-pure
grid/normalization function in ``go2_locomotion/pgtt_heightmap.py``. Run 11
(run_sim_20260711_234004_424) evidence: isaac_env.jsonl ``pgtt_heightscan`` events showed
``hs_max`` jump 0.0 -> 2.1 and ``action_norm`` 1.1 -> 3.15 the moment the +/-0.5x+/-0.4 m scan
footprint crossed the 2.1 m top-landing edge -- the min-normalization in ``build_heightscan``
shifts every OTHER cell up by however far the single deepest cell reads below the robot base, so
one over-the-edge cell put the whole 99-cell observation wildly out of the training distribution.

The wiring from ``PgttPolicyConfig.heightscan_drop_cap_m`` / ``--pgtt-heightscan-drop-cap-m``
through to this function's ``drop_cap`` argument is verified with static source-scan contract
tests (same technique as ``test_stair_speed_guards.py``'s ``_call_site_lines``), not a runtime
import of ``PgttLocomotionPolicy`` (which pulls in torch / the PGTT net loader).
"""
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # go2_locomotion package lives at the repo root

from go2_locomotion.pgtt_heightmap import (  # noqa: E402
    PGTT_N_COLS,
    PGTT_N_ROWS,
    build_heightscan,
)

_CENTER_ROW = (PGTT_N_ROWS - 1) // 2  # 5
_CENTER_COL = (PGTT_N_COLS - 1) // 2  # 4


def _edge_height_fn(edge_x_m, drop_m):
    """A flat floor (z=0) that steps down by drop_m at world x >= edge_x_m -- mirrors the
    run-11 top-landing edge crossing the scan footprint as the robot rotated/approached it."""
    def height_fn(x, y):
        return -float(drop_m) if x >= edge_x_m else 0.0
    return height_fn


def _rear_stair_height_fn(riser_m=0.175, max_steps=3):
    """A descending staircase BEHIND the robot (x < 0), each 0.1 m row one step down, capped
    at max_steps -- mirrors a crest-straddle where the front is on the landing (x>=0, flat)
    and the rear footprint legitimately reads real risers below the base."""
    def height_fn(x, y):
        if x >= 0.0:
            return 0.0
        steps = min(max_steps, int(round(-x / 0.1)))
        return -float(riser_m) * steps
    return height_fn


def test_no_clamp_by_default_matches_unclamped_output():
    """drop_cap=None (the function's own default) must reproduce the pre-fix behaviour
    exactly -- backward compatible for any other caller/test."""
    hf = _edge_height_fn(edge_x_m=0.15, drop_m=2.1)
    a = build_heightscan((0.0, 0.0), 0.0, hf)
    b = build_heightscan((0.0, 0.0), 0.0, hf, drop_cap=None)
    assert np.array_equal(a, b)


def test_unclamped_run11_edge_reproduces_hs_max_2p1():
    """Reproduces the exact run-11 failure signature: hs_max jumps to ~2.1 with no clamp."""
    hf = _edge_height_fn(edge_x_m=0.15, drop_m=2.1)
    hs = build_heightscan((0.0, 0.0), 0.0, hf)
    assert abs(float(np.max(hs)) - 2.1) < 1e-6, float(np.max(hs))


def test_drop_cap_bounds_hs_max_to_the_cap():
    """With drop_cap=0.6 (the shipped default), the SAME run-11 edge crossing must bound
    hs_max to the cap instead of the full 2.1 m drop."""
    hf = _edge_height_fn(edge_x_m=0.15, drop_m=2.1)
    hs = build_heightscan((0.0, 0.0), 0.0, hf, drop_cap=0.6)
    assert abs(float(np.max(hs)) - 0.6) < 1e-6, float(np.max(hs))
    assert float(np.max(hs)) < 2.1


def test_drop_cap_reports_engagement_via_out_stats():
    hf = _edge_height_fn(edge_x_m=0.15, drop_m=2.1)
    stats = {}
    build_heightscan((0.0, 0.0), 0.0, hf, drop_cap=0.6, out_stats=stats)
    assert stats.get("clamp_engaged") is True
    # 4 forward rows (x=0.2,0.3,0.4,0.5 -- edge_x_m=0.15) x 9 cols = 36 clamped cells.
    assert stats.get("clamp_cells") == 36, stats


def test_drop_cap_out_stats_untouched_when_disabled():
    """drop_cap=None must not touch a caller-supplied out_stats dict at all -- the caller
    checks .get('clamp_engaged') so an untouched dict must read as falsy/absent."""
    hf = _edge_height_fn(edge_x_m=0.15, drop_m=2.1)
    stats = {}
    build_heightscan((0.0, 0.0), 0.0, hf, drop_cap=None, out_stats=stats)
    assert stats == {}
    assert not stats.get("clamp_engaged", False)


def test_drop_cap_no_engagement_on_flat_terrain():
    hf = _edge_height_fn(edge_x_m=0.15, drop_m=0.0)  # no actual drop anywhere
    stats = {}
    hs = build_heightscan((0.0, 0.0), 0.0, hf, drop_cap=0.6, out_stats=stats)
    assert stats.get("clamp_engaged") is False
    assert stats.get("clamp_cells") == 0
    assert np.allclose(hs, 0.0)


def test_drop_cap_preserves_legitimate_stair_reads_at_crest_straddle():
    """Rear cells reading up to ~3 x 0.175 m (0.525 m) of real riser below the base -- well
    within the 0.6 m default cap -- must be UNCHANGED by the clamp (same output with and
    without drop_cap), per the drop_cap docstring's crest-straddle design intent."""
    hf = _rear_stair_height_fn(riser_m=0.175, max_steps=3)
    unclamped = build_heightscan((0.0, 0.0), 0.0, hf, drop_cap=None)
    stats = {}
    clamped = build_heightscan((0.0, 0.0), 0.0, hf, drop_cap=0.6, out_stats=stats)
    assert stats.get("clamp_engaged") is False
    assert np.array_equal(unclamped, clamped)
    # Sanity: the deepest legitimate rear read (3 risers x 0.175 m = 0.525 m) is indeed
    # inside the 0.6 m cap -- else the fixture would not exercise the "preserved" case.
    assert 3 * 0.175 < 0.6


def test_drop_cap_clamps_from_below_only_never_raises_shallower_cells():
    """The clamp must never LOWER a cell that already reads above the floor -- only raise
    (max) cells reading below it."""
    def hf(x, y):
        # A single very deep outlier plus a shallow legitimate dip -- the shallow one must
        # be untouched by the clamp.
        if abs(x - 0.5) < 1e-9 and abs(y - 0.4) < 1e-9:
            return -2.1
        if abs(x - (-0.1)) < 1e-9:
            return -0.2  # shallow, inside the 0.6 m cap
        return 0.0
    stats = {}
    hs = build_heightscan((0.0, 0.0), 0.0, hf, drop_cap=0.6, out_stats=stats)
    assert stats.get("clamp_cells") == 1  # only the single -2.1 outlier
    hs_grid = hs.reshape(PGTT_N_ROWS, PGTT_N_COLS)
    # Row for x=-0.1 is idx_h=6 (p = 5-6 = -1 -> x = -0.1); its shallow -0.2 dip must still
    # read as a relative height distinct from the flat 0.0 cells (i.e. not clamped away).
    assert hs_grid[6, _CENTER_COL] != hs_grid[_CENTER_ROW, _CENTER_COL]


# --------------------------------------------------------------------------------------
# Static wiring contract: PgttPolicyConfig / PgttLocomotionPolicy / go2_control.py /
# isaac_args.py must actually thread heightscan_drop_cap_m through to build_heightscan's
# drop_cap argument (not a tested-but-dead extra parameter). Source-scan only -- no torch /
# Isaac imports (mirrors test_stair_speed_guards.py's _call_site_lines technique).
# --------------------------------------------------------------------------------------

_PGTT_POLICY_PY = os.path.join(REPO, "go2_locomotion", "pgtt_locomotion_policy.py")
_GO2_CONTROL_PY = os.path.join(REPO, "sim", "isaac", "env", "go2_control.py")
_ISAAC_ARGS_PY = os.path.join(REPO, "sim", "isaac", "isaac_args.py")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_pgtt_policy_config_has_heightscan_drop_cap_default_0p6():
    src = _read(_PGTT_POLICY_PY)
    m = re.search(r"heightscan_drop_cap_m:\s*Optional\[float\]\s*=\s*([\d.]+)", src)
    assert m is not None, "PgttPolicyConfig.heightscan_drop_cap_m field not found"
    assert abs(float(m.group(1)) - 0.6) < 1e-9, m.group(1)


def test_pgtt_locomotion_policy_passes_drop_cap_to_build_heightscan():
    src = _read(_PGTT_POLICY_PY)
    call_re = re.compile(r"build_heightscan\(.*?\)", re.DOTALL)
    m = call_re.search(src)
    assert m is not None, "build_heightscan(...) call not found in pgtt_locomotion_policy.py"
    call_text = m.group(0)
    assert "drop_cap=self.config.heightscan_drop_cap_m" in call_text, call_text
    assert "out_stats=" in call_text, call_text


def test_isaac_args_defines_pgtt_heightscan_drop_cap_flag_default_0p6():
    src = _read(_ISAAC_ARGS_PY)
    assert "--pgtt-heightscan-drop-cap-m" in src
    m = re.search(r"--pgtt-heightscan-drop-cap-m.*?default=([\d.]+)", src, re.DOTALL)
    assert m is not None
    assert abs(float(m.group(1)) - 0.6) < 1e-9, m.group(1)


def test_go2_control_wires_drop_cap_arg_into_pgtt_policy_config():
    src = _read(_GO2_CONTROL_PY)
    assert "pgtt_heightscan_drop_cap_m" in src
    assert "heightscan_drop_cap_m=" in src


if __name__ == "__main__":
    test_no_clamp_by_default_matches_unclamped_output()
    test_unclamped_run11_edge_reproduces_hs_max_2p1()
    test_drop_cap_bounds_hs_max_to_the_cap()
    test_drop_cap_reports_engagement_via_out_stats()
    test_drop_cap_out_stats_untouched_when_disabled()
    test_drop_cap_no_engagement_on_flat_terrain()
    test_drop_cap_preserves_legitimate_stair_reads_at_crest_straddle()
    test_drop_cap_clamps_from_below_only_never_raises_shallower_cells()
    test_pgtt_policy_config_has_heightscan_drop_cap_default_0p6()
    test_pgtt_locomotion_policy_passes_drop_cap_to_build_heightscan()
    test_isaac_args_defines_pgtt_heightscan_drop_cap_flag_default_0p6()
    test_go2_control_wires_drop_cap_arg_into_pgtt_policy_config()
    print("ALL PGTT HEIGHTSCAN DROP-CAP TESTS PASS")
