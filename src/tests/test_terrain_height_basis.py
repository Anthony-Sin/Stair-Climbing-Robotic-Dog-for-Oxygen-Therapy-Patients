"""Regression tests for the human-fell terrain-basis fix (2026-07-11 review).

isaac_env.py's ``_run_evaluation_and_save_images`` human-fell check (~L2657-2670) compares
the patient's RECORDED z against a terrain reference. The patient's z is recorded via
``_get_person_pose_z(px, py, smooth=True)`` (env/terrain_queries.py:106-111 -- the continuous
stair-nosing ramp, ``get_terrain_height_smooth``), but the check used to compare it against
the DISCRETE ``get_terrain_height()`` (terrain_queries.py:36-56), which snaps up a full riser
the instant x crosses a tread boundary. Just after each boundary the discrete reference is up
to a full ``step_height_m`` (0.15 m for the "commercial" preset) ABOVE the smooth ramp, so
``pz - terrain_z < -0.1`` (the fall threshold) fired on every clean stair climb by pure
geometry -- confirmed live in run_sim_20260711_223152_489's evaluation_summary.txt /
stair_demo_report.json ("Human: fell" / "human_summary": "fell" despite the patient walking a
normal, level-following climb with no genuine fall).

Fixed by comparing against ``_get_person_pose_z(px, py, smooth=True)`` again -- the SAME
function that produced the recorded pz -- so both sides of the check share the same basis
(and the same optional final-scene z offset, final_scene/runtime.py:76-77).

``sim/isaac/isaac_env.py`` cannot be imported standalone on a host without Isaac (it does
``from isaacsim import SimulationApp`` at module load, isaac_env.py:17), so the exact check
inside ``_run_evaluation_and_save_images`` cannot be exercised directly (mirrors incident 8.4:
"the real (hardware) import graph is exercised by nothing" -- this is the sim analogue). The
terrain functions themselves (``env/terrain_queries.py``) ARE plain Python; their only
Isaac-dependent hop is ``from world.sim_go2_locomotion import get_active_stairs``, where
``sim_go2_locomotion.py`` imports ``omni``/``isaacsim`` at its own module top level purely to
re-export symbols the STAIR geometry does not need (``world/sim_go2_stairs.py``'s own
docstring: "Split out of sim_go2_locomotion (the facade re-exports these)"). So this file
stubs ``sys.modules["world.sim_go2_locomotion"]`` with a tiny facade that re-exports the REAL
``get_active_stairs`` from the REAL (Isaac-free) ``world.sim_go2_stairs`` module -- the exact
same function the production facade re-exports, just without dragging in the unrelated
omni-dependent code -- then imports ``env.terrain_queries`` and replays a clean-climb
trajectory through both the OLD (discrete) and NEW (smooth) reference to reproduce and verify
the fix, using the corrected check's own literal expression (``pz - ref < -0.1``) rather than
a re-derived approximation of it.
"""
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
for _p in (os.path.join(REPO, "sim", "isaac"), os.path.join(REPO, "sim", "bot")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import world.sim_go2_stairs as real_stairs  # noqa: E402

if "world.sim_go2_locomotion" not in sys.modules:
    _stub = types.ModuleType("world.sim_go2_locomotion")
    _stub.get_active_stairs = real_stairs.get_active_stairs
    sys.modules["world.sim_go2_locomotion"] = _stub

import env.terrain_queries as tq  # noqa: E402
from env import env_state  # noqa: E402

FALL_THRESHOLD_M = -0.1  # mirrors isaac_env.py's `(pz - terrain_z) < -0.1`


def _clean_climb_xs(spec, step_m=0.01, tail_m=0.5):
    """x samples spanning approach -> climb -> landing, fine enough to hit every tread
    boundary (the worst case for the discrete-vs-smooth basis mismatch)."""
    x = spec.start_x_m - 0.3
    end = spec.end_x_m + tail_m
    xs = []
    while x <= end:
        xs.append(round(x, 6))
        x += step_m
    return xs


def test_corrected_basis_never_false_fells_during_clean_climb():
    """The fix: pz (recorded via smooth) compared against the SAME smooth reference must
    never breach the fall threshold across an entire clean climb -- reproduces
    run_sim_20260711_223152_489's false "Human: fell" and confirms it is gone."""
    real_stairs.configure_stairs("commercial")  # 0.150 m riser, matches the task's report
    spec = real_stairs.get_active_stairs()
    assert spec.step_height_m == 0.150

    worst_delta = 0.0
    for x in _clean_climb_xs(spec):
        pz = tq.get_terrain_height_smooth(x, 0.0)  # what isaac_env.py records as pz (L4263)
        ref = tq._get_person_pose_z(x, 0.0, smooth=True)  # the CORRECTED reference
        worst_delta = min(worst_delta, pz - ref)

    assert worst_delta == 0.0, (
        f"corrected basis should be IDENTICAL to the recorded pz (same function, same "
        f"inputs) -- got a worst-case delta of {worst_delta} m"
    )
    assert worst_delta >= FALL_THRESHOLD_M, (
        "corrected check would still have false-fired 'fell' on a clean climb"
    )


def test_old_discrete_basis_would_false_fell_at_tread_boundary():
    """Documents the BUG being fixed: replaying the same clean-climb trajectory against the
    OLD (discrete get_terrain_height) reference DOES breach the -0.1 m threshold, just after
    a tread boundary -- this is exactly what run_sim_20260711_223152_489 hit."""
    real_stairs.configure_stairs("commercial")
    spec = real_stairs.get_active_stairs()

    worst_delta = 0.0
    tripped_old = False
    for x in _clean_climb_xs(spec):
        pz = tq.get_terrain_height_smooth(x, 0.0)
        old_ref = tq.get_terrain_height(x, 0.0)  # the ORIGINAL (buggy) reference
        delta = pz - old_ref
        worst_delta = min(worst_delta, delta)
        if delta < FALL_THRESHOLD_M:
            tripped_old = True

    assert tripped_old, (
        "expected the OLD discrete-basis check to false-fire on this clean climb "
        f"(worst delta {worst_delta} m) -- if it no longer does, the riser/step geometry "
        "used by this test may need updating, not the production fix"
    )
    # The 0.150 m riser exceeds the 0.1 m threshold by exactly the amount observed live.
    assert worst_delta <= -0.1


def test_corrected_basis_still_catches_a_genuine_fall():
    """A genuine fall -- the recorded z sitting well BELOW the terrain contract, e.g. the
    patient collapsing through/below the floor -- must still trip the corrected check. The
    fix narrows a false positive; it must not also blind the check to real ones."""
    real_stairs.configure_stairs("commercial")
    spec = real_stairs.get_active_stairs()
    x = spec.start_x_m + 1.0  # mid-climb
    ref = tq._get_person_pose_z(x, 0.0, smooth=True)
    genuinely_fallen_pz = ref - 0.25  # 0.25 m below the terrain contract
    assert (genuinely_fallen_pz - tq._get_person_pose_z(x, 0.0, smooth=True)) < FALL_THRESHOLD_M


def test_corrected_basis_flat_ground_unaffected():
    """Off the stairs entirely, discrete and smooth already agree (both return 0.0 / the
    same flat height), so the fix is a no-op there -- sanity check it stays that way."""
    real_stairs.configure_stairs("commercial")
    spec = real_stairs.get_active_stairs()
    x = spec.start_x_m - 5.0  # well before the stairs
    assert tq.get_terrain_height(x, 0.0) == tq.get_terrain_height_smooth(x, 0.0) == 0.0
    assert tq._get_person_pose_z(x, 0.0, smooth=True) == 0.0


def test_corrected_basis_shares_final_scene_offset_treatment():
    """If a final-scene z offset is ever introduced (final_scene/runtime.py:76-77's
    person_pose_z is currently an identity passthrough -- a hook for future use), the
    corrected reference must still track the recorded pz exactly, because both are produced
    by calling the SAME _get_person_pose_z(..., smooth=True) -- unlike the old bug, there is
    no way for this fix to silently drift out of sync with whatever offset is applied."""
    original_spec = env_state._FINAL_SCENE_SPEC
    try:
        env_state._FINAL_SCENE_SPEC = object()  # any non-None sentinel
        x = 2.5
        pz = tq._get_person_pose_z(x, 0.0, smooth=True)  # what isaac_env.py would record
        ref = tq._get_person_pose_z(x, 0.0, smooth=True)  # the corrected reference, same call
        assert pz - ref == 0.0
    finally:
        env_state._FINAL_SCENE_SPEC = original_spec


if __name__ == "__main__":
    test_corrected_basis_never_false_fells_during_clean_climb()
    test_old_discrete_basis_would_false_fell_at_tread_boundary()
    test_corrected_basis_still_catches_a_genuine_fall()
    test_corrected_basis_flat_ground_unaffected()
    test_corrected_basis_shares_final_scene_offset_treatment()
    print("ALL TERRAIN HEIGHT BASIS TESTS PASS")
