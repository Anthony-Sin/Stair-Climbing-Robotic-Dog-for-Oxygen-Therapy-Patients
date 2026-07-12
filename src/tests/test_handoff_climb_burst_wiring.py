"""Static wiring contract for the post-ENGAGE blind-mount speed step-down (task, 2026-07-12,
run 32 review -- "the dog starts way too late on the stairs" -- CLAUDE.md incident 8.15/8.16
continuation).

The decision logic itself (``go2_locomotion.locomotion_arbiter.blind_mount_climb_vx_floor``)
is a pure host-safe function, fully unit-tested in ``test_locomotion_arbiter.py``. This file
covers the WIRING around it -- the two new CLI args, the import, and the call site inside
``sim/isaac/isaac_env.py``'s ``_step_go2_locomotion`` blind_rl hot-swap branch -- which
imports isaacsim/omniverse and cannot be imported on a plain host at all (incident 8.4).
Mirrors ``test_park_request_wiring.py``'s source-scan technique exactly.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ISAAC_ENV_PY = os.path.join(REPO, "sim", "isaac", "isaac_env.py")
_ISAAC_ARGS_PY = os.path.join(REPO, "sim", "isaac", "isaac_args.py")
_HANDOFF_CONFIG_PY = os.path.join(REPO, "go2_locomotion", "handoff_config.py")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------------------
# sim/isaac/isaac_args.py -- the two new CLI args.
# --------------------------------------------------------------------------------------

def test_isaac_args_defines_handoff_climb_burst_sec():
    src = _read(_ISAAC_ARGS_PY)
    assert '"--handoff-climb-burst-sec"' in src, (
        "isaac_args.py must define --handoff-climb-burst-sec (the momentum-burst window "
        "duration, sim-seconds)"
    )
    idx = src.find('"--handoff-climb-burst-sec"')
    window = src[idx: idx + 200]
    assert "default=2.0" in window, "the burst window must default to 2.0 sim-seconds"


def test_isaac_args_defines_handoff_climb_blind_vx():
    src = _read(_ISAAC_ARGS_PY)
    assert '"--handoff-climb-blind-vx"' in src, (
        "isaac_args.py must define --handoff-climb-blind-vx (the post-burst blind-mount "
        "forward floor)"
    )
    idx = src.find('"--handoff-climb-blind-vx"')
    window = src[idx: idx + 200]
    assert "default=0.30" in window or "default=0.3" in window, (
        "the blind-mount floor must default to 0.30 m/s (matches --stair-loss-forward-floor)"
    )


# --------------------------------------------------------------------------------------
# sim/isaac/isaac_env.py -- import + call-site wiring.
# --------------------------------------------------------------------------------------

def test_isaac_env_imports_blind_mount_climb_vx_floor():
    src = _read(_ISAAC_ENV_PY)
    assert "blind_mount_climb_vx_floor" in src, (
        "isaac_env.py must import/use blind_mount_climb_vx_floor from "
        "go2_locomotion.locomotion_arbiter"
    )
    import_idx = src.find("from go2_locomotion.locomotion_arbiter import")
    assert import_idx != -1
    import_line_end = src.find("\n", import_idx)
    # The import spans to the closing paren; grab a generous window for the multi-line form.
    window = src[import_idx: import_idx + 200]
    assert "blind_mount_climb_vx_floor" in window, (
        "blind_mount_climb_vx_floor must be imported alongside arbitrate_climb_vx/"
        "arbitrate_climb_wz, not re-implemented inline in isaac_env.py"
    )


def test_isaac_env_blind_rl_branch_calls_blind_mount_climb_vx_floor_with_sim_time():
    """The call must read climb_elapsed_sec from the HandoffController's returned dict (the
    dt-accumulated sim-time watchdog, incident 8.6) -- NEVER time.monotonic()/perf_counter()
    or a frame count -- and forward person_detected + the two new CLI args."""
    src = _read(_ISAAC_ENV_PY)
    call_idx = src.find("blind_mount_climb_vx_floor(\n")
    assert call_idx != -1, "expected a blind_mount_climb_vx_floor(...) call site"
    close_idx = src.find(")\n", call_idx)
    window = src[call_idx: close_idx + 1]
    assert '_ho.get("climb_elapsed_sec"' in window, (
        "climb_elapsed_sec must come from the HandoffController's returned dict (sim-time), "
        "not a wall-clock read"
    )
    assert "time.monotonic()" not in window and "time.perf_counter()" not in window, (
        "the blind-mount step-down must not be timed against wall-clock (incident 8.6)"
    )
    assert "handoff_climb_burst_sec" in window
    assert "handoff_climb_blind_vx" in window
    assert "person_detected" in window


def test_isaac_env_step_down_result_feeds_arbitrate_climb_vx_climb_vx_argument():
    """The step-down's output must be what arbitrate_climb_vx receives as its climb_vx floor
    (not bypassed, not added as a second independent floor)."""
    src = _read(_ISAAC_ENV_PY)
    helper_idx = src.find("_climb_floor_vx = blind_mount_climb_vx_floor(")
    assert helper_idx != -1, (
        "expected the step-down result assigned to a local (_climb_floor_vx) before being "
        "passed into arbitrate_climb_vx"
    )
    arb_idx = src.find("_cvx = arbitrate_climb_vx(", helper_idx)
    assert arb_idx != -1 and arb_idx > helper_idx, (
        "arbitrate_climb_vx must be called AFTER the step-down helper computes its result"
    )
    close_idx = src.find(")\n", arb_idx)
    window = src[arb_idx: close_idx + 1]
    assert "climb_vx=_climb_floor_vx" in window, (
        "arbitrate_climb_vx's climb_vx= argument must be fed the step-down helper's result"
    )


def test_isaac_env_brake_scale_composition_unchanged_at_the_call_site():
    """The existing mid-climb gap-brake product (_climb_vx_brake_scale * _gt_gap_scale) must
    still be computed and passed into the step-down helper's brake_scale= argument -- the
    step-down composes with it (via min() inside the helper), it does not replace it."""
    src = _read(_ISAAC_ENV_PY)
    helper_idx = src.find("_climb_floor_vx = blind_mount_climb_vx_floor(")
    assert helper_idx != -1
    close_idx = src.find(")\n", helper_idx)
    window = src[helper_idx: close_idx + 1]
    assert "brake_scale=(_climb_vx_brake_scale * _gt_gap_scale)" in window, (
        "the step-down helper must receive the SAME combined brake product the pre-existing "
        "climb_vx composition used, so behavior is numerically identical whenever the "
        "step-down cap does not bind (person visible or still in the burst window)"
    )


# --------------------------------------------------------------------------------------
# go2_locomotion/handoff_config.py -- the paired entry-lead reduction.
# --------------------------------------------------------------------------------------

def test_stair_entry_min_lead_m_reduced_to_1_9():
    src = _read(_HANDOFF_CONFIG_PY)
    assert "stair_entry_min_lead_m: float = 1.9" in src, (
        "the S1 stair-entry head-start gate must be reduced 2.2 -> 1.9 (task, 2026-07-12, "
        "run 32 review), paired with the blind-mount step-down that bounds the resulting "
        "blind-carry closure instead"
    )
