"""Static wiring contract for the `park_request` UDP payload field (task, 2026-07-12, runs
31/32 review -- the stair-base approach-squeeze patient-gap dip fix, CLAUDE.md incident 8.15
continuation).

The caller-side TRIGGER (core/control/stair_policy.base_approach_park_request) and its
core/main.py call site are pure-Python/source-scan tested in test_stair_speed_guards.py. This
file covers the WIRE CROSSING + sim-side receiving end, which live in two modules that cannot
both be safely unit-run on a plain host:

  * sim/bot/sim_robot_controller.py -- host-importable (stdlib + a local logger module only),
    but instantiating SimRobotController opens a real JSONL log file under <repo>/log/ as a
    side effect of __init__ (configure_sim_logger); this file avoids that by SOURCE-SCANNING
    instead of importing, mirroring how isaac_env.py itself (which genuinely cannot be
    imported without isaacsim/omniverse, incident 8.4) is already handled everywhere else in
    this suite (test_stair_speed_guards.py's module docstring; `python -m compileall` covers
    syntax).
  * sim/isaac/isaac_env.py -- imports Isaac/omniverse, cannot be imported on a plain host at
    all (incident 8.4).

Both are pure regex/string source scans, same technique as test_stair_speed_guards.py's
`_call_site_lines` helper.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SIM_ROBOT_CONTROLLER_PY = os.path.join(REPO, "sim", "bot", "sim_robot_controller.py")
_ISAAC_ENV_PY = os.path.join(REPO, "sim", "isaac", "isaac_env.py")
_HOLD_PARK_PY = os.path.join(REPO, "go2_locomotion", "hold_park.py")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------------------
# sim/bot/sim_robot_controller.py -- the encode side.
# --------------------------------------------------------------------------------------

def test_sim_robot_controller_move_accepts_park_request():
    src = _read(_SIM_ROBOT_CONTROLLER_PY)
    assert "park_request: bool = False" in src, (
        "move()/_send() must accept a park_request bool kwarg (default False, backward "
        "compatible with older callers, mirroring the hold/person_detected plain-bool "
        "convention -- park_request needs no None-sentinel since 'not sent' and 'explicitly "
        "False' mean the same thing)"
    )


def test_sim_robot_controller_move_forwards_park_request_to_send():
    src = _read(_SIM_ROBOT_CONTROLLER_PY)
    def_idx = src.find("def move(")
    send_call_idx = src.find("self._send(", def_idx)
    assert def_idx != -1 and send_call_idx != -1
    window = src[send_call_idx: send_call_idx + 300]
    assert "park_request" in window, "move() must forward park_request into self._send(...)"


def test_sim_robot_controller_payload_encodes_park_request():
    src = _read(_SIM_ROBOT_CONTROLLER_PY)
    assert '"park_request": bool(park_request)' in src, (
        "the JSON payload dict must encode park_request as a plain bool"
    )


def test_sim_robot_controller_stop_sends_park_request_false():
    """stop() is a full idle/emergency stop, not a caller park intent -- its hardcoded _send()
    call must pass an explicit park_request=False positional value (matching the pre-existing
    hold=True/gap_brake_scale=None/yaw_align_rate=None hardcoded stop() payload)."""
    src = _read(_SIM_ROBOT_CONTROLLER_PY)
    stop_idx = src.find("def stop(self)")
    assert stop_idx != -1
    window = src[stop_idx: stop_idx + 200]
    assert "self._send(0.0, 0.0, 0.0, False, 0.0, None, False, True, False, None, None, None, False)" in window


# --------------------------------------------------------------------------------------
# sim/isaac/isaac_env.py -- the decode side + PARK engage wiring.
# --------------------------------------------------------------------------------------

def test_isaac_env_cmd_receiver_decodes_park_request():
    src = _read(_ISAAC_ENV_PY)
    assert 'payload.get("park_request", False)' in src, (
        "_cmd_receiver_thread must decode park_request from the UDP payload, defaulting "
        "False for an absent/older-sender field"
    )
    assert '_cmd_vel["park_request"] = park_request' in src, (
        "the decoded park_request must be stored into _cmd_vel under the cmd lock, mirroring "
        "gap_brake_scale/yaw_align_rate"
    )


def test_isaac_env_main_loop_reads_park_request_on_fresh_command():
    src = _read(_ISAAC_ENV_PY)
    assert '_cmd_vel.get("park_request", False)' in src, (
        "the main loop must read park_request back out of _cmd_vel on a fresh command"
    )


def test_isaac_env_main_loop_fails_safe_on_stale_command():
    """A dead UDP link (age > CMD_TIMEOUT_SEC) must fail toward NOT requesting the immediate
    path (CLAUDE.md 8.8), mirroring yaw_align_rate's fail-safe default in the same branch."""
    src = _read(_ISAAC_ENV_PY)
    stale_idx = src.find("if age > CMD_TIMEOUT_SEC:")
    else_idx = src.find("\n                else:", stale_idx)
    assert stale_idx != -1 and else_idx != -1 and stale_idx < else_idx
    stale_block = src[stale_idx:else_idx]
    assert "park_request = False" in stale_block, (
        "the stale-command branch must explicitly set park_request = False"
    )


def test_isaac_env_step_go2_locomotion_accepts_park_request():
    src = _read(_ISAAC_ENV_PY)
    def_idx = src.find("def _step_go2_locomotion(")
    assert def_idx != -1
    sig_end = src.find(") -> None:", def_idx)
    sig = src[def_idx:sig_end]
    assert "park_request: bool = False" in sig, (
        "_step_go2_locomotion must accept a park_request kwarg (default False)"
    )


def test_isaac_env_step_go2_locomotion_call_sites_forward_park_request():
    """Both live call sites of _step_go2_locomotion (the fresh-nonzero-command branch and the
    'momentarily no fresh command' hold branch -- the latter is the one the stair-base
    approach-squeeze actually reaches, since a hold with an effectively-zero commanded vx/wz
    routes there, not the former) must forward park_request. The two early-setup /
    self-test-only call sites (constant zero commands during stand-up) are out of scope."""
    src = _read(_ISAAC_ENV_PY)
    hits = []
    start = 0
    while True:
        idx = src.find("_step_go2_locomotion(go2, rl_policy,", start)
        if idx == -1:
            break
        hits.append(idx)
        start = idx + 1
    assert len(hits) >= 2, f"expected at least 2 non-trivial call sites, found {len(hits)}"
    forwarding = []
    for idx in hits:
        # A call site spans until the matching close-paren; grab a generous window (these
        # calls are multi-line but never exceed ~500 chars in this file).
        window = src[idx: idx + 600]
        close = window.find(")\n")
        window = window[:close + 1] if close != -1 else window
        if "climb_vx_brake_scale=" in window or "yaw_align_rate=" in window or "hold=True" in window:
            forwarding.append((idx, "park_request=" in window))
    assert forwarding, "could not identify the live (non-early-setup) call sites"
    missing = [idx for idx, has_it in forwarding if not has_it]
    assert not missing, (
        f"_step_go2_locomotion call site(s) at byte offset(s) {missing} do not forward "
        "park_request -- a caller park request would be silently dropped there"
    )


def test_isaac_env_park_block_computes_gated_park_requested():
    src = _read(_ISAAC_ENV_PY)
    assert "_park_requested = (" in src, (
        "the PARK block must compute a gated _park_requested local (AND-ed with the same "
        "not-climbing / not-yaw-aligning / caller-hold conditions as hold_requested)"
    )
    gate_idx = src.find("_park_requested = (")
    window = src[gate_idx: gate_idx + 300]
    for term in ("park_request", "_climb_fsm_active", "_yaw_aligning", "_motion_hold_requested"):
        assert term in window, f"_park_requested gating is missing {term!r}"


def test_isaac_env_park_block_passes_park_requested_into_update():
    src = _read(_ISAAC_ENV_PY)
    update_idx = src.find("_PGTT_HOLD_PARK.update(")
    assert update_idx != -1
    window = src[update_idx: update_idx + 300]
    assert "park_requested=_park_requested" in window, (
        "_PGTT_HOLD_PARK.update(...) must be called with park_requested=_park_requested"
    )


def test_isaac_env_park_block_uses_distinct_immediate_log_event_name():
    src = _read(_ISAAC_ENV_PY)
    assert '"pgtt_hold_park_engaged_immediate"' in src, (
        "an immediate (requested) park engage must log a DISTINCT event name from the "
        "ordinary timed 'pgtt_hold_park_engaged' so a run is diagnosable (CLAUDE.md 8.8)"
    )
    assert "_hp_decision.engaged_immediate" in src


# --------------------------------------------------------------------------------------
# go2_locomotion/hold_park.py -- confirm release semantics are untouched (park_requested
# plays no role in the top-of-function release branch).
# --------------------------------------------------------------------------------------

def test_hold_park_release_branch_does_not_reference_park_requested():
    src = _read(_HOLD_PARK_PY)
    release_idx = src.find("if not bool(hold_requested):")
    walk_idx = src.find('if self.state == "walk":', release_idx)
    assert release_idx != -1 and walk_idx != -1 and release_idx < walk_idx
    release_block = src[release_idx:walk_idx]
    assert "park_requested" not in release_block, (
        "release semantics must stay governed solely by hold_requested -- park_requested "
        "must not appear in the release branch"
    )
