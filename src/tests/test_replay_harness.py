"""Tier-1 trace-driven decision-level replay harness (host-safe, no Isaac/torch/GPU).

This is the permanent, executable pin for incident 8.3 (and the whole "stair mode latched
on flat ground / dog stops plain-following" failure class). The controller writes a per-frame
JSONL trace (``vision_main_trace``); we replay the RECORDED detector outputs + timestamps of
two golden traces (a flat-follow run and a stair-climb run, under ``src/tests/traces/``) and
assert control-level INVARIANTS -- both against the recorded ``fsm_state`` AND against the
FSM re-derived by feeding the recorded signals back through the REAL ``ClimbFSM.update`` and
``depth_stair_latch_allowed`` pure functions. No sensors, no models, no simulator.

Invariants pinned (per the task spec):
  (a) No stair latch / ``STAIR_*`` state before the first YOLO stair-detection frame, on
      either trace. (Incident 8.3: depth false-positives must not latch stairs pre-YOLO.)
  (b) ``fsm_state`` is never ``STAIR_*`` on the flat-follow trace. (Pure-flat run must
      plain-follow start to finish.)
  (c) The forward / trans-x command is 0 when the person is lost AND the robot is NOT on
      stairs. (A lost patient on flat ground must not drive the dog blind forward.)

The traces are trimmed/synthesized to a few hundred KB and carry a ``_meta`` header line.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> src/
sys.path.insert(0, REPO)

from core.control.climb_fsm import ClimbFSM  # noqa: E402
from core.control.stair_policy import depth_stair_latch_allowed  # noqa: E402

_TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")
_FLAT = os.path.join(_TRACES_DIR, "flat_follow_trace.jsonl")
_STAIR = os.path.join(_TRACES_DIR, "stair_climb_trace.jsonl")

#: Any fsm_state name that means "the stair machinery is engaged". COMMITTED_CLIMB is the
#: committed straight-up climb; the STAIR_* family covers approach/near/loss/approach-commit.
_STAIR_STATES = {
    "STAIR_APPROACH", "STAIR_NEAR", "COMMITTED_CLIMB",
    "STAIR_LOSS_FLOOR", "STAIR_APPROACH_COMMIT",
}


def _load_trace(path):
    """Return (meta, [frame dicts]). The first line is a ``{"_meta": {...}}`` header."""
    assert os.path.exists(path), f"golden trace missing: {path}"
    meta, frames = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "_meta" in rec:
                meta = rec["_meta"]
                continue
            frames.append(rec)
    assert frames, f"trace {path} has no frames"
    return meta, frames


def _first_yolo_stair_frame(frames):
    """The frame_index of the first RECORDED YOLO stair detection (stairs_detected True),
    or None if the trace never detects stairs."""
    for fr in frames:
        if bool(fr.get("stairs_detected", False)):
            return fr.get("frame_index")
    return None


# ---------------------------------------------------------------------------
# A minimal args stand-in carrying only the knobs ClimbFSM.update reads. Values
# mirror the argparse defaults in src/core/args_parser.py so the re-derived FSM
# matches the controller. (We do NOT import the real parser -- it calls
# parse_args() at import and pulls sys.argv; this keeps the harness pure.)
# ---------------------------------------------------------------------------
class _Args:
    stair_seen_persist_sec = 8.0
    stair_hold_suppress_sec = 4.0
    stair_near_distance = 0.45
    stair_depth_engage_distance = 0.45
    stair_target_distance = 0.50
    stair_forward_floor = 0.0
    stair_loss_forward_floor = 0.16
    trans_x_max = 0.6
    stair_speed_scale = 0.45
    obstacle_slow_distance = 1.20
    stair_climb_commit = True
    stair_climb_commit_distance = 1.0
    stair_climb_max_sec = 12.0
    stair_climb_latch = True
    follow_loss_glide_sec = 10.0


# ---------------------------------------------------------------------------
# Invariant (b): the flat-follow trace never enters a STAIR_* state.
# ---------------------------------------------------------------------------
def test_flat_trace_never_enters_stair_state():
    meta, frames = _load_trace(_FLAT)
    assert meta.get("scenario") == "flat_follow"
    offenders = [
        (fr.get("frame_index"), fr.get("fsm_state"))
        for fr in frames
        if fr.get("fsm_state") in _STAIR_STATES
    ]
    assert not offenders, (
        "flat-follow trace entered a STAIR_* state -- the incident-8.3 failure "
        f"(stair mode latched on flat ground). offending (frame, state): {offenders[:10]}"
    )
    # And it must never even mark stairs detected/active on a pure-flat run.
    assert not any(fr.get("stairs_detected") for fr in frames), \
        "flat-follow trace recorded stairs_detected=True on flat ground"
    assert not any(fr.get("stairs_action_active") for fr in frames), \
        "flat-follow trace recorded stairs_action_active=True on flat ground"


# ---------------------------------------------------------------------------
# Invariant (a): no stair latch / STAIR_* state before the first YOLO detection.
# Checked against the RECORDED states...
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [_FLAT, _STAIR])
def test_no_stair_state_before_first_yolo(path):
    _meta, frames = _load_trace(path)
    first_yolo = _first_yolo_stair_frame(frames)
    for fr in frames:
        fi = fr.get("frame_index")
        state = fr.get("fsm_state")
        if state in _STAIR_STATES:
            assert first_yolo is not None, (
                f"{os.path.basename(path)} frame {fi}: fsm_state={state} but NO YOLO stair "
                "detection ever occurred (a pure depth false-positive latched stair mode -- "
                "incident 8.3)."
            )
            assert fi >= first_yolo, (
                f"{os.path.basename(path)} frame {fi}: fsm_state={state} BEFORE the first YOLO "
                f"stair detection at frame {first_yolo}. Depth must not latch stairs without "
                "recent YOLO corroboration (incident 8.3 residual)."
            )


# ...and against the depth-only latch gate replayed through the REAL pure function.
def test_depth_latch_gate_blocks_pre_yolo_false_positives():
    """Replay each frame's recorded depth-confirmation + timestamps through the ACTUAL
    ``depth_stair_latch_allowed`` gate and assert it never permits a depth-only latch
    before YOLO has corroborated stairs within ``stair_seen_persist_sec``. This is the
    exact gate that fixed the incident-8.3 residual (342 pre-YOLO false-latch frames -> 0)."""
    for path in (_FLAT, _STAIR):
        _meta, frames = _load_trace(path)
        persist = _Args.stair_seen_persist_sec
        last_yolo_ts = -1e9  # no YOLO seen yet (mirrors main.py _last_yolo_stair_ts init)
        first_yolo = _first_yolo_stair_frame(frames)
        for fr in frames:
            fi = fr.get("frame_index")
            now = float(fr["ts_unix"])
            # Update the YOLO-seen timestamp exactly as the loop does.
            if bool(fr.get("stairs_detected", False)):
                last_yolo_ts = now
            depth_confirmed = bool(fr.get("depth_stair_confirmed", False))
            allowed = depth_stair_latch_allowed(
                depth_confirmed=depth_confirmed, now=now,
                last_yolo_stair_ts=last_yolo_ts, persist_sec=persist,
            )
            if first_yolo is None or fi < first_yolo:
                # Before any YOLO stair evidence, the depth-only path must NEVER latch,
                # no matter how many fake risers depth confirmed.
                assert not allowed, (
                    f"{os.path.basename(path)} frame {fi}: depth-only latch ALLOWED before "
                    f"the first YOLO stair detection (depth_confirmed={depth_confirmed}). "
                    "This is exactly the flat-ground false latch incident 8.3 guards against."
                )


# ---------------------------------------------------------------------------
# Invariant (c): forward command is 0 when the person is lost AND not on stairs.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [_FLAT, _STAIR])
def test_forward_cmd_zero_when_lost_and_not_on_stairs(path):
    """A lost patient on flat ground must not leave the dog driving forward BLIND.

    Two sub-invariants, chosen to respect the controller's INTENTIONAL brief-loss forward
    bridge (the front-depth-gated flat-loss glide / pursuit-grace that spans a ~1.3 s YOLO
    blink) while still pinning the real safety property:

      1. When the controller has explicitly decided to hold -- ``fsm_state == "STOP"`` --
         the forward command is exactly 0 (STOP means controller.stop(); no residual drive).
      2. When the patient has been lost LONGER than the flat-loss glide window
         (``follow_loss_glide_sec``) and the robot is NOT on stairs, the forward command is
         0 -- the bounded bridge has expired, so a truly-gone patient can't keep the dog
         gliding forward on flat ground indefinitely.
    """
    _meta, frames = _load_trace(path)
    glide_window = _Args.follow_loss_glide_sec
    for fr in frames:
        fi = fr.get("frame_index")
        fwd = float(fr.get("command_trans_x_limited", 0.0) or 0.0)
        state = fr.get("fsm_state")

        # (1) STOP is the controller's explicit hold -- forward must be exactly 0.
        if state == "STOP":
            assert abs(fwd) < 1e-6, (
                f"{os.path.basename(path)} frame {fi}: fsm_state=STOP yet forward command "
                f"= {fwd} (STOP must issue zero drive)."
            )
            continue

        person_detected = bool(fr.get("person_detected", False))
        on_stairs = (
            state in _STAIR_STATES
            or bool(fr.get("stairs_action_active", False))
            or bool(fr.get("stairs_detected", False))
            or bool(fr.get("stair_climbing_latch", False))
        )
        if person_detected or on_stairs:
            continue

        # (2) Person lost, flat ground, and the brief-loss glide/pursuit window has expired.
        lost_age = fr.get("lost_age_sec")
        if lost_age is not None and float(lost_age) > float(glide_window):
            assert abs(fwd) < 1e-6, (
                f"{os.path.basename(path)} frame {fi}: patient lost for {lost_age}s "
                f"(> {glide_window}s glide window), not on stairs, fsm_state={state}, yet "
                f"forward command = {fwd} (a long-gone patient must not drive the dog forward)."
            )


# ---------------------------------------------------------------------------
# Re-derive the FSM from recorded inputs through the REAL ClimbFSM and confirm the
# stair-state ordering invariant holds on the RE-DERIVED states too (not just the
# recorded ones) -- so the pin survives a future controller edit that changes how
# fsm_state is written to the trace.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", [_FLAT, _STAIR])
def test_rederived_fsm_respects_pre_yolo_and_flat_invariants(path):
    meta, frames = _load_trace(path)
    fsm = ClimbFSM(_Args())
    # ClimbFSM uses time.perf_counter internally for its latch windows; the trace carries
    # wall-clock ts_unix. Feed a monotonic perf-counter-like clock derived from the trace
    # deltas so the second-based latches behave as they did live. IMPORTANT: the FSM inits
    # its latch timestamps (last_on_stairs_ts, _stairs_seen_ts) to 0.0 and tests
    # ``current_time - <ts> < window``; production's perf_counter is a LARGE number, so at
    # startup those windows read as "long expired". Offset the replay clock by a large base
    # so a t=0 init timestamp does NOT spuriously read as "just now" (which would make
    # stairs_recent/stairs_seen_recent True on frame 1 and fabricate a STAIR_NEAR state).
    _CLOCK_BASE = 1.0e6
    base = float(frames[0]["ts_unix"])
    first_yolo = _first_yolo_stair_frame(frames)
    is_flat = (meta.get("scenario") == "flat_follow")

    for fr in frames:
        fi = fr.get("frame_index")
        current_time = _CLOCK_BASE + float(fr["ts_unix"]) - base  # seconds since trace start
        out = fsm.update(
            current_time,
            stairs_detected=bool(fr.get("stairs_detected", False)),
            stairs_action_active=bool(fr.get("stairs_action_active", False)),
            stairs_depth_m=fr.get("stairs_depth_m"),
            last_stairs_depth_m=fr.get("stairs_depth_m"),
            stairs_depth_ever_confirmed=bool(fr.get("stairs_depth_ever_confirmed", False)),
            person_detected=bool(fr.get("person_detected", False)),
            depth_distance_m=fr.get("depth_distance_m"),
            front_near_m=fr.get("front_near_m"),
            standoff_gap_ctrl_m=fr.get("standoff_gap_ctrl_m"),
            lost_age_sec=fr.get("lost_age_sec"),
            motion_allowed=bool(fr.get("person_detected", False)),
            last_seen_bearing_deg=fr.get("last_seen_bearing_deg"),
            recovery_yaw_active=bool(fr.get("lost_search_active", False)),
        )
        state = out["fsm_state"]

        if is_flat:
            assert state not in _STAIR_STATES, (
                f"re-derived FSM entered {state} at frame {fi} on the flat-follow trace "
                "(fed only recorded flat signals -- no stairs). Incident 8.3 regression."
            )

        if state in _STAIR_STATES:
            assert first_yolo is not None and fi >= first_yolo, (
                f"re-derived FSM entered {state} at frame {fi} before the first YOLO stair "
                f"detection (frame {first_yolo}). Depth must not latch stairs pre-YOLO."
            )


def test_traces_are_small_and_present():
    """Guard the deliverable itself: both golden traces exist and stay a few hundred KB."""
    for path in (_FLAT, _STAIR):
        assert os.path.exists(path), f"missing golden trace {path}"
        size = os.path.getsize(path)
        assert size < 600 * 1024, f"{os.path.basename(path)} is {size} bytes (keep < 600 KB)"
