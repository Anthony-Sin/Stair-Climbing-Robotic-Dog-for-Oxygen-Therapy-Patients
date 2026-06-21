"""Host-side tests for sim/bot/sim_logging_utils.py.

Covers the three logging guarantees a cold reader most needs to trust:
  * recordings are opt-in (SHOW_RECORDINGS) and hidden by default,
  * the scene baseline records the environment with NO person present,
  * every line is stamped with its run_id and each run starts from an empty file
    (so current-run output can never blend with a prior run's).

No Isaac / omni needed -- the module is pure stdlib logging.
Run directly (python tests/test_sim_logging.py) or via pytest.
"""
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BOT = os.path.join(_REPO, "sim", "bot")
if _BOT not in sys.path:
    sys.path.insert(0, _BOT)

import sim_logging_utils as slu  # noqa: E402


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _logger(run_folder_name):
    debug_dir = Path(tempfile.mkdtemp()) / run_folder_name / "debug"
    debug_dir.mkdir(parents=True)
    return slu.configure_sim_logger("isaac_env", log_dir=str(debug_dir), console=False)


# --------------------------------------------------------------------------
# recordings_visible -- the SHOW_RECORDINGS opt-in gate
# --------------------------------------------------------------------------

def test_recordings_hidden_by_default():
    os.environ.pop("SHOW_RECORDINGS", None)
    assert slu.recordings_visible() is False, "recordings must NOT be surfaced unless explicitly enabled"


def test_recordings_opt_in_truthy_and_falsy():
    try:
        for v in ("1", "true", "TRUE", "yes", "on"):
            os.environ["SHOW_RECORDINGS"] = v
            assert slu.recordings_visible() is True, f"{v!r} should enable recordings"
        for v in ("0", "false", "no", "off", ""):
            os.environ["SHOW_RECORDINGS"] = v
            assert slu.recordings_visible() is False, f"{v!r} should keep recordings hidden"
    finally:
        os.environ.pop("SHOW_RECORDINGS", None)


# --------------------------------------------------------------------------
# run_id derivation + per-line stamping + per-run reset
# --------------------------------------------------------------------------

def test_run_id_derivation():
    assert slu._derive_run_id(Path("/x/log/run_sim_20260621_010101_123/debug")) == "run_sim_20260621_010101_123"
    assert slu._derive_run_id(Path("/x/log/warm_isaac/debug")) == "warm_isaac"
    assert slu._derive_run_id(Path("/x/log/scratch")) is None, "ad-hoc log dirs have no run_id"


def test_every_line_stamped_with_run_id():
    lg = _logger("run_sim_20260621_333333_000")
    slu.log_scene_baseline(lg, terrain="t")
    slu.log_event(lg, logging.INFO, "demo", "hi", foo=1)
    rows = _read(lg.sim_log_path)
    assert rows, "logger produced no lines"
    assert all(r["labels"].get("run_id") == "run_sim_20260621_333333_000" for r in rows), \
        "every line must carry its run_id so current/prior runs never blend in output"


def test_reset_truncates_prior_run():
    debug_dir = Path(tempfile.mkdtemp()) / "run_sim_20260621_444444_000" / "debug"
    debug_dir.mkdir(parents=True)
    lg1 = slu.configure_sim_logger("isaac_env", log_dir=str(debug_dir), console=False)
    slu.log_event(lg1, logging.INFO, "old", "prior run line")
    lg2 = slu.configure_sim_logger("isaac_env", log_dir=str(debug_dir), console=False)  # reset=True default
    slu.log_event(lg2, logging.INFO, "new", "current run line")
    rows = _read(lg2.sim_log_path)
    assert len(rows) == 1 and rows[0]["event"]["action"] == "new", \
        "reset=True must start each run from an empty file (no prior-run lines survive)"


# --------------------------------------------------------------------------
# log_scene_baseline -- the empty-scene reference
# --------------------------------------------------------------------------

def test_scene_baseline_forces_person_absent():
    lg = _logger("run_sim_20260621_222222_000")
    # Caller passes person_present=True on purpose -- the baseline must override it.
    slu.log_scene_baseline(lg, terrain="commercial", step_height_m=0.15, person_present=True)
    events = [e for e in _read(lg.sim_log_path) if e["event"]["action"] == "scene_baseline"]
    assert len(events) == 1, "exactly one scene_baseline event expected"
    sim = events[0]["sim"]
    assert sim["person_present"] is False, "the baseline must record the scene with NO person"
    assert sim["terrain"] == "commercial"
    assert sim["step_height_m"] == 0.15


def test_scene_baseline_tolerates_missing_logger():
    # Must be a safe no-op when logging is disabled (logger is None).
    slu.log_scene_baseline(None, terrain="x")


# --------------------------------------------------------------------------
# Standalone runner (mirrors the other dual-mode tests in this suite)
# --------------------------------------------------------------------------

def _run_all() -> int:
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
