"""Pytest configuration for the tests/ suite.

Most tests here are pure-Python and run on any dev machine. A few boot Isaac Sim
or load GPU model weights *at import time*; on a host without those runtimes the
import raises and would fail collection for the WHOLE run. We skip collecting
those files cleanly instead, so `pytest tests/` stays green on a plain checkout
and only runs what the machine can actually support.

See tests/README.md for the host-vs-sim split and how to run the heavy tests.
"""
import importlib.util


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _missing(*modules: str) -> bool:
    return not all(_have(m) for m in modules)


# Test file -> the runtime modules it imports at module load. When any are
# absent the file cannot run here, so it is skipped from collection rather than
# raising an ImportError that would abort the entire session.
_ENV_DEPENDENT = {
    "test_anim_features.py": ("isaacsim",),            # boots Isaac Sim + fetches remote USD
    "test_usd_assets.py": ("isaacsim", "pxr"),         # boots Isaac Sim + inspects USD stages
    "test_robot_simulation.py": ("isaacsim",),         # boots Isaac Sim + renders a frame
    "test_detection_on_sim_textured.py": ("ultralytics",),  # loads a YOLO-World GPU model
}

collect_ignore = [name for name, mods in _ENV_DEPENDENT.items() if _missing(*mods)]
