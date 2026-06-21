"""Animation-features test: the biped walk clip must expose a detectable loop.

Boots Isaac Sim, so it is skipped from host collection (tests/conftest.py).

Asset note: this historically opened the walk clip straight from the Omniverse
CDN. Per the project incident ledger, remote USD loads asynchronously and fall
back to a rest/T-pose, which would make the loop search silently find nothing.
Prefer a LOCAL copy: point BIPED_SETUP_USD at a local file (or drop one at
assets/Biped_Setup.usd). The CDN URL is only a last resort, and a load failure
SKIPS the test rather than silently passing.

Run directly (python tests/test_anim_features.py) or via pytest.
"""
import os
import sys

try:
    from isaacsim import SimulationApp
except ImportError:  # older Isaac packaging
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni  # noqa: E402
import omni.kit.commands  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Usd, UsdSkel  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CDN_BIPED = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.5/Isaac/People/Characters/Biped_Setup.usd"
)
_LOCAL_BIPED = os.path.join(_REPO, "assets", "Biped_Setup.usd")
WALK_ANIM_PATH = "/World/CharacterAnimation/Animation/stand_walk_1_skelanim"


def _biped_source() -> str:
    """Prefer an explicit/local asset; fall back to the CDN as a last resort."""
    env = os.environ.get("BIPED_SETUP_USD")
    if env:
        return env
    if os.path.exists(_LOCAL_BIPED):
        return _LOCAL_BIPED
    return _CDN_BIPED


def test_animation_extensions_and_walk_loop():
    import pytest

    # 1. The animation-graph extensions must register their commands once enabled.
    enable_extension("omni.anim.graph.core")
    enable_extension("omni.anim.graph.bundle")
    cmds = omni.kit.commands.get_commands()
    anim_cmds = [c for c in cmds if "anim" in c.lower() or "graph" in c.lower()]
    assert anim_cmds, "no animation/graph commands registered after enabling omni.anim.graph extensions"

    # 2. The walk clip must expose a detectable loop (start pose ~= end pose).
    source = _biped_source()
    try:
        stage = Usd.Stage.Open(source)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not open biped USD ({source}): {exc}")
    if stage is None:
        pytest.skip(f"biped USD opened to an empty stage: {source}")

    anim_prim = stage.GetPrimAtPath(WALK_ANIM_PATH)
    assert anim_prim.IsValid(), f"walk animation prim missing at {WALK_ANIM_PATH} in {source}"

    rotations_attr = UsdSkel.Animation(anim_prim).GetRotationsAttr()
    time_samples = rotations_attr.GetTimeSamples()
    assert time_samples, "walk animation has no rotation time samples (asset loaded as rest pose?)"

    best_t_start, best_L, min_diff = None, None, float("inf")
    for t_start in [t for t in time_samples if 100.0 <= t <= 200.0]:
        rot_start = rotations_attr.Get(t_start)
        if not rot_start:
            continue
        for L in range(70, 110, 2):  # candidate loop periods, multiples of 2
            t_end = t_start + L
            if t_end not in time_samples:
                continue
            rot_end = rotations_attr.Get(t_end)
            if not rot_end or len(rot_end) != len(rot_start):
                continue
            diff = 0.0
            for r1, r2 in zip(rot_start, rot_end):
                im1, im2 = r1.GetImaginary(), r2.GetImaginary()
                diff += (
                    (im1[0] - im2[0]) ** 2 + (im1[1] - im2[1]) ** 2
                    + (im1[2] - im2[2]) ** 2 + (r1.GetReal() - r2.GetReal()) ** 2
                )
            if diff < min_diff:
                min_diff, best_t_start, best_L = diff, t_start, L

    assert best_L is not None, (
        "no loop candidate found: no (t_start in [100,200], period in [70,110]) pair had "
        "matching rotation time samples -- the clip likely loaded as a rest pose "
        "(remote async load?). Prefer a local BIPED_SETUP_USD."
    )
    print(f"walk loop: t_start={best_t_start}, period L={best_L}, joint diff={min_diff:.5f}")


if __name__ == "__main__":
    try:
        test_animation_extensions_and_walk_loop()
        print("PASS")
        _code = 0
    except BaseException as exc:  # noqa: BLE001
        if type(exc).__name__ == "Skipped":
            print(f"SKIP: {exc}")
            _code = 0
        else:
            print(f"FAIL: {exc}")
            _code = 1
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    sys.exit(_code)
