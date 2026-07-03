"""Cross-process configuration contract (single source of truth).

Several independently-launched processes (the sim `isaac_env`, the real ROS 2
sidecars, and the core controller in ``src/core``) each parse their OWN CLI
args, yet a handful of knobs MUST agree numerically across all of them or the
follow/stair behaviour silently diverges (e.g. the controller thinks the
standoff is 0.6 m while the sim spawns the patient assuming 1.5 m).

This module is the ONE place those cross-process defaults live. Each owner
imports the canonical value from here (or, where a hard import is awkward,
keeps its parser default numerically equal and cites this file in a comment)
so a change lands in every process at once.

Do NOT put per-process-only tunables here -- only values that two or more
processes must share. Keep the dict flat and documented.
"""
from typing import Any, Dict

# Canonical cross-process defaults. Every entry is consumed by >= 2 processes.
CONTRACT: Dict[str, Any] = {
    # Follow standoff (m): the gap the dog holds behind the patient on flat ground.
    # The sim spawns/animates the patient assuming this gap; the controller's
    # PersonFollowingConfig.target_distance must match or the go/hold band is wrong.
    # Raised from the old tight 0.6 m: the body-mounted D435 sits low, so at close range the patient
    # is only-LEGS in frame and the pose detector loses the lock (verified in run_sim_20260703_150109's
    # loss frame). A wider standoff keeps the patient's body inside the RGB/YOLO cone. Both sim and
    # real now use 1.0 m (sim via run_sim -TargetDistance; real via the args_parser --target-distance
    # default that run_real.sh inherits).
    "follow_standoff_sim": 1.0,     # was 0.6 -- raised to keep the patient framed for detection
    "follow_standoff_real": 1.0,    # was 1.5 -- real robot standoff (args_parser --target-distance)

    # Stair follow bearing scale: gain applied to the person-bearing heading term
    # injected in hybrid mode while on the stairs (sim + real parkour policy).
    "stair_bearing_scale": 0.9,

    # Default stair scene preset used by the demo/HUD overlay and the sim scene
    # loader; both must agree on which staircase geometry is in play.
    "stair_preset_demo": "commercial",

    # Standoff hysteresis "start" band offset (m) relative to the standoff target.
    # Widened from 0.15 -> 0.35 so a floor-speed burst overshoots the slow leader
    # drift and settles inside the band instead of immediately re-triggering GO
    # (the frozen policy cannot burst gently). Shared by the controller shaping and
    # any sim/analysis tool that reconstructs the go/hold decision.
    "standoff_band_out": 0.35,

    # UDP command port the controller publishes follow targets on and the ROS 2 /
    # sim sidecar subscribes to. Both ends must bind the same port.
    "cmd_port": 52100,
}


def contract_value(key: str) -> Any:
    """Return the canonical cross-process default for ``key``.

    Raises ``KeyError`` (fail loud) rather than returning a silent default so a
    typo in a consumer surfaces immediately instead of drifting the two ends
    apart with a wrong value.
    """
    return CONTRACT[key]
