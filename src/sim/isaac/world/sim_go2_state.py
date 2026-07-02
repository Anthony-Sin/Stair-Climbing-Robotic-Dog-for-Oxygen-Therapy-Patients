"""Go2 telemetry/bookkeeping state struct and its rate-limited logging helper.

Split out of ``sim_go2_locomotion`` (the facade re-exports these). Holds the
per-run perception/telemetry fields and logging latches shared with the
stair-demo reporter; the locomotion policy (parkour_locomotion_policy.py) drives
the joints -- this struct only carries observation/telemetry state.

Locomotion is driven by the parkour depth/vision policy in
parkour_locomotion_policy.py. This module now only owns the stair-demo
perception/telemetry and the analytical terrain model used to build that
telemetry from the robot's measured body pose.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sim_logging_utils import log_event


@dataclass
class Go2LocomotionState:
    # Telemetry/bookkeeping state shared with the stair-demo reporter. The
    # locomotion policy (parkour_locomotion_policy.py) drives the joints; this
    # struct only carries perception/telemetry fields and per-run logging latches.
    target_height_m: float = 0.30
    stand_joint_positions: Optional[np.ndarray] = None
    dof_names: List[str] = field(default_factory=list)
    rigid_body_path: str = ""
    rigid_body_logged: bool = False
    gait_logged: bool = False
    stair_hold_logged: bool = False
    warning_times: Dict[str, float] = field(default_factory=dict)
    stable_hold_logged: bool = False

    # Front-camera handheld-shake clock. set_front_camera_local_pose() uses these
    # to add subtle walking motion to the robot-POV camera; gait_period sets the
    # shake rate. (These are NOT a locomotion gait -- the RL policy moves the legs.)
    gait_time: float = 0.0
    gait_period: float = 0.6

    # Tracking states
    joint_gains_set: bool = False
    joint_gains_unavailable: bool = False
    dof_map: Dict[Tuple[str, str], int] = field(default_factory=dict)
    stair_demo_telemetry: Dict[str, Any] = field(default_factory=dict)
    # Per-leg command summary from the locomotion policy
    # (parkour_locomotion_policy.ParkourLocomotionPolicy.leg_command_summary()):
    #   {"swing_legs": [...], "leg_commands": {LEG: {...}}}.
    # This is the single source of truth for the leg/gait telemetry and HUD,
    # replacing the removed procedural-gait swing bookkeeping.
    leg_summary: Dict[str, Any] = field(default_factory=dict)
    policy_name: str = ""
    # Dual-policy stair-handoff telemetry (pgtt_stair_handoff.HandoffController.
    # telemetry()), surfaced into fall_diag so the WALK<->CLIMB switch is verifiable
    # from the authoritative motion log. None when the handoff is inactive/disabled.
    handoff: Any = None
    stair_demo_climb_logged: bool = False
    stair_demo_complete_logged: bool = False
    stair_crawl_logged: bool = False
    stair_visual_deferred_logged: bool = False
    fallback_body_motion_disabled_logged: bool = False

    # Diagnostics: most-recent measured base velocity (m/s). diag_body_* is the
    # measured velocity rotated into the robot's heading frame, so diag_body_vx > 0
    # means the body is actually translating forward. Compared against the
    # commanded vx this reveals whether the gait produces forward thrust or the
    # body is recoiling/slipping backward.
    diag_body_vx: float = 0.0
    diag_body_vy: float = 0.0
    diag_cmd_vx: float = 0.0


def _warn_rate_limited(
    logger: Optional[logging.Logger],
    state: Go2LocomotionState,
    key: str,
    message: str,
    *,
    interval_sec: float = 2.0,
    **fields: Any,
) -> None:
    if logger is None:
        return
    now = time.monotonic()
    last = state.warning_times.get(key, 0.0)
    if now - last < interval_sec:
        return
    state.warning_times[key] = now
    log_event(logger, logging.WARNING, key, message, **fields)
