"""Run the Extreme-Parkour-Onboard Go2 perceptive policy on the physical robot.

Exposes ``step(articulation, cmd, dt, *, delta_yaw=...)`` + ``leg_command_summary()`` + ``diagnostics()``.
"""

from __future__ import annotations

# Facade: parkour_locomotion_policy was split into cohesive sibling modules
# (parkour_locomotion_contract = joint order / default pose / dims / config
# dataclass + max_body_tilt_rad; parkour_locomotion_runner = the
# ParkourLocomotionPolicy runner). This module re-exports every previously
# top-level name so the historical import paths
# (go2_locomotion.parkour_locomotion_policy.*) keep resolving unchanged. This is
# a pure structural move -- no behaviour change.

import logging  # noqa: F401
import math  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np  # noqa: F401
import torch  # noqa: F401

from go2_locomotion.go2_locomotion_utils import (  # noqa: F401
    PARKOUR_DEFAULT_POSE,
    add_sensor_noise,
    apply_joint_efforts,
    apply_obs_latency,
    classify_dof,
    leg_extension_m,
    log_event,
    pd_torque,
    quat_to_matrix,
    read_joint_limits,
    safe_joint_vector,
)
from go2_locomotion.parkour_depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone  # noqa: F401
from go2_locomotion.scripted_stair_gait import ScriptedStairGait  # noqa: F401
from go2_locomotion.closed_loop_stair_climber import ClosedLoopStairClimber  # noqa: F401

from go2_locomotion.parkour_locomotion_contract import (  # noqa: F401
    PARKOUR_DEFAULT_POS,
    PARKOUR_DELTA_YAW_CLAMP,
    PARKOUR_DEPTH_HW,
    PARKOUR_JOINT_ORDER,
    PARKOUR_N_DEPTH_LATENT,
    PARKOUR_N_HIST,
    PARKOUR_N_PROPRIO,
    PARKOUR_TORQUE_LIMITS,
    ParkourPolicyConfig,
    max_body_tilt_rad,
)
from go2_locomotion.parkour_locomotion_runner import ParkourLocomotionPolicy  # noqa: F401
