from __future__ import annotations

# Facade: rl_locomotion_policy was split into cohesive sibling modules
# (rl_locomotion_contract = pinned contract constants + config dataclass;
# rl_locomotion_runner = the RLLocomotionPolicy runner + get_dof_names). This
# module re-exports every previously top-level name so the historical import
# paths (go2_locomotion.rl_locomotion_policy.*) keep resolving unchanged. This is
# a pure structural move -- no behaviour change.

import hashlib  # noqa: F401
import logging  # noqa: F401
import math  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np  # noqa: F401

from go2_locomotion.rl_locomotion_contract import (  # noqa: F401
    GO2_CALF_LEN_M,
    GO2_THIGH_LEN_M,
    POLICY_ACTION_SCALE_BY_JOINT,
    POLICY_DEFAULT_BY_JOINT,
    POLICY_JOINT_ORDER,
    RLLocomotionPolicyConfig,
    SWING_CLEARANCE_THRESHOLD_M,
    log_event,
)
from go2_locomotion.rl_locomotion_runner import (  # noqa: F401
    RLLocomotionPolicy,
    get_dof_names,
)
