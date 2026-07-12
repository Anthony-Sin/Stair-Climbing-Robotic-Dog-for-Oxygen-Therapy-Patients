"""Dual-policy walk<->climb handoff for the PGTT walker.

WHY: PGTT is an excellent flat-ground walker but has NO stair-climb path (its
``step`` never receives the scripted-climb signal). The deterministic
``ClosedLoopStairClimber`` (lifted from commit 7ddb1f7) CAN attempt a step-up and
fails safe upright. This module is the glue that, while PGTT walks, watches for the
walker to STALL in front of a staircase and, when it does, hands the leg targets to
the climber for ONE riser, then hands back to PGTT -- exactly the Task-2 contract:

    if walking_policy_is_stalled()          # StallDetector (2a)
       and stair_detector.stair_detected    # DepthStairDetector (2b)
       and stair_detector.stair_count >= 2:
           switch_to -> ClosedLoopStairClimber (2c)
           climb one stair
           switch_back -> PGTT

Nothing here is hardcoded into the control loop: every threshold lives in
``HandoffConfig`` so it is tunable from the isaac_env argparse / run_sim launcher.
The detector runs on the body-mounted parkour depth camera (the robot's real depth
sensor feed), NOT a flat ground-truth heightmap, per the Task-2b requirement.

The climber emits 12 joint targets in PGTT ACT order (FR/FL/RR/RL x hip/thigh/calf)
-- identical to PgttLocomotionPolicy.ACT_ORDER -- so they apply through the walker's
existing name-based joint map and the same Kp40 position drive (see
PgttLocomotionPolicy.apply_external_act_targets).

Facade: the implementation now lives in single-responsibility sibling modules --
``handoff_config`` (the ``HandoffConfig`` tunables + ``GO2_LEG_CLEARANCE_M``),
``handoff_detectors`` (``StallDetector`` 2a + ``DepthStairDetector`` 2b), and
``handoff_controller`` (the ``HandoffController`` 2c state machine). They are
re-exported here so ``from go2_locomotion.pgtt_stair_handoff import ...`` keeps
working unchanged for the sim and the real ROS2 port.
"""

from __future__ import annotations

from go2_locomotion.handoff_config import (  # noqa: F401
    GO2_LEG_CLEARANCE_M,
    HandoffConfig,
)
from go2_locomotion.handoff_controller import (  # noqa: F401
    HandoffController,
    log_event,
    stair_engage_person_ghost_veto,
    stair_entry_lead_ok,
)
from go2_locomotion.handoff_detectors import (  # noqa: F401
    DepthStairDetector,
    StallDetector,
)
