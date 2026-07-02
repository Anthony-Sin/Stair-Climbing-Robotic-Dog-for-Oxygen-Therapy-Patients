"""Extreme-Parkour-Onboard Go2 policy contract — re-export shim.

The constants + config dataclass were byte-identical between the real robot and the
Isaac sim. The single source of truth is now
``go2_locomotion/parkour_locomotion_contract.py`` — the shared locomotion-policy
package that both targets already import (repo root is on ``sys.path`` in both, via
``real/main.py``'s insert and the sim launcher). This shim keeps the real-side import
path ``from real_parkour_contract import ...`` (used by ``real_parkour_runner`` and the
``parkour_locomotion_policy`` facade) working unchanged.
"""
from go2_locomotion.parkour_locomotion_contract import (  # noqa: F401
    PARKOUR_DEFAULT_POSE,
    PARKOUR_JOINT_ORDER,
    PARKOUR_DEFAULT_POS,
    PARKOUR_TORQUE_LIMITS,
    PARKOUR_N_PROPRIO,
    PARKOUR_N_HIST,
    PARKOUR_N_DEPTH_LATENT,
    PARKOUR_DEPTH_HW,
    max_body_tilt_rad,
    PARKOUR_DELTA_YAW_CLAMP,
    ParkourPolicyConfig,
)
