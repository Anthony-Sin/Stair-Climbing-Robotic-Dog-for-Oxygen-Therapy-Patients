"""Maps a recorder's ``dof_names`` (native articulation order) to URDF joint names.

CRITICAL (per the asset pipeline contract): articulation ``dof_names`` are in
joint-TYPE-major order (all 4 hips, then all 4 thighs, then all 4 calves) -- this is
Isaac/PhysX's native articulation DOF ordering, which groups by joint role across the
whole body, NOT the URDF's per-leg document order (FL hip/thigh/calf, then FR, ...) and
NOT the RL policy's own action-vector order (yet another convention). The three orders
share the same 12 logical joints but permute them differently, so mapping MUST go by
name (substring classification), never by raw index equality between arrays.

Each ``dof_names`` entry is classified case-insensitively by:
  * leg prefix: one of "FR", "FL", "RR", "RL" (front/rear x right/left)
  * joint role: one of "hip", "thigh", "calf"
and resolved to the URDF joint literally named ``f"{LEG}_{role}_joint"`` (go2.urdf's
exact naming), which is guaranteed to exist for all 12 combinations (verified below).
"""
from __future__ import annotations

import re
from typing import Dict, List

from urdf_parser import UrdfModel

_LEG_PATTERNS = [
    ("FR", re.compile(r"fr", re.IGNORECASE)),
    ("FL", re.compile(r"fl", re.IGNORECASE)),
    ("RR", re.compile(r"rr", re.IGNORECASE)),
    ("RL", re.compile(r"rl", re.IGNORECASE)),
]
_ROLE_PATTERNS = [
    ("hip", re.compile(r"hip", re.IGNORECASE)),
    ("thigh", re.compile(r"thigh", re.IGNORECASE)),
    ("calf", re.compile(r"calf", re.IGNORECASE)),
]


class DofMappingError(ValueError):
    pass


def classify_dof_name(name: str) -> tuple[str, str]:
    """Return (LEG, role) for a single dof name, e.g. 'FL_hip_joint' -> ('FL', 'hip'),
    or a policy-style name like 'go2_front_left_hip' would need its own aliasing (not
    handled here -- go2.urdf / the Isaac articulation both use FR/FL/RR/RL substrings,
    which is what real recorder dof_names are expected to contain per the data contract).
    Raises DofMappingError if leg or role cannot be determined, or is ambiguous.
    """
    leg_matches = [leg for leg, pat in _LEG_PATTERNS if pat.search(name)]
    role_matches = [role for role, pat in _ROLE_PATTERNS if pat.search(name)]

    if len(leg_matches) != 1:
        raise DofMappingError(
            f"dof name {name!r}: expected exactly one leg-prefix match among "
            f"FR/FL/RR/RL, found {leg_matches!r}"
        )
    if len(role_matches) != 1:
        raise DofMappingError(
            f"dof name {name!r}: expected exactly one joint-role match among "
            f"hip/thigh/calf, found {role_matches!r}"
        )
    return leg_matches[0], role_matches[0]


def build_dof_to_urdf_joint(dof_names: List[str], urdf: UrdfModel) -> Dict[int, str]:
    """dof_names[i] (native articulation order) -> URDF joint name. Returns a dict
    keyed by the dof INDEX (0..len(dof_names)-1) so callers can look up "which URDF
    joint does per-frame dof_pos[i] drive" without assuming any particular order.
    """
    revolute_names = {j.name for j in urdf.revolute_joints()}
    mapping: Dict[int, str] = {}
    seen_joints: Dict[str, int] = {}
    for i, name in enumerate(dof_names):
        leg, role = classify_dof_name(name)
        urdf_joint_name = f"{leg}_{role}_joint"
        if urdf_joint_name not in revolute_names:
            raise DofMappingError(
                f"dof name {name!r} classified as leg={leg} role={role} -> "
                f"expected URDF joint {urdf_joint_name!r}, but it is not a revolute "
                f"joint in the URDF (available: {sorted(revolute_names)})"
            )
        if urdf_joint_name in seen_joints:
            raise DofMappingError(
                f"dof name {name!r} (index {i}) maps to URDF joint {urdf_joint_name!r}, "
                f"which was already claimed by dof_names[{seen_joints[urdf_joint_name]}] "
                f"({dof_names[seen_joints[urdf_joint_name]]!r}) -- duplicate/ambiguous mapping"
            )
        seen_joints[urdf_joint_name] = i
        mapping[i] = urdf_joint_name

    missing = revolute_names - set(mapping.values())
    if missing:
        raise DofMappingError(
            f"dof_names did not cover all 12 URDF revolute joints; missing: {sorted(missing)}"
        )
    return mapping


# Canonical joint-type-major order matching Isaac/PhysX articulation convention
# (all hips, then all thighs, then all calves), used ONLY by synthetic mode to
# fabricate a plausible dof_names header -- never assumed for real recorder data,
# which is always mapped by name via build_dof_to_urdf_joint() above.
SYNTHETIC_DOF_NAMES: List[str] = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]


if __name__ == "__main__":
    from pathlib import Path
    from urdf_parser import parse_urdf

    urdf_path = Path(__file__).resolve().parents[3] / "sim" / "isaac" / "assets" / "go2.urdf"
    model = parse_urdf(urdf_path)

    # Self-test with the synthetic (joint-type-major) order.
    mapping = build_dof_to_urdf_joint(SYNTHETIC_DOF_NAMES, model)
    print("dof_index -> urdf_joint (joint-type-major synthetic order):")
    for i, name in enumerate(SYNTHETIC_DOF_NAMES):
        print(f"  [{i:2d}] {name:16s} -> {mapping[i]}")

    # Self-test with URDF document order (FL hip/thigh/calf, FR, RL, RR) to prove the
    # mapping is order-independent (name-driven, not index-driven).
    doc_order = [j.name for j in model.revolute_joints()]
    mapping2 = build_dof_to_urdf_joint(doc_order, model)
    assert all(mapping2[i] == doc_order[i] for i in range(12)), "doc-order self-map failed"
    print("\ndoc-order self-map OK (name-driven mapping is order-independent)")

    # Negative test: ambiguous/ bad name should raise.
    try:
        build_dof_to_urdf_joint(["FL_hip_joint", "FL_hip_joint"] + SYNTHETIC_DOF_NAMES[2:], model)
        print("FAIL: expected DofMappingError for duplicate mapping")
    except DofMappingError as e:
        print(f"\nnegative test OK (duplicate correctly rejected): {e}")
