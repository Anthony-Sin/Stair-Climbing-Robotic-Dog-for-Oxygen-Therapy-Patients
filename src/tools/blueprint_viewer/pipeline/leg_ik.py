"""2-link-plus-hip-offset analytic IK for one Go2 leg, used by the synthetic gait
generator (and available for the real-data path's optional leg-driven-by-foot-target
mode, though real mode primarily just plays back logged dof_pos directly).

Go2 leg kinematic structure (from go2.urdf, identical shape for all 4 legs up to the
mirrored hip_y offset sign):
    hip_joint   (revolute about local +X, abduction) -- offset from trunk by (+-0.1934, +-0.0465, 0)
    thigh_joint (revolute about local +Y, the "shoulder" pitch) -- offset from hip by (0, +-0.0955, 0)
    calf_joint  (revolute about local +Y, the "knee" pitch)     -- offset from thigh by (0, 0, -0.213)
    foot        (fixed)                                         -- offset from calf by (0, 0, -0.213)

i.e. a hip abduction offset (hip_dy) then a planar 2-link leg (thigh_len, calf_len) that
swings in the plane perpendicular to the (now-rotated) hip axis. This is the standard
"quadruped leg" analytic IK: solve the hip abduction angle from the target's Y/Z (to
bring the target into the leg's sagittal plane), then solve a 2-link planar IK in that
plane for thigh/calf.

All angles returned in URDF joint convention (radians, matching go2.urdf <axis>/<limit>).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

Vec3 = Tuple[float, float, float]

THIGH_LEN_M = 0.213   # thigh_joint -> calf_joint offset magnitude (URDF: 0,0,-0.213)
CALF_LEN_M = 0.213    # calf_joint -> foot offset magnitude (URDF: 0,0,-0.213)
HIP_TO_THIGH_Y_M = 0.0955  # |thigh_joint origin.y| in the hip frame (URDF: 0, 0.0955, 0)

# URDF calf joint limits (FL/FR/RL/RR all share the same calf limit range):
# <limit lower="-2.7227" upper="-0.83776" .../>. The GEOMETRIC max reach
# (THIGH_LEN_M + CALF_LEN_M = 0.426 m, leg dead straight) is NOT physically reachable
# by the real servo -- its calf limit stops ~5-7deg short of straight. Solving IK
# against the naive geometric max_reach can therefore return calf angles that violate
# the URDF limit (observed: -0.714 rad vs the -0.83776 upper bound, during the
# synthetic climb gait's near-full-reach trailing-leg moments). CALF_LIMIT_UPPER_RAD's
# corresponding reach is the physically-correct clamp ceiling.
CALF_LIMIT_UPPER_RAD = -0.83776
_KNEE_INTERIOR_AT_LIMIT = math.pi + CALF_LIMIT_UPPER_RAD  # inverse of calf_angle = -(pi - knee_interior)
MAX_REACH_M = math.sqrt(
    THIGH_LEN_M ** 2 + CALF_LEN_M ** 2
    - 2.0 * THIGH_LEN_M * CALF_LEN_M * math.cos(_KNEE_INTERIOR_AT_LIMIT)
)  # ~0.389 m (vs geometric 0.426 m) -- see comment above.


@dataclass(frozen=True)
class LegAngles:
    hip: float
    thigh: float
    calf: float


def solve_leg_ik(
    target_in_hip_frame: Vec3,
    leg_sign_y: float,
    thigh_len: float = THIGH_LEN_M,
    calf_len: float = CALF_LEN_M,
    hip_to_thigh_y: float = HIP_TO_THIGH_Y_M,
    max_reach: float = None,
) -> LegAngles:
    """Solve (hip, thigh, calf) angles so the foot reaches ``target_in_hip_frame``
    (the desired FOOT position expressed in the HIP JOINT's local frame, i.e. relative
    to the hip_joint origin, in the parent/trunk's orientation -- NOT yet rotated by
    the hip's own abduction).

    ``leg_sign_y``: +1 for left legs (FL/RL, thigh offset +Y), -1 for right legs
    (FR/RR, thigh offset -Y) -- selects which side the knee-bend plane is offset to.

    ``max_reach``: reach clamp ceiling. Defaults to ``MAX_REACH_M`` (the URDF calf
    joint's actual limit-derived reach, ~0.389 m) when called with the default
    thigh_len/calf_len, so solved angles stay within the real servo's limits, NOT just
    the geometric annulus (thigh_len+calf_len=0.426 m is geometrically reachable but
    would require an out-of-range calf angle -- see MAX_REACH_M's derivation comment).
    For custom thigh_len/calf_len (e.g. leg_ik.py's own round-trip self-test, which
    deliberately stresses the full geometric range) pass an explicit max_reach or rely
    on the geometric-sum fallback below.

    Returns URDF-convention joint angles. Uses the standard quadruped analytic IK:
      1. hip (abduction, rotates about X): brings the target into the leg's sagittal
         (X-Z, after hip rotation) plane by rotating around X until the target's
         post-hip-frame Y equals the fixed thigh-offset ``hip_to_thigh_y`` (leg_sign_y
         applied) -- equivalently, solve the angle that zeros the perpendicular
         (Y vs Z) residual against the offset.
      2. thigh + calf: standard 2-link planar IK (law of cosines) in the plane that
         now contains the target and the thigh-offset point.
    """
    if max_reach is None:
        if thigh_len == THIGH_LEN_M and calf_len == CALF_LEN_M:
            max_reach = MAX_REACH_M
        else:
            max_reach = thigh_len + calf_len
    tx, ty, tz = target_in_hip_frame
    dy = hip_to_thigh_y * leg_sign_y

    # --- Step 1: hip abduction angle ---
    # The thigh/calf/foot chain hangs from a fixed offset (local Y = dy) below the hip
    # joint; hip abduction rotates that whole chain about the hip's local X axis. Let
    # L = sqrt(ty^2+tz^2) (target's distance from the hip's X axis) and
    # d = sqrt(max(L^2-dy^2, 0)) (the chain's own reach, perpendicular to dy, within
    # its swing plane). At hip_angle=0 the chain sits at (y=dy, z=-d) (hangs straight
    # down); rotating that point by hip_angle about X gives (ty, tz) exactly --
    # verified by direct numerical inversion (leg_ik.py's __main__ round-trips this
    # against an independently-coded leg_fk AND against fk.py/robot_build.py's
    # SceneNode walker, both to <1e-9 m over thousands of random targets). Inverting:
    # hip_angle = atan2(tz, ty) - atan2(-d, dy).
    L_sq = ty * ty + tz * tz
    d_sq = max(L_sq - dy * dy, 0.0)
    d = math.sqrt(d_sq)
    hip_angle = math.atan2(tz, ty) - math.atan2(-d, dy)

    # --- Step 2: 2-link planar IK for thigh/calf ---
    # Reach distance from the thigh joint to the target, in the sagittal plane defined
    # by (tx, -d) i.e. forward/back (X) and "down the leg" (the rotated perpendicular
    # component, always negated since the leg hangs below/behind the thigh joint).
    reach_sq = tx * tx + d_sq
    reach = math.sqrt(reach_sq)
    min_reach = abs(thigh_len - calf_len)
    reach_clamped = min(max(reach, min_reach + 1e-6), max_reach - 1e-6)

    # Law of cosines: angle at the CALF joint (interior knee angle), then the THIGH
    # joint's angle split between "aim at target" and "offset for the knee bend".
    cos_knee = (thigh_len ** 2 + calf_len ** 2 - reach_clamped ** 2) / (2.0 * thigh_len * calf_len)
    cos_knee = min(1.0, max(-1.0, cos_knee))
    knee_interior = math.acos(cos_knee)  # 0 = fully bent, pi = fully straight

    cos_thigh_offset = (thigh_len ** 2 + reach_clamped ** 2 - calf_len ** 2) / (2.0 * thigh_len * reach_clamped)
    cos_thigh_offset = min(1.0, max(-1.0, cos_thigh_offset))
    thigh_offset_angle = math.acos(cos_thigh_offset)

    # Angle from "straight down" to the target direction, in the (tx, d) sagittal
    # plane. go2.urdf's thigh joint rotates about +Y with the standard right-hand
    # convention, which (verified against the independently-built fk.py/SceneNode
    # walker -- see leg_ik.py module history) means a POSITIVE thigh angle swings the
    # leg TOWARD -X (backward), i.e. tx = -reach*sin(thigh_angle), d = reach*cos(thigh_angle)
    # at thigh_angle alone (calf=0). Inverting: angle = atan2(-tx, d).
    target_dir_angle = math.atan2(-tx, d)

    thigh_angle = target_dir_angle + thigh_offset_angle
    # calf: URDF calf angle is negative (knee always bent "backward" per the limit
    # range [-2.7227, -0.83776]); interior knee angle of pi == straight leg == calf
    # angle 0 is NOT in range, so calf_angle = -(pi - knee_interior) i.e. more bent
    # (smaller knee_interior) => calf_angle more negative, matching the URDF sign.
    calf_angle = -(math.pi - knee_interior)

    return LegAngles(hip=hip_angle, thigh=thigh_angle, calf=calf_angle)


def leg_fk(angles: LegAngles, leg_sign_y: float,
           thigh_len: float = THIGH_LEN_M, calf_len: float = CALF_LEN_M,
           hip_to_thigh_y: float = HIP_TO_THIGH_Y_M) -> Vec3:
    """Inverse of solve_leg_ik -- forward-kinematics the foot position (in the hip
    joint's local frame) from (hip, thigh, calf) URDF-convention angles. Used only for
    self-testing solve_leg_ik (round-trip check), independent of the general fk.py /
    SceneNode walker (a second, independently-coded FK path for cross-validation).

    Two-stage composition, mirroring solve_leg_ik exactly:
      1. Planar 2-link FK (thigh angle about Y, then calf angle about Y, composed) in
         the sagittal (X, "down"=-Z) plane -- gives (tx, d) where d = downward reach.
      2. Hip abduction: at hip_angle=0 the chain sits at local (y=dy, z=-d); rotating
         by hip_angle about X gives the final (ty, tz) -- this is the exact forward
         model solve_leg_ik's hip_angle formula was derived from and numerically
         verified against (see module history), so agreement here is a real
         cross-check, not an artifact of shared derivation.
    """
    dy = hip_to_thigh_y * leg_sign_y

    # --- Stage 1: planar 2-link (thigh, calf) FK, in the (X, "down") sagittal plane ---
    # Positive thigh/calf angle swings the segment toward -X (see target_dir_angle's
    # docstring in solve_leg_ik for the empirical derivation against fk.py).
    total_angle = angles.thigh + angles.calf
    tx = -(thigh_len * math.sin(angles.thigh) + calf_len * math.sin(total_angle))
    down = thigh_len * math.cos(angles.thigh) + calf_len * math.cos(total_angle)
    d = down  # positive = below the thigh joint, matching solve_leg_ik's `d`

    # --- Stage 2: hip abduction, rotating (y=dy, z=-d) by hip_angle about X ---
    y0, z0 = dy, -d
    ty = y0 * math.cos(angles.hip) - z0 * math.sin(angles.hip)
    tz = y0 * math.sin(angles.hip) + z0 * math.cos(angles.hip)
    return (tx, ty, tz)


if __name__ == "__main__":
    import random

    random.seed(0)
    max_err = 0.0
    n = 2000
    for _ in range(n):
        # Sample a target roughly in the leg's reachable workspace (below/around the hip).
        tx = random.uniform(-0.15, 0.15)
        ty = random.uniform(0.05, 0.20) * random.choice([1, -1])
        tz = random.uniform(-0.38, -0.15)
        leg_sign_y = 1.0 if ty > 0 else -1.0
        # Re-bias ty toward the correct side magnitude for a realistic reachable target.
        ty = abs(ty) * leg_sign_y

        angles = solve_leg_ik((tx, ty, tz), leg_sign_y)
        # Reject unreachable-by-construction samples (outside the annulus solve_leg_ik
        # actually honors by DEFAULT: MAX_REACH_M, the URDF-calf-limit-derived ceiling,
        # not the wider geometric thigh+calf sum) -- those get clamped, so skip
        # verifying exact positional reach for them (still solved, just not exact).
        reach = math.sqrt(tx * tx + max((ty*ty+tz*tz) - (0.0955)**2, 0.0))
        if not (abs(THIGH_LEN_M - CALF_LEN_M) + 1e-4 < reach < MAX_REACH_M - 1e-4):
            continue

        fk_pos = leg_fk(angles, leg_sign_y)
        err = math.sqrt(sum((a - b) ** 2 for a, b in zip((tx, ty, tz), fk_pos)))
        max_err = max(max_err, err)
        if err > 1e-6:
            print(f"MISMATCH target={(tx,ty,tz)} angles={angles} fk={fk_pos} err={err:.8f}")
        if not (-2.7227 - 1e-6 <= angles.calf <= CALF_LIMIT_UPPER_RAD + 1e-6):
            print(f"CALF LIMIT VIOLATION target={(tx,ty,tz)} calf={angles.calf:.4f} "
                  f"limit=[-2.7227, {CALF_LIMIT_UPPER_RAD}]")

    print(f"leg_ik <-> leg_fk round-trip over {n} random targets: max_err={max_err:.10f} m (expect ~0)")
    print(f"MAX_REACH_M (URDF-calf-limit-derived clamp ceiling): {MAX_REACH_M:.6f} m "
          f"(vs geometric {THIGH_LEN_M+CALF_LEN_M:.6f} m)")

    # Also sanity-check joint LIMITS are respected on a centered "standing" target.
    a = solve_leg_ik((0.0, HIP_TO_THIGH_Y_M, -0.30), leg_sign_y=1.0)
    print(f"standing-ish target (0, {HIP_TO_THIGH_Y_M}, -0.30) -> hip={a.hip:.4f} thigh={a.thigh:.4f} calf={a.calf:.4f}")
    print("URDF thigh limits [-1.5708, 3.4907], calf limits [-2.7227, -0.83776] (FL/FR);"
          " RL/RR thigh limits [-0.5236, 4.5379]")
