"""Standalone pre-run sanity checks -- run BEFORE any motion (no ROS graph needed).

These are the offline-checkable gates: the policy weights load, the LowCmd CRC is
working, and the policies' default poses are inside the Go2 joint limits. A failure
here means do NOT start the run. The LIVE checks (/lowstate freshness, sport mode
released, camera/LiDAR producing frames) are enforced by the control node's startup
gate at runtime, where the ROS graph exists.

Run:  python -m real.verification.preflight [--pgtt PATH --rl PATH]
Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

# Go2 joint position limits (rad), per (leg, joint) -- the same envelope the IK
# climber clamps to. Used to confirm a policy's neutral pose cannot command a stop.
_JOINT_LIMITS = {"hip": (-1.00, 1.00), "thigh": (-1.00, 3.40), "calf": (-2.68, -0.90)}

# SELF-CONSISTENCY regression golden (NOT hardware-validated). This pins the CRC OUR
# encoder currently produces for the FIXED documented input
#   build_low_cmd_fields([0.1*i for i in range(12)], 40.0, 0.5)
# serialized exactly as _finalize_crc / the roundtrip below does (mode word + the five
# float32-as-uint32 words per slot, over all N_CMD_SLOTS). Its ONLY job is to catch an
# accidental byte-layout / field-ordering regression in our serialization. It does NOT
# prove the layout matches the firmware struct -- that cross-check can only be done on
# the robot (see the hardware TODO in check_crc_roundtrip). If the firmware's true CRC
# is ever measured, add a SEPARATE hardware-golden assert; do not overwrite this one.
_CRC_SELFCONSISTENCY_GOLDEN = 0x1F9E26F0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    # A warn-only check does NOT fail preflight (ok stays True) but flags a condition
    # that must be resolved before a real run (e.g. an unmeasured placeholder value).
    warn: bool = False


def check_crc_roundtrip() -> CheckResult:
    from real.control.lowcmd_builder import build_low_cmd_fields, crc32_core, N_CMD_SLOTS

    f = build_low_cmd_fields([0.1 * i for i in range(12)], 40.0, 0.5)
    words: List[int] = []
    for i in range(N_CMD_SLOTS):
        words.append(int(f.mode[i]) & 0xFFFFFFFF)
        for v in (f.q[i], f.dq[i], f.kp[i], f.kd[i], f.tau[i]):
            words.append(int(np.float32(v).view(np.uint32)))
    c1 = crc32_core(words)
    # Golden vector: empty input must return the CRC seed (0xFFFFFFFF) — verifies
    # initialization. c1 must be non-zero for a non-trivial input.
    c_seed_ok = crc32_core([]) == 0xFFFFFFFF
    # SELF-CONSISTENCY golden: our serialization for this fixed input must not drift.
    # This catches accidental byte-layout / ordering regressions in OUR encoder; it is
    # NOT hardware-validated. HARDWARE TODO: once the firmware's true CRC for a known
    # command is measured on the robot, add a separate hardware-golden assert here:
    #   assert <firmware_crc_for_known_cmd> == <measured_on_robot>
    golden_ok = (c1 == _CRC_SELFCONSISTENCY_GOLDEN)
    ok = c_seed_ok and golden_ok and (0 < c1 <= 0xFFFFFFFF)
    return CheckResult(
        "crc_roundtrip", ok,
        f"crc={c1:#010x} (seed_ok={c_seed_ok}, golden_ok={golden_ok} "
        f"[self-consistency, expect {_CRC_SELFCONSISTENCY_GOLDEN:#010x}])",
    )


def check_joint_limits() -> CheckResult:
    from go2_locomotion.go2_locomotion_utils import PGTT_DEFAULT_POSE
    from go2_locomotion.rl_locomotion_policy import POLICY_DEFAULT_BY_JOINT

    bad = []
    for (leg, joint), v in PGTT_DEFAULT_POSE.items():
        lo, hi = _JOINT_LIMITS[joint]
        if not (lo <= v <= hi):
            bad.append(f"pgtt {leg}_{joint}={v}")
    for joint, v in POLICY_DEFAULT_BY_JOINT.items():
        lo, hi = _JOINT_LIMITS[joint]
        if not (lo <= v <= hi):
            bad.append(f"blind_rl {joint}={v}")
    return CheckResult("joint_limits", not bad, "all default poses in-limit" if not bad else ", ".join(bad))


def check_lidar_extrinsics() -> CheckResult:
    """WARN (not fail) when the LiDAR->base extrinsics are still the placeholder value.

    The extrinsics can only be MEASURED on the robot (DEPLOY.md). We cannot measure
    them here, so we flag when the shipped placeholder (identity R + t=[0,0,0.10]) is
    still in place, so it can't silently ship. Detectable because the placeholder is a
    clearly-identifiable constant in ``real.perception.pointcloud_interface._EXTRINSICS``.
    """
    _PLACEHOLDER_T = (0.0, 0.0, 0.10)
    try:
        from real.perception.pointcloud_interface import _EXTRINSICS  # type: ignore
    except Exception as exc:
        # Cannot introspect -> warn (better than silently assuming it's fine).
        return CheckResult("lidar_extrinsics", True,
                           f"could not import extrinsics ({type(exc).__name__}); MEASURE before a run",
                           warn=True)

    stale = []
    for sku, ext in _EXTRINSICS.items():
        is_identity_R = bool(np.allclose(np.asarray(ext.R, dtype=np.float32), np.eye(3, dtype=np.float32)))
        is_placeholder_t = bool(np.allclose(np.asarray(ext.t, dtype=np.float32),
                                            np.asarray(_PLACEHOLDER_T, dtype=np.float32)))
        if is_identity_R and is_placeholder_t:
            stale.append(str(sku))
    if stale:
        return CheckResult(
            "lidar_extrinsics", True,
            f"PLACEHOLDER extrinsics still set for {stale} (identity R + t={_PLACEHOLDER_T}); "
            f"MEASURE lidar->base and set real_robot.yaml before a run",
            warn=True,
        )
    return CheckResult("lidar_extrinsics", True, "extrinsics differ from placeholder")


def check_weights_present(pgtt_path: str, rl_path: str) -> CheckResult:
    missing = [p for p in (pgtt_path, rl_path) if not os.path.exists(p)]
    return CheckResult("weights_present", not missing,
                       "found" if not missing else f"MISSING: {missing}")


def check_policies_load(pgtt_path: str, rl_path: str) -> CheckResult:
    try:
        from go2_locomotion.pgtt_locomotion_policy import PgttLocomotionPolicy, PgttPolicyConfig
        from go2_locomotion.rl_locomotion_policy import RLLocomotionPolicy, RLLocomotionPolicyConfig
        from real.control.lowstate_articulation import GO2_DOF_NAMES

        dof = list(GO2_DOF_NAMES)
        PgttLocomotionPolicy(PgttPolicyConfig(policy_path=pgtt_path), dof)
        RLLocomotionPolicy(RLLocomotionPolicyConfig(policy_path=rl_path), dof)
        return CheckResult("policies_load", True, "PGTT + blind_rl instantiated on CPU")
    except Exception as exc:
        return CheckResult("policies_load", False, f"{type(exc).__name__}: {exc}")


def run_pure_checks(pgtt_path: str, rl_path: str, *, load_policies: bool = True) -> List[CheckResult]:
    checks = [
        check_crc_roundtrip(), check_joint_limits(),
        check_lidar_extrinsics(), check_weights_present(pgtt_path, rl_path),
    ]
    if load_policies:
        checks.append(check_policies_load(pgtt_path, rl_path))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-run sanity checks for the real Go2 controller.")
    ap.add_argument("--pgtt", default="sim/models/pgtt/pgtt_go2_level17.npz")
    ap.add_argument("--rl", default="sim/models/locomotion/go2_robot_lab_policy.pt")
    ap.add_argument("--no-load", action="store_true", help="skip the (slow) policy-load check")
    args = ap.parse_args()

    results = run_pure_checks(args.pgtt, args.rl, load_policies=not args.no_load)
    ok = True
    for r in results:
        if not r.ok:
            tag = "FAIL"
        elif r.warn:
            tag = "WARN"
        else:
            tag = "PASS"
        print(f"[{tag}] {r.name:16s} {r.detail}")
        ok = ok and r.ok
    print("PREFLIGHT OK" if ok else "PREFLIGHT FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
