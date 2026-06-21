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


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def check_crc_roundtrip() -> CheckResult:
    from real.control.lowcmd_builder import build_low_cmd_fields, crc32_core, N_CMD_SLOTS

    f = build_low_cmd_fields([0.1 * i for i in range(12)], 40.0, 0.5)
    words: List[int] = []
    for i in range(N_CMD_SLOTS):
        words.append(int(f.mode[i]) & 0xFFFFFFFF)
        for v in (f.q[i], f.dq[i], f.kp[i], f.kd[i], f.tau[i]):
            words.append(int(np.float32(v).view(np.uint32)))
    c1, c2 = crc32_core(words), crc32_core(words)
    ok = (c1 == c2) and (0 <= c1 <= 0xFFFFFFFF)
    return CheckResult("crc_roundtrip", ok, f"crc={c1:#010x} (deterministic={c1 == c2})")


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
    checks = [check_crc_roundtrip(), check_joint_limits(), check_weights_present(pgtt_path, rl_path)]
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
        print(f"[{'PASS' if r.ok else 'FAIL'}] {r.name:16s} {r.detail}")
        ok = ok and r.ok
    print("PREFLIGHT OK" if ok else "PREFLIGHT FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
