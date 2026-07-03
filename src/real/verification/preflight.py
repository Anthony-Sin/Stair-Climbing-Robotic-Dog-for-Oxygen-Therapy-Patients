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


def check_lidar_extrinsics(heightscan_mode: str = "flat") -> CheckResult:
    """Placeholder LiDAR->base extrinsics: WARN in flat mode, FAIL in lidar mode.

    Truth (corrected from the old docstring that implied a settable ROS param): the
    lidar->base extrinsic is a HARDCODED placeholder in
    ``real.perception.pointcloud_interface._EXTRINSICS`` (identity R + t=[0,0,0.10]). It is
    NOT read from a ROS parameter -- to change it you edit that constant. It can only be
    MEASURED on the robot (DEPLOY.md), and a wrong extrinsic shifts the whole elevation map.

    In FLAT heightscan mode the extrinsic is unused, so the placeholder is a WARN. In LIDAR
    mode it drives control, so an unmeasured placeholder is a FAIL -- do NOT let it silently
    ship a run that steers on a mis-registered heightmap.
    """
    _PLACEHOLDER_T = (0.0, 0.0, 0.10)
    lidar = str(heightscan_mode).strip().lower() == "lidar"
    try:
        from real.perception.pointcloud_interface import _EXTRINSICS  # type: ignore
    except Exception as exc:
        # Cannot introspect: in lidar mode this is unsafe to run past -> FAIL; in flat -> warn.
        return CheckResult("lidar_extrinsics", not lidar,
                           f"could not import extrinsics ({type(exc).__name__}); MEASURE before a run",
                           warn=not lidar)

    stale = []
    for sku, ext in _EXTRINSICS.items():
        is_identity_R = bool(np.allclose(np.asarray(ext.R, dtype=np.float32), np.eye(3, dtype=np.float32)))
        is_placeholder_t = bool(np.allclose(np.asarray(ext.t, dtype=np.float32),
                                            np.asarray(_PLACEHOLDER_T, dtype=np.float32)))
        if is_identity_R and is_placeholder_t:
            stale.append(str(sku))
    if stale:
        detail = (f"PLACEHOLDER extrinsics still set for {stale} (identity R + t={_PLACEHOLDER_T}); "
                  f"edit pointcloud_interface._EXTRINSICS with the MEASURED lidar->base transform "
                  f"before a run")
        if lidar:
            # Lidar mode drives control on this: FAIL (not just warn).
            return CheckResult("lidar_extrinsics", False, detail + " [FAIL: heightscan_mode=lidar]")
        return CheckResult("lidar_extrinsics", True, detail + " [warn: flat mode, unused]", warn=True)
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


def run_pure_checks(pgtt_path: str, rl_path: str, *, load_policies: bool = True,
                    heightscan_mode: str = "flat") -> List[CheckResult]:
    checks = [
        check_crc_roundtrip(), check_joint_limits(),
        check_lidar_extrinsics(heightscan_mode), check_weights_present(pgtt_path, rl_path),
    ]
    if load_policies:
        checks.append(check_policies_load(pgtt_path, rl_path))
    return checks


def _repo_root() -> str:
    # .../src/real/verification/preflight.py -> repo root is four dirs up.
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))


def _scalar_from_yaml(params_path: str, key: str, default: str) -> str:
    """Return the value for a simple ``  key: value`` line in the params yaml.

    Prefers PyYAML (present in any ROS 2 env); if PyYAML is unavailable (a bare dev host)
    falls back to a minimal line scan so preflight still validates the ACTUAL yaml paths --
    the whole point of the config-rot fix -- rather than a divergent hardcoded set. The
    params yaml is flat ``key: value`` scalars, so the line scan is sufficient here.
    """
    try:
        import yaml  # lazy
        with open(params_path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        for section in doc.values():
            params = (section or {}).get("ros__parameters") if isinstance(section, dict) else None
            if isinstance(params, dict) and key in params:
                return str(params[key])
        return default
    except ImportError:
        import re
        pat = re.compile(r"^\s*" + re.escape(key) + r"\s*:\s*(.+?)\s*(#.*)?$")
        try:
            with open(params_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    m = pat.match(line)
                    if m:
                        return m.group(1).strip().strip('"').strip("'")
        except OSError:
            pass
        return default


def weight_paths_from_yaml(params_path: str) -> "tuple[str, str]":
    """Read the SAME real_robot.yaml the launch loads and return (pgtt, rl) weight paths.

    This closes the config-rot gap: preflight used to validate DIFFERENT hardcoded absolute
    paths than the node loaded from the yaml, so a stale/relocated yaml path passed preflight
    yet crashed the control node on FileNotFoundError (robot folded, nothing driving). We
    parse the yaml's weight paths and resolve any relative path against the repo root (the
    launch/run CWD per DEPLOY.md), exactly as the node's CWD-relative load does.
    """
    pgtt = _scalar_from_yaml(params_path, "pgtt_policy_path",
                             "src/sim/models/pgtt/pgtt_go2_level17.npz")
    rl = _scalar_from_yaml(params_path, "rl_policy_path",
                           "src/sim/models/locomotion/go2_robot_lab_policy.pt")
    root = _repo_root()
    resolve = lambda p: p if os.path.isabs(p) else os.path.join(root, p)
    return resolve(pgtt), resolve(rl)


def heightscan_mode_from_yaml(params_path: str) -> str:
    """Read ``heightscan_mode`` ('flat'|'lidar') from the launch yaml so the extrinsics
    check can FAIL (not warn) when lidar mode ships with placeholder extrinsics.
    """
    return _scalar_from_yaml(params_path, "heightscan_mode", "flat")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-run sanity checks for the real Go2 controller.")
    _default_params = os.path.join(_repo_root(), "src", "real", "config", "real_robot.yaml")
    ap.add_argument("--params", default=_default_params,
                    help="the real_robot.yaml the launch loads; its weight paths are validated "
                         "(so preflight checks EXACTLY what the control node will load)")
    ap.add_argument("--pgtt", default=None,
                    help="override the pgtt weight path (else read from --params yaml)")
    ap.add_argument("--rl", default=None,
                    help="override the rl weight path (else read from --params yaml)")
    ap.add_argument("--no-load", action="store_true", help="skip the (slow) policy-load check")
    args = ap.parse_args()

    # Validate the SAME paths the node loads from the yaml unless explicitly overridden.
    pgtt_path, rl_path = args.pgtt, args.rl
    if pgtt_path is None or rl_path is None:
        try:
            y_pgtt, y_rl = weight_paths_from_yaml(args.params)
        except Exception as exc:
            print(f"[FAIL] params_yaml       could not read {args.params}: {exc}")
            sys.exit(1)
        pgtt_path = pgtt_path or y_pgtt
        rl_path = rl_path or y_rl
    print(f"[INFO] validating weights the control node will load (from {args.params}):")
    print(f"[INFO]   pgtt = {pgtt_path}")
    print(f"[INFO]   rl   = {rl_path}")
    hs_mode = heightscan_mode_from_yaml(args.params)
    print(f"[INFO]   heightscan_mode = {hs_mode} "
          f"(lidar mode FAILS on placeholder extrinsics)")

    results = run_pure_checks(pgtt_path, rl_path, load_policies=not args.no_load,
                              heightscan_mode=hs_mode)
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
