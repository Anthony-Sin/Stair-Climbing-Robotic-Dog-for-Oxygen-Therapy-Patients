"""UsdSkel animation-channel surgery for the simulated patient's walk clip.

Split out of ``sim_person_actor``: low-level helpers that zero the baked root
motion/rotation, loop the walk animation at a faster cadence, and estimate the
clip's gait period. They operate on a UsdSkel ``SkelAnimation`` prim passed in
and have no dependency back on ``sim_person_actor``.
"""
import logging
import math
from typing import Any, List, Optional

import numpy as np
from pxr import Gf

from sim_logging_utils import log_event

# Leg-cycle speed-up baked into the walk clip loop. >1 makes the legs cycle
# faster, so at a given body speed each step covers less ground -> shorter,
# quicker steps (the baked clip's stride is otherwise long/gliding). Applied in
# _loop_animation_channels at load time. Tunable: raise for shorter steps, 1.0
# for the original cadence. Eyeball and adjust; the per-run diagnostics log the
# clip's true period so this can be set precisely.
_PERSON_GAIT_CADENCE_MULT = 1.8

def _is_zero_vec3(value: Any, *, tol: float = 1e-4) -> bool:
    try:
        return bool(Gf.IsClose(value, Gf.Vec3f(0.0, 0.0, 0.0), tol))
    except Exception:
        try:
            return (
                abs(float(value[0])) <= tol
                and abs(float(value[1])) <= tol
                and abs(float(value[2])) <= tol
            )
        except Exception:
            return False


def _zero_root_translation_channel(anim: Any, root_idx: int) -> bool:
    trans_attr = anim.GetTranslationsAttr()
    changed = False

    time_samples = trans_attr.GetTimeSamples()
    if time_samples:
        for t in time_samples:
            vals = trans_attr.Get(t)
            if vals and len(vals) > root_idx and not _is_zero_vec3(vals[root_idx]):
                vals_list = list(vals)
                vals_list[root_idx] = Gf.Vec3f(0.0, 0.0, 0.0)
                trans_attr.Set(vals_list, t)
                changed = True
    else:
        val = trans_attr.Get()
        if val and len(val) > root_idx and not _is_zero_vec3(val[root_idx]):
            vals_list = list(val)
            vals_list[root_idx] = Gf.Vec3f(0.0, 0.0, 0.0)
            trans_attr.Set(vals_list)
            changed = True

    return changed


def _identity_quat_like(value: Any) -> Any:
    try:
        if isinstance(value, Gf.Quatf):
            return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        if isinstance(value, Gf.Quath):
            return Gf.Quath(1.0, 0.0, 0.0, 0.0)
    except Exception:
        pass
    return Gf.Quatd(1.0, 0.0, 0.0, 0.0)


def _zero_root_rotation_channel(anim: Any, root_idx: int) -> bool:
    rotations_attr = anim.GetRotationsAttr()
    changed = False

    time_samples = rotations_attr.GetTimeSamples()
    if time_samples:
        for t in time_samples:
            vals = rotations_attr.Get(t)
            if vals and len(vals) > root_idx:
                vals_list = list(vals)
                vals_list[root_idx] = _identity_quat_like(vals_list[root_idx])
                rotations_attr.Set(vals_list, t)
                changed = True
    else:
        val = rotations_attr.Get()
        if val and len(val) > root_idx:
            vals_list = list(val)
            vals_list[root_idx] = _identity_quat_like(vals_list[root_idx])
            rotations_attr.Set(vals_list)
            changed = True

    return changed


def _loop_animation_channels(
    anim: Any,
    loop_duration: float = 80.0,
    t_start: float = 186.0,
    cadence_mult: float = 1.0,
) -> bool:
    """Re-tile all animation channels using a stable mid-clip walk cycle window.

    Instead of looping from t=0 (which includes a startup transition / rest pose
    that causes a visible freeze at every loop boundary), we extract the cycle
    window [t_start, t_start + loop_duration) from a region where the walk is
    already fully stabilised, normalise those time samples to [0, loop_duration),
    then fill every time sample in the original clip by mapping it into that
    normalised window.  The result is a seamlessly repeating mid-stride cycle
    with no rest-pose pop at the seam.

    cadence_mult > 1 advances the loop phase faster relative to the timeline, so
    the legs cycle quicker and each step covers less ground (shorter, quicker
    steps) WITHOUT shrinking the content window -- the full stride content is
    preserved, only its playback speed changes.
    """
    changed = False
    t_end = t_start + loop_duration
    cadence_mult = float(cadence_mult) if cadence_mult and cadence_mult > 0.0 else 1.0
    for attr in [anim.GetTranslationsAttr(), anim.GetRotationsAttr(), anim.GetScalesAttr()]:
        if not attr.IsValid():
            continue
        time_samples = attr.GetTimeSamples()
        if not time_samples:
            continue

        # 1. Collect the stable window [t_start, t_end) and normalise to [0, loop_duration)
        cache = {}  # normalised_t -> value
        for t in time_samples:
            if t_start <= t < t_end:
                cache[t - t_start] = attr.Get(t)

        # Fallback: if no samples found in window, use the original approach from t=0
        if not cache:
            for t in time_samples:
                if t <= loop_duration:
                    cache[t] = attr.Get(t)
            if not cache:
                continue

        sorted_keys = sorted(cache.keys())

        # 2. For every time sample in the original clip, map into [0, loop_duration)
        #    and pick the nearest cached sample. cadence_mult speeds up the phase
        #    advance so the same stride content plays in fewer timeline units.
        for t in time_samples:
            t_norm = (t * cadence_mult) % loop_duration
            best_key = min(sorted_keys, key=lambda k: abs(k - t_norm))
            val = cache[best_key]
            attr.Set(val, t)
            changed = True
    return changed



def _estimate_gait_period(anim: Any, joints_list: List[str], logger: Optional[logging.Logger]) -> None:
    """LOG-ONLY diagnostic for the walk clip — never edits the clip, never raises.

    The local Biped_Setup copy is a binary USDC, so we cannot read its timing
    offline. This logs, from a normal run: the full joint-name list (needed to
    resolve leg joints for any future foot work), the rotation channel's
    time-sample span/count + sampling rate, and an autocorrelation estimate of the
    true full gait period of a left-leg joint. That estimate tells us whether the
    current loop window (loop_duration=80, t_start=186) captures a FULL two-step
    cycle or only a half cycle (the suspected cause of the one-sided "left side"
    look). The loop value is then set from this logged number rather than guessed.
    """
    try:
        rot_attr = anim.GetRotationsAttr()
        ts = sorted(rot_attr.GetTimeSamples()) if (rot_attr and rot_attr.IsValid()) else []

        try:
            tcps = float(anim.GetPrim().GetStage().GetTimeCodesPerSecond())
        except Exception:
            tcps = None

        # Joint tokens are full paths (e.g. "Root/Pelvis/L_UpLeg"); match the leaf
        # bone name. The upper-leg (hip) joint carries the clearest 1-per-cycle
        # swing, so prefer L_UpLeg / LeftUpLeg / *Thigh.
        leg_idx = None
        leg_name = None
        for idx, j in enumerate(joints_list):
            leaf = str(j).rsplit("/", 1)[-1].lower()
            is_left = leaf.startswith("l_") or leaf.startswith("left")
            is_upper_leg = any(k in leaf for k in ("upleg", "thigh", "femur"))
            if is_left and is_upper_leg:
                leg_idx, leg_name = idx, str(j)
                break

        period = None
        confidence = 0.0
        if ts and leg_idx is not None:
            angles = []
            for t in ts:
                vals = rot_attr.Get(t)
                if not vals or len(vals) <= leg_idx:
                    angles = []
                    break
                q = vals[leg_idx]
                try:
                    w = float(q.GetReal())
                except Exception:
                    w = float(getattr(q, "real", 1.0))
                w = max(-1.0, min(1.0, w))
                angles.append(2.0 * math.acos(abs(w)))  # rotation magnitude per sample
            if len(angles) > 8:
                sig = np.asarray(angles, dtype=float)
                sig = sig - sig.mean()
                if np.any(sig):
                    ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
                    ac = ac / ac[0]
                    lag = None
                    for i in range(2, len(ac) - 1):
                        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.3:
                            lag = i
                            break
                    if lag is not None:
                        dts = np.diff(np.asarray(ts, dtype=float))
                        med = float(np.median(dts)) if len(dts) else 1.0
                        period = lag * med
                        confidence = float(ac[lag])

        period_seconds = (period / tcps) if (period and tcps) else None
        leg_joints = [str(j).rsplit("/", 1)[-1] for j in joints_list
                      if str(j).rsplit("/", 1)[-1].lower().startswith(("l_", "r_"))
                      and any(k in str(j).lower() for k in ("upleg", "loleg", "ankle", "ball"))]
        if logger is not None:
            log_event(
                logger,
                logging.INFO,
                "person_walk_clip_diagnostics",
                "Walk clip timing + estimated gait period (diagnostic only; clip unchanged)",
                leg_joints=leg_joints,
                joints_count=len(joints_list),
                time_sample_count=len(ts),
                t_min=(float(ts[0]) if ts else None),
                t_max=(float(ts[-1]) if ts else None),
                time_codes_per_second=tcps,
                leg_joint=leg_name,
                leg_joint_index=leg_idx,
                estimated_full_period=period,
                estimated_full_period_seconds=(round(period_seconds, 4) if period_seconds else None),
                autocorr_confidence=round(confidence, 3),
                current_loop_duration=80.0,
                current_loop_t_start=186.0,
                current_cadence_mult=_PERSON_GAIT_CADENCE_MULT,
            )
    except Exception as exc:
        if logger is not None:
            log_event(
                logger,
                logging.WARNING,
                "person_walk_clip_diagnostics_failed",
                "Could not compute walk clip diagnostics",
                error=str(exc),
            )
