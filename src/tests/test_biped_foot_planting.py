"""Host (no-Isaac) tests for the patient's foot-planting gait.

These prove the property that removes the "gliding" symptom WITHOUT booting Isaac:
a planted foot stays fixed in world space through the whole stance phase, on flat
ground and on stairs. They also check the sagittal 2-bone IK round-trips and that
the swing foot actually lifts.

Run: python tests/test_biped_foot_planting.py  (or via pytest)
"""

import math
import os
import sys
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "isaac"))

from biped_anim import BipedAnimationController  # noqa: E402
from biped_anim.foot_planting import (  # noqa: E402
    LegGeometry,
    foot_cycle,
    neutral_leg_angles,
    planted_foot_offset,
    solve_leg_ik,
)
from biped_anim.gait import FlatWalk, Idle, StairClimb  # noqa: E402
from biped_anim.terrain_classifier import StairGeometry, TerrainClassifier  # noqa: E402
from biped_anim.types import AnimStyle  # noqa: E402

# Representative adult-leg proportions (metres); the real values come from rig FK.
GEOM = LegGeometry(thigh_m=0.45, shin_m=0.45, reach_m=0.86)


def _fk(hip_flex: float, knee_flex: float, geom: LegGeometry):
    """Forward kinematics: (hip_flex, knee_flex) -> foot (forward, down) below hip."""
    knee_x = geom.thigh_m * math.sin(hip_flex)
    knee_d = geom.thigh_m * math.cos(hip_flex)
    shin_ang = hip_flex - knee_flex
    foot_x = knee_x + geom.shin_m * math.sin(shin_ang)
    foot_d = knee_d + geom.shin_m * math.cos(shin_ang)
    return foot_x, foot_d


def test_ik_round_trips():
    """IK then FK returns the requested foot target across the reachable range."""
    checked = 0
    for fwd in (-0.25, -0.1, 0.0, 0.15, 0.3):
        for down in (0.6, 0.7, 0.8, 0.85):
            if math.hypot(fwd, down) > GEOM.max_reach_m - 1e-3:
                continue  # beyond leg reach -> IK clamps (covered separately below)
            hip, knee = solve_leg_ik(fwd, down, GEOM)
            fx, fd = _fk(hip, knee, GEOM)
            assert abs(fx - fwd) < 1e-6, (fwd, down, fx)
            assert abs(fd - down) < 1e-6, (fwd, down, fd)
            assert knee >= -1e-9, knee  # knee only ever flexes (>= straight)
            checked += 1
    assert checked >= 12, checked


def test_ik_clamps_unreachable_target():
    """A foot target past full reach straightens the leg instead of producing NaN."""
    hip, knee = solve_leg_ik(0.3, 0.85, GEOM)  # hypot 0.901 > max reach 0.90
    assert math.isfinite(hip) and math.isfinite(knee)
    assert knee < 0.12, knee  # leg driven essentially straight (~5 deg of bend)


def test_neutral_is_straight_below():
    """The standing neutral places the foot straight under the hip at full reach."""
    hip, knee = neutral_leg_angles(GEOM)
    fx, fd = _fk(hip, knee, GEOM)
    assert abs(fx) < 1e-6, fx
    assert abs(fd - GEOM.reach_m) < 1e-6, fd


def _stance_world_track(stride, stance_frac, reach, cycle_rise, n=40):
    """Sample the planted foot's WORLD position across the stance phase.

    The body advances ``stride`` forward and ``cycle_rise`` up per cycle, linearly
    in phase (phase is distance-synced; the stair ramp is linear in distance). A
    foot that does not skate must hold one world position for the whole stance.
    """
    xs, zs = [], []
    for k in range(n):
        phi = stance_frac * (k / (n - 1)) * 0.999  # span [0, stance_frac)
        s, down = planted_foot_offset(
            phi, stride, stance_frac=stance_frac, clearance_m=0.1,
            reach_m=reach, cycle_rise_m=cycle_rise,
        )
        hip_x = phi * stride
        hip_z = phi * cycle_rise
        xs.append(hip_x + s)        # world forward position of the foot
        zs.append(hip_z - down)     # world height of the foot (down is below hip)
    return xs, zs


def test_planted_foot_does_not_skate_flat():
    xs, zs = _stance_world_track(stride=0.8, stance_frac=0.62, reach=0.86, cycle_rise=0.0)
    assert max(xs) - min(xs) < 1e-9, ("flat foot slid forward", max(xs) - min(xs))
    assert max(zs) - min(zs) < 1e-9, ("flat foot slid vertically", max(zs) - min(zs))


def test_planted_foot_does_not_skate_stairs():
    # 0.30 m tread, 0.15 m riser -> stride 0.60, cycle climbs 2 risers = 0.30 m.
    xs, zs = _stance_world_track(stride=0.6, stance_frac=0.52, reach=0.80, cycle_rise=0.30)
    assert max(xs) - min(xs) < 1e-9, ("stair foot slid forward", max(xs) - min(xs))
    assert max(zs) - min(zs) < 1e-9, ("stair foot slid vertically", max(zs) - min(zs))


def test_swing_foot_lifts_and_lands():
    reach, stride, c = 0.86, 0.8, 0.62
    # Foot at mid-swing must be higher (smaller "down") than when planted.
    _, down_plant = planted_foot_offset(
        0.0, stride, stance_frac=c, clearance_m=0.1, reach_m=reach, cycle_rise_m=0.0
    )
    mid_swing = c + (1.0 - c) * 0.5
    _, down_mid = planted_foot_offset(
        mid_swing, stride, stance_frac=c, clearance_m=0.1, reach_m=reach, cycle_rise_m=0.0
    )
    assert down_mid < down_plant - 0.05, (down_plant, down_mid)


def test_gait_evaluate_uses_ik_and_is_finite():
    """FlatWalk/StairClimb with geometry produce finite, sensible leg deltas."""
    flat = FlatWalk(leg_geom=GEOM)
    assert flat.leg_geom is not None
    stairs = StairClimb(leg_geom=GEOM)
    geom = StairGeometry(
        start_x_m=2.0, end_x_m=4.0, step_height_m=0.15, step_depth_m=0.30,
        half_width_m=1.05, step_count=6,
    )
    # Stair stride locks to two treads; the cycle climbs two risers.
    assert abs(stairs.stride_length(0.5, geom) - 0.6) < 1e-9
    assert abs(stairs._cycle_rise(0.6, geom) - 0.30) < 1e-9

    stair_knees = []
    for phase in [i / 24.0 for i in range(24)]:
        pf = flat.evaluate(phase, 0.5, None)
        ps = stairs.evaluate(phase, 0.5, geom)
        for v in (pf.hip_l, pf.knee_l, pf.ankle_l, ps.hip_l, ps.knee_l):
            assert math.isfinite(v)
        stair_knees.append(ps.knee_l)
    # The knee must cycle: bend hard during swing, much less while planted.
    assert max(stair_knees) > 0.5, max(stair_knees)
    assert max(stair_knees) - min(stair_knees) > 0.2, (min(stair_knees), max(stair_knees))


def test_foot_cycle_stance_then_swing():
    """foot_cycle reports stance (w=-1) with a linear sweep, then a 0..1 swing."""
    stride, c = 0.6, 0.5
    s0, w0 = foot_cycle(0.0, stride, c)
    s_mid_stance, w_ms = foot_cycle(c * 0.5, stride, c)
    assert w0 == -1.0 and w_ms == -1.0  # planted
    assert s0 > s_mid_stance            # sweeping backward (front -> back)
    s_sw, w_sw = foot_cycle(c + (1 - c) * 0.5, stride, c)
    assert 0.0 <= w_sw <= 1.0           # mid-swing


@dataclass
class _Spec:
    start_x_m: float = 2.0
    end_x_m: float = 4.0
    step_height_m: float = 0.15
    step_depth_m: float = 0.30
    half_width_m: float = 1.05
    step_count: int = 6
    top_height_m: float = 0.9


def _tread_top(spec, x, y):
    """Discrete tread-top height (mirror of isaac_env.get_terrain_height)."""
    if not (-spec.half_width_m <= y <= spec.half_width_m):
        return 0.0
    if spec.start_x_m <= x < spec.end_x_m:
        idx = int((x - spec.start_x_m) / spec.step_depth_m)
        return min(spec.top_height_m, (idx + 1) * spec.step_height_m)
    return spec.top_height_m if x >= spec.end_x_m else 0.0


class _FakeRig:
    ready = True

    def __init__(self, geom):
        self.leg_geometry = geom
        self.last = None

    def apply(self, pose):
        self.last = pose


def test_ground_referenced_feet_plant_on_treads_without_pop():
    """Full controller pipeline on stairs: feet land ON the tread tops (no float,
    no penetration) and the swing foot does not pop at nosing crossings."""
    spec = _Spec()
    ankle_above_root = 0.09
    hip_above_root = ankle_above_root + GEOM.reach_m
    rig = _FakeRig(GEOM)
    clock = [0.0]
    ctrl = BipedAnimationController(
        rig,
        classifier=TerrainClassifier(lambda: spec),
        gaits={
            AnimStyle.IDLE: Idle(),
            AnimStyle.FLAT_WALK: FlatWalk(leg_geom=GEOM),
            AnimStyle.STAIR_CLIMB: StairClimb(leg_geom=GEOM),
        },
        clock=lambda: clock[0],
    )
    h0, k0 = neutral_leg_angles(GEOM)

    def foot_world_z(pose, root_z, leg):
        hip = (pose.hip_l if leg == "l" else pose.hip_r) + h0
        knee = (pose.knee_l if leg == "l" else pose.knee_r) + k0
        fx = GEOM.thigh_m * math.sin(hip) + GEOM.shin_m * math.sin(hip - knee)
        fd = GEOM.thigh_m * math.cos(hip) + GEOM.shin_m * math.cos(hip - knee)
        return fx, (root_z + hip_above_root) - fd

    ctrl.reset((2.0, 0.0))
    dt, speed, tau = 0.02, 0.55, 0.12
    vis = _tread_top(spec, 2.2, 0.0)
    x = 2.2
    cl_min, cl_max, max_jump = 9.0, -9.0, 0.0
    prev_lz = None
    while x < 3.7:
        clock[0] += dt
        x += speed * dt
        vis += min(1.0, dt / tau) * (_tread_top(spec, x, 0.0) - vis)
        ctrl.update(x, 0.0, moving_hint=True, body_z=vis,
                    ground_height_fn=lambda fx, fy: _tread_top(spec, fx, fy))
        pose = rig.last
        if clock[0] <= 1.0:  # skip the entry crossfade
            continue
        for leg in ("l", "r"):
            fx, fz = foot_world_z(pose, vis, leg)
            tread = _tread_top(spec, x + fx, 0.0)
            cl = fz - (tread + ankle_above_root)  # 0 == foot resting on its tread
            cl_min, cl_max = min(cl_min, cl), max(cl_max, cl)
        _, lz = foot_world_z(pose, vis, "l")
        if prev_lz is not None:
            max_jump = max(max_jump, abs(lz - prev_lz))
        prev_lz = lz

    assert cl_min > -0.03, ("a foot penetrates the tread", cl_min)
    assert cl_max > 0.08, ("swing foot never clears the step", cl_max)
    assert max_jump < 0.05, ("swing foot pops at a nosing crossing", max_jump)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
