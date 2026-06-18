"""Spawn the oxygen-concentrator payload onto the Go2 in Isaac Sim.

This is the only Isaac-Sim-dependent builder. ``pxr`` is imported lazily inside
:func:`attach_o2_payload` so the rest of the package (spec / geometry / asset
generation) stays importable under plain Python.

Physics model (matches the developer's intent):

  * RAILS (the 3D-printed cradle, 0.3 lb) are created as collision + mass
    children of the Go2 trunk link, so they are rigidly bolted to the robot and
    genuinely add 0.136 kg to that link's mass / shift its CoM.
  * TANK (the 4.6 lb concentrator) is its OWN free rigid body under
    ``/World/O2Payload`` -- so it can actually fall off and tumble with real
    physics -- secured to the trunk by a *breakable* fixed joint that models the
    retaining strap / quick-release clip. A hard fall or impact spikes the joint
    reaction past its break threshold, the strap lets go, and the tank drops.
  * The detailed visuals are referenced from the generated ``.usda`` assets;
    collisions use clean analytic boxes so the simulation stays stable.

The function returns an :class:`O2PayloadHandle` describing every prim it made,
which :mod:`o2_payload.isaac_monitor` consumes to watch for a fall and to report
the weight/CoM effect on the robot.

Nothing here is wired into the running sim automatically -- call
``attach_o2_payload(stage, resolve_go2_body_prim_path(stage))`` from wherever the
robot is spawned when you are ready (see README / AGENTS.md).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .spec import SPEC, O2PayloadSpec

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CONCENTRATOR_USDA = os.path.join(_ASSETS_DIR, "o2_concentrator.usda")
RAILS_USDA = os.path.join(_ASSETS_DIR, "o2_rails.usda")

_LOGGER = logging.getLogger("o2_payload.mount")

# Optional structured logger matching the host project's
# ``log_event(logger, level, action, message, **fields)`` signature. Callers may
# inject one so the payload events land in the same JSONL stream as the rest of
# the sim; otherwise we fall back to the module logger.
LogFn = Callable[..., None]


def _default_log(level: int, action: str, message: str, **fields) -> None:
    _LOGGER.log(level, "%s %s", message, fields if fields else "")


@dataclass
class O2PayloadHandle:
    """Everything the monitor needs to track the spawned payload."""

    trunk_prim_path: str
    tank_prim_path: str
    tank_collision_path: str
    rails_prim_path: str
    rail_collision_paths: List[str]
    joint_prim_path: str
    payload_root: str
    # Tank centre expressed in the trunk frame (m). The monitor recomputes the
    # *expected* tank world pose each step as trunk_world * this offset and
    # compares it to where the tank actually is to detect a fall.
    tank_local_offset_m: Tuple[float, float, float]
    spec: O2PayloadSpec = field(default=SPEC)


def _asset_uri(path: str) -> str:
    return path.replace("\\", "/")


def _make_invisible(prim) -> None:
    from pxr import UsdGeom

    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def _box_inertia(mass: float, lx: float, ly: float, lz: float):
    """Solid-box principal inertia (kg.m^2) about the centre."""
    from pxr import Gf

    ix = mass / 12.0 * (ly * ly + lz * lz)
    iy = mass / 12.0 * (lx * lx + lz * lz)
    iz = mass / 12.0 * (lx * lx + ly * ly)
    return Gf.Vec3f(float(ix), float(iy), float(iz))


def _ensure_friction_material(stage, prim_paths, *, static=1.2, dynamic=1.0,
                              restitution=0.0,
                              material_path="/World/PhysicsMaterials/O2PayloadMaterial"):
    """Create (once) and bind a high-friction material so the tank does not slide
    out of the cradle under normal motion."""
    from pxr import UsdPhysics, UsdShade, Sdf

    mat_prim = stage.GetPrimAtPath(material_path)
    if not mat_prim.IsValid():
        material = UsdShade.Material.Define(stage, material_path)
        phys_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        phys_mat.CreateStaticFrictionAttr().Set(float(static))
        phys_mat.CreateDynamicFrictionAttr().Set(float(dynamic))
        phys_mat.CreateRestitutionAttr().Set(float(restitution))
    for p in prim_paths:
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            continue
        col = UsdPhysics.CollisionAPI(prim)
        try:
            col.GetPhysicsMaterialRel().SetTargets([Sdf.Path(material_path)])
        except Exception:
            prim.CreateRelationship("physics:material").SetTargets([Sdf.Path(material_path)])


# ---------------------------------------------------------------------------
def attach_o2_payload(
    stage,
    trunk_prim_path: str,
    *,
    payload_root: str = "/World/O2Payload",
    spec: O2PayloadSpec = SPEC,
    log: Optional[LogFn] = None,
    reference_visuals: bool = True,
) -> Optional[O2PayloadHandle]:
    """Mount the rail cradle + oxygen concentrator on the Go2.

    Parameters
    ----------
    stage              : the live ``Usd.Stage``.
    trunk_prim_path    : the Go2 body/link prim that follows root motion, e.g. the
                         result of ``resolve_go2_body_prim_path(stage)``.
    payload_root       : where the free tank rigid body + joint live.
    spec               : payload spec (defaults to the validated module SPEC).
    log                : optional ``(level, action, message, **fields)`` logger.
    reference_visuals  : reference the generated ``.usda`` meshes for looks. The
                         physics (colliders/mass/joint) is created regardless.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    logf = log or _default_log

    trunk_prim = stage.GetPrimAtPath(trunk_prim_path)
    if not trunk_prim or not trunk_prim.IsValid():
        logf(logging.ERROR, "o2_attach_failed",
             "Trunk prim not found; cannot mount O2 payload", trunk=trunk_prim_path)
        return None

    c = spec.concentrator
    holder_local = spec.holder_center_m
    tank_local = spec.tank_center_m
    # Bounding-box extents in the trunk frame for the chosen mount orientation
    # (flat by default): ext_x = fore-aft, ext_y = lateral, ext_z = vertical
    # height. Flat lays the concentrator on its side so ext_z is the short 3.5 in
    # width -- a much lower CoM than standing it tall.
    ext_x, ext_y, ext_z = spec.mounted_extents_m

    # -----------------------------------------------------------------
    # 1) RAILS -- collision + mass children of the trunk link (bolted on)
    # -----------------------------------------------------------------
    rails_group_path = f"{trunk_prim_path}/o2_rails"
    rails_group = UsdGeom.Xform.Define(stage, rails_group_path)
    _set_local_translate(rails_group, holder_local)

    if reference_visuals and os.path.exists(RAILS_USDA):
        vis = UsdGeom.Xform.Define(stage, f"{rails_group_path}/visual")
        vis.GetPrim().GetReferences().AddReference(_asset_uri(RAILS_USDA))
    elif reference_visuals:
        logf(logging.WARNING, "o2_rails_visual_missing",
             "Rails .usda not found -- run `python -m o2_payload.build_assets`",
             path=RAILS_USDA)

    # Collision approximation: base plate + two side walls (invisible boxes).
    # The footprint hugs the tank's fore-aft (ext_x) and lateral (ext_y) extents;
    # the side walls only need to hug the tank's vertical height (ext_z), so cap
    # their height -- a flat tank gets a low cradle, not tall upright walls.
    rail = spec.rail
    plate_l = ext_x + 2.0 * rail.fore_aft_overhang_m
    plate_w = ext_y + 2.0 * (rail.side_gap_m + rail.rail_thickness_m)
    rail_y = ext_y / 2.0 + rail.side_gap_m + rail.rail_thickness_m / 2.0
    wall_h = min(rail.wall_height_m, 0.7 * ext_z)
    rail_colliders = [
        ("plate", (0.0, 0.0, -rail.base_plate_thickness_m / 2.0),
         (plate_l, plate_w, rail.base_plate_thickness_m)),
        ("wall_l", (0.0, rail_y, wall_h / 2.0),
         (plate_l, rail.rail_thickness_m, wall_h)),
        ("wall_r", (0.0, -rail_y, wall_h / 2.0),
         (plate_l, rail.rail_thickness_m, wall_h)),
    ]
    rail_collision_paths: List[str] = []
    for i, (name, center, size) in enumerate(rail_colliders):
        cpath = f"{rails_group_path}/col_{name}"
        cube = _define_box(stage, cpath, center, size)
        _make_invisible(cube.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        # Put the whole printed-part mass on the base plate collider; because it
        # is a child of the trunk rigid body this genuinely adds to the link.
        if name == "plate":
            mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
            mass_api.CreateMassAttr(float(rail.mass_kg))
        rail_collision_paths.append(cpath)

    # -----------------------------------------------------------------
    # 2) TANK -- free rigid body under payload_root (can fall off)
    # -----------------------------------------------------------------
    UsdGeom.Xform.Define(stage, payload_root)  # container scope
    tank_path = f"{payload_root}/o2_tank"
    tank_xform = UsdGeom.Xform.Define(stage, tank_path)
    tank_prim = tank_xform.GetPrim()

    # Place the tank in the cradle: world pose = trunk_world * tank_local.
    trunk_world = UsdGeom.Xformable(trunk_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_pos = trunk_world.Transform(Gf.Vec3d(*tank_local))
    world_quat = trunk_world.ExtractRotationQuat()  # Gf.Quatd (double precision)
    tank_xform.ClearXformOpOrder()
    tank_xform.AddTranslateOp().Set(world_pos)
    # AddOrientOp() defaults to float precision (expects a Quatf); ExtractRotationQuat
    # returns a Quatd, so author the op as double to match or pxr raises a type error.
    tank_xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(world_quat)

    # Rigid body + mass + inertia (CoM at the tank centre == this prim's origin).
    UsdPhysics.RigidBodyAPI.Apply(tank_prim)
    tank_mass = UsdPhysics.MassAPI.Apply(tank_prim)
    tank_mass.CreateMassAttr(float(c.mass_kg))
    tank_mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    tank_mass.CreateDiagonalInertiaAttr(
        _box_inertia(c.mass_kg, ext_x, ext_y, ext_z)
    )
    tank_mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    if reference_visuals and os.path.exists(CONCENTRATOR_USDA):
        vis = UsdGeom.Xform.Define(stage, f"{tank_path}/visual")
        vis.GetPrim().GetReferences().AddReference(_asset_uri(CONCENTRATOR_USDA))
        # The mesh is authored UPRIGHT (L=+X, W=+Y, 7.2 in H=+Z). Rotate it to match
        # the mounted collider for the chosen orientation:
        #   * flat:      +90 deg about X -> H goes lateral, the 3.5 in W goes up.
        #   * crosswise: +90 deg about Z (yaw) -> the 9.1 in L runs side-to-side,
        #                the 3.5 in W runs fore-aft (stays tall, H up).
        #   * upright:   no rotation.
        if spec.mount.orientation == "flat":
            vis.AddRotateXOp().Set(90.0)
        elif spec.mount.orientation == "crosswise":
            vis.AddRotateZOp().Set(90.0)
    elif reference_visuals:
        logf(logging.WARNING, "o2_tank_visual_missing",
             "Concentrator .usda not found -- run `python -m o2_payload.build_assets`",
             path=CONCENTRATOR_USDA)

    # Clean analytic box collider sized to the shell bounding box in the mounted
    # orientation (flat by default), matching the inertia and the rotated visual.
    tank_col_path = f"{tank_path}/collision"
    tank_cube = _define_box(stage, tank_col_path, (0.0, 0.0, 0.0),
                            (ext_x, ext_y, ext_z))
    _make_invisible(tank_cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(tank_cube.GetPrim())

    # -----------------------------------------------------------------
    # 3) STRAP -- breakable fixed joint trunk <-> tank
    # -----------------------------------------------------------------
    joint_path = f"{payload_root}/strap_joint"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([trunk_prim_path])
    joint.CreateBody1Rel().SetTargets([tank_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*tank_local))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Break force/torque live on the BASE UsdPhysics.Joint schema
    # (physics:breakForce / physics:breakTorque), not on PhysxSchema.PhysxJointAPI
    # -- PhysX honours the core-schema values. (PhysxJointAPI carries armature /
    # joint friction / projection, none of which we need here.) Authoring these on
    # the FixedJoint makes the strap a breakable retaining clip.
    joint.CreateBreakForceAttr().Set(float(spec.strap.break_force_n))
    joint.CreateBreakTorqueAttr().Set(float(spec.strap.break_torque_nm))

    # -----------------------------------------------------------------
    # 4) Friction so the tank grips the cradle under normal motion.
    # -----------------------------------------------------------------
    try:
        _ensure_friction_material(stage, [tank_col_path] + rail_collision_paths)
    except Exception as exc:
        logf(logging.WARNING, "o2_friction_failed",
             "Could not bind payload friction material", error=str(exc))

    handle = O2PayloadHandle(
        trunk_prim_path=trunk_prim_path,
        tank_prim_path=tank_path,
        tank_collision_path=tank_col_path,
        rails_prim_path=rails_group_path,
        rail_collision_paths=rail_collision_paths,
        joint_prim_path=joint_path,
        payload_root=payload_root,
        tank_local_offset_m=tuple(float(v) for v in tank_local),
        spec=spec,
    )

    com_shift = spec.com_shift_m(tank_attached=True)
    logf(
        logging.INFO, "o2_payload_attached",
        "Mounted mock-up P2-E6 oxygen concentrator + rail cradle on the Go2",
        trunk=trunk_prim_path, tank=tank_path, joint=joint_path,
        orientation=spec.mount.orientation,
        tank_extents_mm=[round(ext_x * 1000.0, 1), round(ext_y * 1000.0, 1),
                         round(ext_z * 1000.0, 1)],
        tank_mass_kg=round(c.mass_kg, 3),
        rail_mass_kg=round(spec.rail.mass_kg, 3),
        payload_total_kg=round(spec.total_payload_mass_kg, 3),
        payload_fraction_of_trunk=round(spec.payload_mass_fraction, 3),
        com_shift_mm=[round(v * 1000.0, 1) for v in com_shift],
        static_pitch_torque_nm=round(spec.pitch_torque_nm, 3),
        strap_break_force_n=spec.strap.break_force_n,
        lidar_clearance_mm=round(spec.lidar_clearance_actual_m * 1000.0, 1),
    )
    return handle


def release_o2_tank(stage, handle: O2PayloadHandle, *, log: Optional[LogFn] = None) -> bool:
    """Manually pop the strap (disable the joint) -- handy for testing that the
    monitor reports a fall. Returns True if the joint was disabled."""
    from pxr import UsdPhysics

    logf = log or _default_log
    joint_prim = stage.GetPrimAtPath(handle.joint_prim_path)
    if not joint_prim or not joint_prim.IsValid():
        return False
    UsdPhysics.Joint(joint_prim).CreateJointEnabledAttr().Set(False)
    logf(logging.WARNING, "o2_strap_released",
         "O2 strap joint manually released (test)", joint=handle.joint_prim_path)
    return True


# ---------------------------------------------------------------------------
# Small USD helpers
# ---------------------------------------------------------------------------
def _set_local_translate(xform, translate) -> None:
    from pxr import Gf

    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))


def _define_box(stage, path: str, center, size):
    """An axis-aligned UsdGeom.Cube of unit size, scaled+placed to ``size``."""
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.ClearXformOpOrder()
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3d(float(size[0]), float(size[1]), float(size[2])))
    # Keep the analytic box extent in sync with the unit size.
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    return cube
