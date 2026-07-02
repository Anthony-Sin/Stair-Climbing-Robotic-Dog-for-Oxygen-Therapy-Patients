"""MJCF/USD physics-body construction for the simulated patient.

Builds rigid links + revolute joints, imports the CMU humanoid MJCF as a
gravity-off / collision-off near-massless articulation, and drives individual
joints. Split out of ``sim_person_actor``; re-exported by the ``sim_person_actor``
facade. These helpers are self-contained (no dependency back on the actor module).
"""
import logging
import math
from typing import List

try:
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ModuleNotFoundError:
    from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema
from sim_logging_utils import log_event


def create_link(stage, path, mass, col_type=None, col_size=None, col_offset=None):
    prim = stage.DefinePrim(path, "Xform")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))

    if col_type == "capsule":
        r, h = col_size
        cap = UsdGeom.Capsule.Define(stage, f"{path}/collider")
        cap.CreateRadiusAttr().Set(float(r))
        cap.CreateHeightAttr().Set(float(h))
        cap.CreateAxisAttr().Set("Z")
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        if col_offset is not None:
            cap.AddTranslateOp().Set(col_offset)
    elif col_type == "sphere":
        r = col_size
        sph = UsdGeom.Sphere.Define(stage, f"{path}/collider")
        sph.CreateRadiusAttr().Set(float(r))
        UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
        if col_offset is not None:
            sph.AddTranslateOp().Set(col_offset)
    elif col_type == "box":
        size = col_size
        box = UsdGeom.Cube.Define(stage, f"{path}/collider")
        box.CreateSizeAttr().Set(1.0)
        box.AddScaleOp().Set(Gf.Vec3d(float(size[0]), float(size[1]), float(size[2])))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        if col_offset is not None:
            box.AddTranslateOp().Set(col_offset)

    return prim

def create_revolute_joint(stage, path, parent_path, child_path, parent_pos, child_pos, axis="Y"):
    joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(path))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(parent_pos[0], parent_pos[1], parent_pos[2]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(child_pos[0], child_pos[1], child_pos[2]))
    joint.CreateAxisAttr().Set(axis)

    # Enable joint drive
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateStiffnessAttr().Set(600.0)
    drive.CreateDampingAttr().Set(40.0)
    drive.CreateMaxForceAttr().Set(1500.0)
    drive.CreateTargetPositionAttr().Set(0.0)
    return joint

def build_patient_physics(stage, x, y, start_z=0.8742):
    import urllib.request
    import os

    # Path to assets folder
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    xml_path = os.path.join(assets_dir, "humanoid_CMU_V2020.xml").replace("\\", "/")

    if not os.path.exists(xml_path):
        url = "https://raw.githubusercontent.com/google-deepmind/dm_control/main/dm_control/locomotion/walkers/assets/humanoid_CMU_V2020.xml"
        try:
            urllib.request.urlretrieve(url, xml_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download humanoid_CMU_V2020.xml from {url}: {e}")

    # Import MJCF
    try:
        import isaacsim.asset.importer.mjcf as mjcf_importer
    except ModuleNotFoundError:
        import omni.importer.mjcf as mjcf_importer

    importer = mjcf_importer.MJCFImporter()
    config = mjcf_importer.MJCFImporterConfig()
    config.mjcf_path = xml_path
    config.fix_base = False
    config.allow_self_collision = False
    # Near-massless body (very low density). The patient is animated kinematically
    # (velocity-servoed dynamic root + PD joints); a normal ~75 kg mass makes the limb
    # PD reaction torques and the root velocity drive generate large forces that diverge
    # the floating articulation. With tiny link masses, no part generates significant
    # force, so the body stays stable while still animating. (Mass realism is moot for a
    # gravity-off, collision-off visual patient; the Phase-6 mass gate was removed.)
    config.link_density = 12.0

    usd_path = importer.import_mjcf(config)

    root_path = "/World/PersonPhysics"
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(Sdf.Path(root_path))

    add_reference_to_stage(usd_path=usd_path, prim_path=root_path)

    prim = stage.GetPrimAtPath(root_path)
    scale = 1.70 / 1.78
    xform = UsdGeom.Xformable(prim)

    # Scale, rotate upright, and position
    scale_op = None
    rotate_op = None
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateX:
            rotate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op

    if scale_op is None:
        scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3d(scale, scale, scale))

    if rotate_op is None:
        rotate_op = xform.AddRotateXOp()
    rotate_op.Set(90.0)

    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(x), float(y), float(start_z)))

    # Apply a PhysX material to all contact bodies
    material_path = f"{root_path}/ContactMaterial"
    if not stage.GetPrimAtPath(material_path).IsValid():
        material_prim = stage.DefinePrim(material_path, "Material")
        physx_material = UsdPhysics.MaterialAPI.Apply(material_prim)
        physx_material.CreateStaticFrictionAttr().Set(1.0)
        physx_material.CreateDynamicFrictionAttr().Set(0.9)
        physx_material.CreateRestitutionAttr().Set(0.0)
    else:
        material_prim = stage.GetPrimAtPath(material_path)

    n_collision_off = 0
    n_gravity_off = 0
    n_mass_fixed = 0
    fixed_mass_paths: List[str] = []
    for child in Usd.PrimRange(prim):
        # DISABLE collision on the patient. It is fully driven (root velocity-servoed,
        # joints PD-tracked to the gait, foot placement from the gait + terrain-tracking
        # pelvis Z). Foot/leg-vs-step contact only injects impulses that diverge PhysX
        # (observed: a foot driven into a stair tread blew the body to 28 m mid-climb).
        # Grounding is judged from computed foot-vs-terrain height, not contact.
        if child.HasAPI(UsdPhysics.CollisionAPI) or child.IsA(UsdGeom.Capsule) or child.IsA(UsdGeom.Sphere) or child.IsA(UsdGeom.Mesh):
            try:
                col = UsdPhysics.CollisionAPI.Apply(child)
                col.CreateCollisionEnabledAttr().Set(False)
                n_collision_off += 1
            except Exception:
                pass
        # Disable gravity on EVERY rigid body (the MJCF importer may expose links via
        # either the UsdPhysics OR the PhysxSchema rigid-body API, so check both -- a
        # one-sided check silently left gravity ON and the collision-free body fell
        # through the floor). Without gravity the gain-light PD limbs don't sag and the
        # velocity-servoed root holds height.
        if child.HasAPI(UsdPhysics.RigidBodyAPI) or child.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            try:
                rb = PhysxSchema.PhysxRigidBodyAPI.Apply(child)
                rb.CreateDisableGravityAttr().Set(True)
                n_gravity_off += 1
            except Exception:
                pass
            # Sanitize degenerate link mass/inertia. The CMU MJCF hand bodies
            # (lhand/rhand) import with a NEGATIVE mass and an invalid inertia
            # tensor {1,1,1} (degenerate collision geom), which seeds a NaN in the
            # PD-driven floating articulation; after some steps PhysX invalidates
            # the simulation view and the whole Kit app shuts down (run_sim "exit
            # 137"). Clamp ONLY non-positive / non-finite authored values to a small
            # valid mass; healthy density-computed links are read as unauthored here
            # and left exactly as imported, so the gait animation is unchanged.
            try:
                m_api = UsdPhysics.MassAPI.Apply(child)
                m_attr = m_api.GetMassAttr()
                m_val = m_attr.Get() if (m_attr and m_attr.HasAuthoredValue()) else None
                bad_mass = m_val is not None and (not math.isfinite(m_val) or m_val <= 0.0)
                i_attr = m_api.GetDiagonalInertiaAttr()
                i_val = i_attr.Get() if (i_attr and i_attr.HasAuthoredValue()) else None
                bad_inertia = i_val is not None and any(
                    (not math.isfinite(c)) or c <= 0.0 for c in (i_val[0], i_val[1], i_val[2])
                )
                # The CMU hand bodies (lhand/rhand) import with a degenerate collision
                # geom whose mass resolves NEGATIVE at solve time -- there is no authored
                # mass attr to read (it is density/geom-derived), so the checks above miss
                # them. Target them by name as well and author explicit, valid mass props
                # (mass + COM + diagonal inertia + principal axes) so PhysX never computes
                # the bad values. Only these tiny end-effectors are touched; the gait is
                # unchanged.
                degenerate_by_name = "hand" in child.GetName().lower()
                if bad_mass or bad_inertia or degenerate_by_name:
                    m_api.CreateMassAttr().Set(0.2)
                    m_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    m_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.01, 0.01, 0.01))
                    m_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    n_mass_fixed += 1
                    fixed_mass_paths.append(str(child.GetPath()))
            except Exception:
                pass
        if child.IsA(UsdPhysics.Joint) or "Joint" in child.GetTypeName():
            try:
                drive_api = UsdPhysics.DriveAPI.Get(child, "angular")
                if not drive_api.IsValid():
                    drive_api = UsdPhysics.DriveAPI.Apply(child, "angular")
                drive_api.CreateTypeAttr().Set("force")
                drive_api.CreateTargetPositionAttr().Set(0.0)
            except Exception:
                pass

    pelvis_prim = stage.GetPrimAtPath(f"{root_path}/Geometry/root")

    # NOTE: a KINEMATIC articulation root is NOT allowed by PhysX ("ArticulationRootAPI
    # on a kinematic rigid body is not allowed" -> articulation disabled). So the root
    # stays DYNAMIC and is driven by velocity. To keep that velocity-servoed root stable
    # against the limb PD reaction torques, the WHOLE body is made near-massless (low
    # import density): with tiny link masses neither the root velocity drive nor the
    # joint PD generates large forces, so nothing diverges, while the legs still animate.
    try:
        log_event(
            logging.getLogger("sim.isaac_env"),
            logging.INFO,
            "patient_physics_body_setup",
            "Patient MJCF bodies configured (collision disabled, gravity disabled, light body)",
            collision_disabled_prims=int(n_collision_off),
            gravity_disabled_bodies=int(n_gravity_off),
            mass_corrected_bodies=int(n_mass_fixed),
            mass_corrected_paths=fixed_mass_paths,
        )
    except Exception:
        pass
    return pelvis_prim

def _drive_joint(stage, joint_path, angle_rad):
    joint_prim = stage.GetPrimAtPath(joint_path)
    if joint_prim.IsValid():
        drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if drive_api.IsValid():
            drive_api.GetTargetPositionAttr().Set(math.degrees(float(angle_rad)))
