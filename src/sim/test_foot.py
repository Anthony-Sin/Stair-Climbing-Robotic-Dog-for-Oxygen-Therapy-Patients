import os
from isaacsim import SimulationApp

# Boot SimulationApp
app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.mjcf")

import sys
import types
import isaacsim.asset.importer.mjcf as mjcf_importer

import omni
if not hasattr(omni, "importer"):
    omni.importer = types.ModuleType("omni.importer")
    sys.modules["omni.importer"] = omni.importer
omni.importer.mjcf = mjcf_importer
sys.modules["omni.importer.mjcf"] = mjcf_importer

import omni.usd
from pxr import UsdGeom, Gf

# Create the Stage and the World
try:
    from omni.isaac.core import World
except ImportError:
    from isaacsim.core.api import World

world = World(stage_units_in_meters=1.0)

# Path to local XML
xml_path = os.path.join(os.path.dirname(__file__), "isaac", "assets", "humanoid_CMU_V2020.xml")

# Import MJCF
importer = mjcf_importer.MJCFImporter()
config = mjcf_importer.MJCFImporterConfig()
config.mjcf_path = xml_path
config.fix_base = False
config.allow_self_collision = False
config.link_density = 1062.0

usd_path = importer.import_mjcf(config)

# Reference USD on Stage
try:
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ImportError:
    from isaacsim.core.utils.stage import add_reference_to_stage

prim_path = "/World/PatientHumanoid"
add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

# Apply uniform scale
stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath(prim_path)
scale = 1.70 / 1.78
xform = UsdGeom.Xformable(prim)
xform.ClearXformOpOrder()
xform.AddScaleOp().Set(Gf.Vec3d(scale, scale, scale))

# Add Articulation to World Scene
try:
    from omni.isaac.core.articulations import Articulation
except ImportError:
    from isaacsim.core.prims import SingleArticulation as Articulation

patient_art = Articulation(prim_path=prim_path, name="patient_humanoid")
world.scene.add(patient_art)

world.reset()

# Query the relative position of the foot joint relative to the root body
# Let's get the world transform of the root and the left foot
view = patient_art._articulation_view
poses = view.get_world_poses() # root poses
root_pos = poses[0][0]

# Let's compute the transform of lfoot relative to root using USD
lfoot_prim = stage.GetPrimAtPath("/World/PatientHumanoid/Geometry/root/lhipjoint/lfemur/ltibia/lfoot")
rfoot_prim = stage.GetPrimAtPath("/World/PatientHumanoid/Geometry/root/rhipjoint/rfemur/rtibia/rfoot")

lfoot_xform = UsdGeom.Xformable(lfoot_prim)
rfoot_xform = UsdGeom.Xformable(rfoot_prim)

lfoot_world = lfoot_xform.ComputeLocalToWorldTransform(0.0)
rfoot_world = rfoot_xform.ComputeLocalToWorldTransform(0.0)

lfoot_pos = lfoot_world.ExtractTranslation()
rfoot_pos = rfoot_world.ExtractTranslation()

print(f"ROOT WORLD POS: {root_pos}")
print(f"LFOOT WORLD POS: {list(lfoot_pos)}")
print(f"RFOOT WORLD POS: {list(rfoot_pos)}")
print(f"LFOOT RELATIVE TO ROOT Z: {lfoot_pos[2] - root_pos[2]}")

app.close()
