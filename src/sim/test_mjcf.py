import os
import numpy as np
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
print("Imported USD Path:", usd_path)

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

# Reset the world to initialize physics
world.reset()

# Apply gains
kps = np.zeros(patient_art.num_dof)
kds = np.zeros(patient_art.num_dof)
for idx, name in enumerate(patient_art.dof_names):
    if "femur" in name or "tibia" in name:
        kps[idx] = 400.0
        kds[idx] = 40.0
    elif "foot" in name:
        kps[idx] = 200.0
        kds[idx] = 20.0
    elif "toes" in name:
        kps[idx] = 80.0
        kds[idx] = 10.0
    else:
        kps[idx] = 100.0
        kds[idx] = 10.0

try:
    # Test setting gains as 1D array
    patient_art._articulation_view.set_gains(kps=kps, kds=kds)
    print("set_gains succeeded with 1D array!")
except Exception as e:
    print("set_gains failed with 1D array:", e)
    try:
        # Test setting gains as 2D array (1, num_dof)
        patient_art._articulation_view.set_gains(kps=np.expand_dims(kps, 0), kds=np.expand_dims(kds, 0))
        print("set_gains succeeded with 2D array!")
    except Exception as e2:
        print("set_gains failed with 2D array:", e2)

app.close()
