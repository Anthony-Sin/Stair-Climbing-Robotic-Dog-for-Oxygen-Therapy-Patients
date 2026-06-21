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

# Reference USD on Stage
try:
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ImportError:
    from isaacsim.core.utils.stage import add_reference_to_stage

prim_path = "/World/PatientHumanoid"
add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

# Add Articulation to World Scene
try:
    from omni.isaac.core.articulations import Articulation
except ImportError:
    from isaacsim.core.prims import SingleArticulation as Articulation

patient_art = Articulation(prim_path=prim_path, name="patient_humanoid")
world.scene.add(patient_art)

world.reset()

print("DOF NAMES:")
for idx, name in enumerate(patient_art.dof_names):
    print(f"  {idx}: {name}")

app.close()
