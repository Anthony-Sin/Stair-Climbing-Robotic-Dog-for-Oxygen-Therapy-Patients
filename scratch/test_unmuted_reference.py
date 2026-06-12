import sys
import os
import carb

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

# Unmute USD diagnostics
carb.settings.get_settings().set("/persistent/app/usd/muteUsdDiagnostics", False)

from pxr import Usd, UsdGeom
import omni.usd
import omni.kit.app

try:
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ModuleNotFoundError:
    from isaacsim.core.utils.stage import add_reference_to_stage

def main():
    base_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/configuration/go2_description_base.usd"
    go2_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd"
    
    stage = omni.usd.get_context().get_stage()
    
    print("--- Adding Sibling References ---")
    visuals_prim = stage.OverridePrim("/visuals")
    visuals_prim.GetReferences().AddReference(assetPath=base_usd, primPath="/visuals")
    
    meshes_prim = stage.OverridePrim("/meshes")
    meshes_prim.GetReferences().AddReference(assetPath=base_usd, primPath="/meshes")
    
    GO2_USD_PATH = "/World/Go2"
    add_reference_to_stage(usd_path=go2_usd, prim_path=GO2_USD_PATH)
    
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    # Select variants
    vsets = go2_prim.GetVariantSets()
    vsets.GetVariantSet("Physics").SetVariantSelection("None")
    vsets.GetVariantSet("Sensor").SetVariantSelection("Sensors")
    vsets.GetVariantSet("Robot").SetVariantSelection("Robot")
    
    # Load the stage
    print("--- Loading Stage ---")
    stage.Load(GO2_USD_PATH)
    stage.Load("/visuals")
    stage.Load("/meshes")
    
    # Update kit
    print("--- Updating Kit ---")
    for _ in range(10):
        omni.kit.app.get_app().update()
        
    print("--- Done ---")
    simulation_app.close()

if __name__ == "__main__":
    main()
