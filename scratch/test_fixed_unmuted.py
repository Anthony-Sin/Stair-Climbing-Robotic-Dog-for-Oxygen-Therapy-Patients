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
    fixed_go2_usd = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2_fixed\go2.usd"
    
    stage = omni.usd.get_context().get_stage()
    GO2_USD_PATH = "/World/Go2"
    
    add_reference_to_stage(usd_path=fixed_go2_usd, prim_path=GO2_USD_PATH)
    
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    # Select variants
    vsets = go2_prim.GetVariantSets()
    vsets.GetVariantSet("Physics").SetVariantSelection("physx")
    vsets.GetVariantSet("Sensor").SetVariantSelection("Sensors")
    vsets.GetVariantSet("Robot").SetVariantSelection("Robot")
    
    print("--- Loading Stage ---")
    stage.Load(GO2_USD_PATH)
    
    # Update kit to process stage loads
    for i in range(10):
        omni.kit.app.get_app().update()
        
    print("--- Done ---")
    simulation_app.close()

if __name__ == "__main__":
    main()
