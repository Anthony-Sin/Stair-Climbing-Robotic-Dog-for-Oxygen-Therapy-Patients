import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom
import omni.usd

def main():
    local_path = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2_fixed\configuration\go2_description_base.usd"
    
    stage = Usd.Stage.Open(local_path)
    print("Opened local base stage:", local_path)
    
    # Iterate over prims and find references/payloads
    for prim in stage.TraverseAll():
        refs = prim.GetMetadata("references")
        payloads = prim.GetMetadata("payloads")
        if refs or payloads:
            print(f"Prim: {prim.GetPath()}")
            if refs:
                print(f"  References: {refs}")
            if payloads:
                print(f"  Payloads: {payloads}")
                
    simulation_app.close()

if __name__ == "__main__":
    main()
