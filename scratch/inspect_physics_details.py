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
    local_path = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2_fixed\configuration\go2_description_physics.usd"
    
    stage = Usd.Stage.Open(local_path)
    print("Opened local physics stage:", local_path)
    
    # Iterate over prims and find references/payloads
    count = 0
    type_counts = {}
    for prim in stage.TraverseAll():
        count += 1
        prim_type = prim.GetTypeName()
        type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
        refs = prim.GetMetadata("references")
        payloads = prim.GetMetadata("payloads")
        if refs or payloads:
            print(f"Prim: {prim.GetPath()}")
            if refs:
                print(f"  References: {refs}")
            if payloads:
                print(f"  Payloads: {payloads}")
                
    print("\n--- Summary ---")
    print("Total prims:", count)
    print("Type counts:")
    for t, c in type_counts.items():
        print(f"  {t}: {c}")
                
    simulation_app.close()

if __name__ == "__main__":
    main()
