import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd

def main():
    base_usd = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2_fixed\configuration\go2_description_base.usd"
    if not os.path.exists(base_usd):
        print("File not found:", base_usd)
        simulation_app.close()
        return
        
    stage = Usd.Stage.Open(base_usd)
    print("Opened base stage:", base_usd)
    
    visuals_prim = stage.GetPrimAtPath("/visuals")
    if not visuals_prim.IsValid():
        print("/visuals is not valid!")
        simulation_app.close()
        return
        
    print("Children of /visuals:")
    for child in visuals_prim.GetChildren():
        print(f"  - {child.GetPath()} ({child.GetTypeName()})")
        # Let's inspect /visuals/base or children
        grand_children = list(child.GetChildren())
        print(f"    Total children: {len(grand_children)}")
        for gc in grand_children[:5]:
            print(f"      * {gc.GetPath()} ({gc.GetTypeName()})")
            
    simulation_app.close()

if __name__ == "__main__":
    main()
