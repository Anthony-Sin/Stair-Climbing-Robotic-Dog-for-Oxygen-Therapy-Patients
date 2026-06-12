import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, Sdf

def main():
    fixed_usd_path = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2_fixed\go2.usd"
    ascii_usd_path = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2_fixed\go2.usda"
    
    if not os.path.exists(fixed_usd_path):
        print(f"Error: {fixed_usd_path} does not exist.")
        simulation_app.close()
        return
        
    layer = Sdf.Layer.FindOrOpen(fixed_usd_path)
    if not layer:
        print(f"Failed to open layer {fixed_usd_path}")
        simulation_app.close()
        return
        
    layer.Export(ascii_usd_path)
    print(f"Exported to {ascii_usd_path}")
    simulation_app.close()

if __name__ == "__main__":
    main()
