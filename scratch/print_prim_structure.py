import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd

def inspect_layer(usd_path):
    print(f"\n=== Inspecting: {os.path.basename(usd_path)} ===")
    if not os.path.exists(usd_path):
        print("File does not exist.")
        return
        
    stage = Usd.Stage.Open(usd_path)
    print("Default Prim:", stage.GetDefaultPrim())
    print("Top-level prims:")
    for prim in stage.GetPseudoRoot().GetChildren():
        print(f"  - {prim.GetPath()} ({prim.GetTypeName()})")

def main():
    config_dir = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2_fixed\configuration"
    files = [
        "go2_description_base.usd",
        "go2_description_physics.usd",
        "go2_description_sensor.usd",
        "go2_description_robot.usd"
    ]
    for f in files:
        inspect_layer(os.path.join(config_dir, f))
        
    simulation_app.close()

if __name__ == "__main__":
    main()
