import sys
import os
import pathlib

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, Sdf

def main():
    assets_dir = pathlib.Path(__file__).parent.parent.parent / "isaac" / "assets" / "go2_fixed"
    fixed_usd_path = str(assets_dir / "go2.usd")
    ascii_usd_path = str(assets_dir / "go2.usda")
    
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
