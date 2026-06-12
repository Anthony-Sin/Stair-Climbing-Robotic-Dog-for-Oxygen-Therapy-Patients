import sys
import os

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, Sdf

def main():
    local_path = r"c:\Users\antho\Downloads\Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients\isaac\assets\go2_fixed\configuration\go2_description_base.usd"
    layer = Sdf.Layer.FindOrOpen(local_path)
    
    prim_spec = layer.GetPrimAtPath("/go2_description/base/visuals")
    if prim_spec:
        ref_list = prim_spec.referenceList
        prepended = list(ref_list.prependedItems)
        for ref in prepended:
            print(f"Ref: {ref}")
            print(f"  type(ref): {type(ref)}")
            print(f"  ref.assetPath: {repr(ref.assetPath)}")
            print(f"  ref.primPath: {repr(ref.primPath)}")
            print(f"  ref.primPath.pathString: {repr(ref.primPath.pathString)}")
            
    simulation_app.close()

if __name__ == "__main__":
    main()
