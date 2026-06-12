import sys, os
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({'headless': True})

from pxr import Usd, UsdGeom, Gf, UsdPhysics
import omni.usd
import omni.kit.app

try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ModuleNotFoundError:
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage

def main():
    nucleus_go2 = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd'
    
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    GO2_USD_PATH = '/World/Go2'
    
    add_reference_to_stage(usd_path=nucleus_go2, prim_path=GO2_USD_PATH)
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    # Print variant sets
    vsets = go2_prim.GetVariantSets()
    print('Variant sets:', vsets.GetNames())
    for vsname in vsets.GetNames():
        vs = vsets.GetVariantSet(vsname)
        print(f'  {vsname}: current={vs.GetVariantSelection()}, choices={vs.GetVariantNames()}')
    
    # Try selecting variants and loading
    if 'Physics' in vsets.GetNames():
        vsets.GetVariantSet('Physics').SetVariantSelection('physx')
    if 'Robot' in vsets.GetNames():
        vsets.GetVariantSet('Robot').SetVariantSelection('Robot')
    if 'Sensor' in vsets.GetNames():
        vsets.GetVariantSet('Sensor').SetVariantSelection('Sensors')
    
    stage.Load(GO2_USD_PATH)
    
    # Update kit a few times for assets to resolve
    for _ in range(30):
        omni.kit.app.get_app().update()
    
    prim_count = 0
    mesh_count = 0
    type_counts = {}
    for prim in Usd.PrimRange(go2_prim):
        prim_count += 1
        t = prim.GetTypeName()
        type_counts[t] = type_counts.get(t, 0) + 1
        if t == 'Mesh':
            mesh_count += 1
    
    print(f'Total prims: {prim_count}, Mesh prims: {mesh_count}')
    print('Type counts:')
    for t, c in sorted(type_counts.items()):
        print(f'  {t}: {c}')
    
    # Print first 10 children of go2_prim
    print('First children:')
    for child in list(go2_prim.GetChildren())[:10]:
        print(f'  {child.GetPath()} ({child.GetTypeName()})')
    
    simulation_app.close()

if __name__ == '__main__':
    main()
