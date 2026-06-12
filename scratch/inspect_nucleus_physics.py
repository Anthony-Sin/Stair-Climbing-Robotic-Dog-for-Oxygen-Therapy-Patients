import sys
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({'headless': True})

from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd
import omni.kit.app

try:
    from omni.isaac.core.utils.stage import add_reference_to_stage
except ModuleNotFoundError:
    from isaacsim.core.utils.stage import add_reference_to_stage

try:
    import isaacsim.storage.native as nucleus_utils
except ImportError:
    import omni.isaac.core.utils.nucleus as nucleus_utils

def main():
    nucleus_go2 = nucleus_utils.get_assets_root_path() + '/Isaac/Robots/Unitree/Go2/go2.usd'
    print('Testing:', nucleus_go2)
    
    stage = omni.usd.get_context().get_stage()
    GO2_USD_PATH = '/World/Go2'
    
    add_reference_to_stage(usd_path=nucleus_go2, prim_path=GO2_USD_PATH)
    go2_prim = stage.GetPrimAtPath(GO2_USD_PATH)
    
    vsets = go2_prim.GetVariantSets()
    if 'Physics' in vsets.GetNames():
        vsets.GetVariantSet('Physics').SetVariantSelection('physx')
    if 'Robot' in vsets.GetNames():
        vsets.GetVariantSet('Robot').SetVariantSelection('Robot')
    if 'Sensor' in vsets.GetNames():
        vsets.GetVariantSet('Sensor').SetVariantSelection('Sensors')
    stage.Load(GO2_USD_PATH)
    
    for _ in range(10):
        omni.kit.app.get_app().update()

    print('--- Articulation roots ---')
    for prim in Usd.PrimRange(go2_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f'  ArticulationRoot: {prim.GetPath()} (type={prim.GetTypeName()})')
    
    print('--- RevoluteJoints (first 15) ---')
    joint_count = 0
    for prim in Usd.PrimRange(go2_prim):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            body0 = prim.GetRelationship('physics:body0').GetTargets() if prim.HasRelationship('physics:body0') else []
            body1 = prim.GetRelationship('physics:body1').GetTargets() if prim.HasRelationship('physics:body1') else []
            print(f'  joint: {prim.GetPath().name}  lower={prim.GetAttribute("physics:lowerLimit").Get()}  upper={prim.GetAttribute("physics:upperLimit").Get()}')
            joint_count += 1
            if joint_count >= 15:
                break
    
    print('--- First 5 children of go2_prim ---')
    for child in list(go2_prim.GetChildren())[:5]:
        mass_api = UsdPhysics.MassAPI.Get(stage, child.GetPath())
        rb_api = UsdPhysics.RigidBodyAPI.Get(stage, child.GetPath())
        print(f'  {child.GetPath().name} type={child.GetTypeName()} hasMass={bool(mass_api)} hasRB={bool(rb_api)}')
        xform = UsdGeom.Xformable(child)
        if xform:
            try:
                t = xform.GetLocalTransformation()
                print(f'    localT = {list(t.ExtractTranslation())}')
            except: pass
    
    print('--- Material variants ---')
    for prim in Usd.PrimRange(go2_prim):
        vs = prim.GetVariantSets()
        if vs.GetNames():
            print(f'  {prim.GetPath()}: variants={vs.GetNames()}')
            for vn in vs.GetNames():
                v = vs.GetVariantSet(vn)
                print(f'    {vn}: current={v.GetVariantSelection()}, options={v.GetVariantNames()}')
    
    simulation_app.close()

if __name__ == '__main__':
    main()
