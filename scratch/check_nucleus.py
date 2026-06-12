import sys
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({'headless': True})
import omni.kit.app
try:
    import isaacsim.storage.native as nucleus_utils
except ImportError:
    import omni.isaac.core.utils.nucleus as nucleus_utils
root = nucleus_utils.get_assets_root_path()
print('Assets root:', root)
paths_to_check = [
    '/Isaac/Robots/Unitree/Go2/go2.usd',
    '/Isaac/Robots/Unitree/Go2/go2.usda',
    '/Isaac/Samples/Mujoco_Menagerie/unitree_go2/go2/go2.usda',
]
for p in paths_to_check:
    full = (root or '') + p
    try:
        found = nucleus_utils.is_file(full)
        print(f'EXISTS({found}): {full}')
    except Exception as e:
        print(f'NOT FOUND: {full} -- {e}')
simulation_app.close()
