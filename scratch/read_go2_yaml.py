try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
app = SimulationApp({'headless':True})
import omni.kit.app
for _ in range(5): omni.kit.app.get_app().update()
try:
    import isaacsim.storage.native as n
except:
    import omni.isaac.core.utils.nucleus as n
root = n.get_assets_root_path()
yaml_path = root + '/Isaac/Samples/Policies/go2/physx_env.yaml'
print('YAML path:', yaml_path)

import omni.client
result, version, content = omni.client.read_file(yaml_path)
print('Result:', result)
if result == omni.client.Result.OK:
    print(content.decode('utf-8'))
else:
    print('FAILED to read:', result)
app.close()
