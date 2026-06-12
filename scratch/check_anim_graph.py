import omni
from isaacsim import SimulationApp

# Start the simulation app
sim_app = SimulationApp({"headless": True})

import omni.kit.commands
from isaacsim.core.utils.extensions import enable_extension

# Let's enable the animation extensions first to make sure their commands are registered
enable_extension("omni.anim.graph.core")
enable_extension("omni.anim.graph.bundle")

print("--- Registered Commands ---")
cmds = omni.kit.commands.get_commands()
for cmd_name in sorted(cmds.keys()):
    if "anim" in cmd_name.lower() or "graph" in cmd_name.lower():
        print(cmd_name)

sim_app.close()
