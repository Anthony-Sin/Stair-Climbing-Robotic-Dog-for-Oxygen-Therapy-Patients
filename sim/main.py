import sys
import os

# Resolve root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add root, core, and sim/bot directories to python path
sys.path.extend([
    ROOT_DIR,
    os.path.join(ROOT_DIR, "core"),
    os.path.join(ROOT_DIR, "sim", "bot")
])

# Ensure the --sim flag is passed since this is the simulation entrypoint
if "--sim" not in sys.argv:
    sys.argv.append("--sim")

from core.main import main

if __name__ == "__main__":
    main()
