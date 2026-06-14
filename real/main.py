import sys
import os

# Resolve root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add root, core, and real/bot directories to python path
sys.path.extend([
    ROOT_DIR,
    os.path.join(ROOT_DIR, "core"),
    os.path.join(ROOT_DIR, "real", "bot")
])

# Ensure the --sim flag is NOT passed since this is the hardware entrypoint
if "--sim" in sys.argv:
    sys.argv.remove("--sim")

from core.main import main

if __name__ == "__main__":
    main()
