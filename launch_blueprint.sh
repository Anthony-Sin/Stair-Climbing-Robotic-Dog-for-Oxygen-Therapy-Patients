#!/usr/bin/env bash
# Launcher for the blueprint viewer (Three.js static web app)
# Serves the directory on port 8741 using serve.py and opens the browser.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8741

echo "Starting Blueprint Viewer server on http://localhost:${PORT}..."

# Open default browser based on OS
if command -v xdg-open > /dev/null; then
  (sleep 1; xdg-open "http://localhost:${PORT}") &
elif command -v open > /dev/null; then
  (sleep 1; open "http://localhost:${PORT}") &
fi

python3 "${SCRIPT_DIR}/src/tools/blueprint_viewer/serve.py" ${PORT}
