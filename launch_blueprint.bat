@echo off
REM Launcher for the blueprint viewer (Three.js static web app)
REM Serves the directory on port 8741 using serve.py and opens the browser.

echo Starting Blueprint Viewer server on http://localhost:8741...
start http://localhost:8741
python "%~dp0src\tools\blueprint_viewer\serve.py" 8741
