@echo off
REM Live colored monitor for the local stair fine-tune (docs\DESIGN.md palette).
REM Watches the rsl_rl logs and graphs Curriculum/terrain_levels so a STALL is obvious.
REM Run it in a second terminal while train_local.bat trains (train_local also auto-opens it).
setlocal
set "WINDIR=%~dp0"
set "WINDIR=%WINDIR:\=/%"
set "DISTRO=%FT_RL_WSL_DISTRO%"
if "%DISTRO%"=="" set "DISTRO=Ubuntu"

for /f "usebackq delims=" %%p in (`wsl -d %DISTRO% wslpath -u "%WINDIR%train_monitor.py"`) do set "MON=%%p"
if "%MON%"=="" (
  echo ERROR: WSL/%DISTRO% not available. Run check_my_computer.bat first.
  pause & exit /b 1
)

REM Run the monitor INSIDE WSL (fast native-fs reads; truecolor shows in Windows Terminal).
REM logdir mirrors train_local_rl.sh: <robot_lab>/logs/rsl_rl/<exptid>. No spaces in these paths.
wsl -d %DISTRO% -e bash -lc "python3 '%MON%' --logdir ${FT_RL_REPO_DIR:-$HOME/robot_lab}/logs/rsl_rl/${FT_RL_EXPTID:-unitree_go2_rough} %*"
