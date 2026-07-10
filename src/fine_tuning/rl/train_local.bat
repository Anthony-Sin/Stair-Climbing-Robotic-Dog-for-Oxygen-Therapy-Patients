@echo off
REM One command to fine-tune the stair-climb policy on THIS laptop's GPU (via WSL2/Ubuntu).
REM First run does a one-time ~30-60 min stack install; later runs skip straight to training.
REM Override anything with FT_RL_* env vars before calling (e.g. set FT_RL_NUM_ENVS=512).
setlocal
echo == Launching local fine-tune inside WSL2 (Ubuntu) ==

REM %~dp0 ends with a backslash; convert to forward slashes -- backslashes get eaten when
REM cmd hands the arg to wsl.exe, but forward slashes survive and wslpath -u accepts them.
set "WINDIR=%~dp0"
set "WINDIR=%WINDIR:\=/%"
set "DISTRO=%FT_RL_WSL_DISTRO%"
if "%DISTRO%"=="" set "DISTRO=Ubuntu"

for /f "usebackq delims=" %%p in (`wsl -d %DISTRO% wslpath -u "%WINDIR%train_local_rl.sh"`) do set "SH=%%p"
if "%SH%"=="" (
  echo ERROR: WSL/%DISTRO% not available or path translation failed. Run check_my_computer.bat.
  if not "%NO_PAUSE%"=="1" pause
  exit /b 1
)

REM Auto-open the live graph in a second window (set NO_MONITOR=1 to skip). It waits
REM gracefully until training emits data -- normal during the one-time install / before iter 1.
if not "%NO_MONITOR%"=="1" start "O2 Stair Monitor" cmd /k "%~dp0watch_training.bat"

REM The .sh computes its own repo root from its location, so no cd is needed. Login shell
REM (-lc) sources ~/.bashrc where conda init lives. Extra args are forwarded to the .sh.
wsl -d %DISTRO% -e bash -lc "bash '%SH%' %*"
set "RC=%ERRORLEVEL%"
if not "%NO_PAUSE%"=="1" pause
exit /b %RC%
