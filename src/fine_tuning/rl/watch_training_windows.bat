@echo off
setlocal
REM Live colored monitor for the native Windows stair fine-tune.
set "REPO_ROOT=%~dp0..\..\.."
set "VENV=%FT_RL_VENV_DIR%"
if "%VENV%"=="" set "VENV=%USERPROFILE%\.venv_rl"
set "MONITOR=%~dp0train_monitor.py"

REM Resolve robot_lab directory (defaulting to USERPROFILE\robot_lab)
set "ROBOT_LAB=%FT_RL_REPO_DIR%"
if "%ROBOT_LAB%"=="" set "ROBOT_LAB=%USERPROFILE%\robot_lab"
set "EXPTID=%FT_RL_EXPTID%"
if "%EXPTID%"=="" set "EXPTID=unitree_go2_rough"

echo == Starting Native Windows Train Monitor ==
echo Log Directory: %ROBOT_LAB%\logs\rsl_rl\%EXPTID%

if not exist "%VENV%" (
    echo ERROR: Virtual environment not found at %VENV%. Run setup_windows_rl.ps1 first.
    pause & exit /b 1
)

call "%VENV%\Scripts\activate.bat"
python "%MONITOR%" --logdir "%ROBOT_LAB%\logs\rsl_rl\%EXPTID%" %*
pause
