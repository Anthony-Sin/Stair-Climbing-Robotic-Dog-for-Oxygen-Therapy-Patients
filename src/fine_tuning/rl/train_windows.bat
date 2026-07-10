@echo off
setlocal
REM Native Windows training launcher.
REM Launches the PowerShell runner to execute local RL fine-tuning on Windows.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0train_windows_rl.ps1" %*
exit /b %ERRORLEVEL%
