@echo off
setlocal EnableExtensions

call "%~dp0run_final_scene.bat" --sim2real-validation-cam %*
exit /b %ERRORLEVEL%
