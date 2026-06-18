@echo off
setlocal EnableExtensions

call "%~dp0..\..\run_sim.bat" --final-scene %*
exit /b %ERRORLEVEL%
