@echo off
REM Export svgs\diagram_*.html to transparent-background PNGs (svgs\png\*.png).
REM Usage:  export_pngs.bat            (2x)
REM         export_pngs.bat -Scale 3   (crisper)
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0export_pngs.ps1" %*
echo.
pause
