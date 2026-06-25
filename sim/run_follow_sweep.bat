@echo off
setlocal EnableDelayedExpansion

REM run_follow_sweep.bat - person-follow step-height sweep
REM Runs the full person-follow pipeline (Docker YOLO controller + Isaac Sim) across
REM realistic riser heights and produces a slide-ready presentation pack.
REM
REM Usage: .\run_follow_sweep.bat [options]
REM   --heights "0.1 0.125 0.15 0.178 0.198"  Riser heights in metres
REM   --climb-backend blind_rl                  Climb backend (default: blind_rl)
REM   --cold                                    Reboot Isaac per height (default: warm/boot-once)
REM   --keep-warm                               Leave warm Isaac alive after sweep
REM   --timeout 600                             Per-episode timeout seconds (default: 600)
REM   --windowed                                Show Isaac window (default: headless)
REM   --no-presentation                         Skip slide-pack generation
REM   --montage-seconds 20                      Target clip length for montage

set "HEIGHTS="
set "BACKEND=blind_rl"
set "COLD="
set "KEEP_WARM="
set "TIMEOUT=600"
set "WINDOWED="
set "NO_PRES="
set "MONTAGE=20"

:parse
if "%~1"=="" goto run
if /i "%~1"=="--heights"             ( set "HEIGHTS=%~2"          & shift & shift & goto parse )
if /i "%~1"=="--climb-backend"       ( set "BACKEND=%~2"          & shift & shift & goto parse )
if /i "%~1"=="--cold"                ( set "COLD=-Cold"           & shift & goto parse )
if /i "%~1"=="--keep-warm"           ( set "KEEP_WARM=-KeepWarm"  & shift & goto parse )
if /i "%~1"=="--timeout"             ( set "TIMEOUT=%~2"          & shift & shift & goto parse )
if /i "%~1"=="--windowed"            ( set "WINDOWED=-Windowed"   & shift & goto parse )
if /i "%~1"=="--no-presentation"     ( set "NO_PRES=-NoPresentation" & shift & goto parse )
if /i "%~1"=="--montage-seconds"     ( set "MONTAGE=%~2"          & shift & shift & goto parse )
echo Unknown argument: %~1
goto :eof

:run
set "SCRIPT_DIR=%~dp0"
set "SWEEPSCRIPT=%SCRIPT_DIR%run_follow_sweep.ps1"

set "PS_CMD=& '%SWEEPSCRIPT%'"
set "PS_CMD=%PS_CMD% -ClimbBackend '%BACKEND%'"
set "PS_CMD=%PS_CMD% -EpisodeTimeoutSec %TIMEOUT%"
set "PS_CMD=%PS_CMD% -MontageSeconds %MONTAGE%"
if defined HEIGHTS (
    REM Convert space-separated heights to PowerShell array literal @(0.1, 0.125, ...)
    set "HPS=@("
    set "FIRST=1"
    for %%H in (%HEIGHTS%) do (
        if "!FIRST!"=="1" ( set "HPS=!HPS!%%H" & set "FIRST=0" ) else ( set "HPS=!HPS!, %%H" )
    )
    set "HPS=!HPS!)"
    set "PS_CMD=%PS_CMD% -Heights !HPS!"
)
if defined COLD      set "PS_CMD=%PS_CMD% %COLD%"
if defined KEEP_WARM set "PS_CMD=%PS_CMD% %KEEP_WARM%"
if defined WINDOWED  set "PS_CMD=%PS_CMD% %WINDOWED%"
if defined NO_PRES   set "PS_CMD=%PS_CMD% %NO_PRES%"

echo.
echo run_follow_sweep : launching person-follow stair sweep
echo.

powershell.exe -ExecutionPolicy Bypass -NoProfile -Command "%PS_CMD%"
exit /b %ERRORLEVEL%
