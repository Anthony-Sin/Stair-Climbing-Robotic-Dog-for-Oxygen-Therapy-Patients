@echo off
setlocal EnableDelayedExpansion

REM run_stair_sweep.bat - stair-height sweep launcher
REM Usage: .\run_stair_sweep.bat [options]
REM   --heights "0.1 0.125 0.15 0.178 0.198"  Riser heights in metres (space-separated)
REM   --waypoint-x 6.77                         Target X for waypoint test
REM   --climb-backend blind_rl                  Climb backend (default: blind_rl)
REM   --keep-run-logs 9999                      How many prior run logs to keep
REM   --cold                                    Reboot Isaac per height (default: warm)
REM   --keep-warm                               Leave warm Isaac alive after sweep
REM   --episode-timeout-sec 1200               Per-episode wall-clock cap in seconds
REM   --windowed                                Show Isaac window (default: headless)
REM   --no-presentation                         Skip slide-pack generation
REM   --montage-seconds 20                      Target clip length for montage

set "HEIGHTS="
set "WAYPOINT_X="
set "BACKEND=blind_rl"
set "KEEP_RUN_LOGS="
set "COLD="
set "KEEP_WARM="
set "TIMEOUT=1200"
set "WINDOWED="
set "NO_PRES="
set "MONTAGE=20"

:parse
if "%~1"=="" goto run
if /i "%~1"=="--heights"              ( set "HEIGHTS=%~2"              & shift & shift & goto parse )
if /i "%~1"=="--waypoint-x"           ( set "WAYPOINT_X=%~2"           & shift & shift & goto parse )
if /i "%~1"=="--climb-backend"        ( set "BACKEND=%~2"              & shift & shift & goto parse )
if /i "%~1"=="--keep-run-logs"        ( set "KEEP_RUN_LOGS=%~2"        & shift & shift & goto parse )
if /i "%~1"=="--episode-timeout-sec"  ( set "TIMEOUT=%~2"              & shift & shift & goto parse )
if /i "%~1"=="--montage-seconds"      ( set "MONTAGE=%~2"              & shift & shift & goto parse )
if /i "%~1"=="--cold"                 ( set "COLD=-Cold"               & shift & goto parse )
if /i "%~1"=="--keep-warm"            ( set "KEEP_WARM=-KeepWarm"      & shift & goto parse )
if /i "%~1"=="--windowed"             ( set "WINDOWED=-Windowed"       & shift & goto parse )
if /i "%~1"=="--no-presentation"      ( set "NO_PRES=-NoPresentation"  & shift & goto parse )
echo Unknown argument: %~1
goto :eof

:run
set "SCRIPT_DIR=%~dp0"
set "SWEEPSCRIPT=%SCRIPT_DIR%run_stair_sweep.ps1"

if not exist "%SWEEPSCRIPT%" (
    echo ERROR: Missing sweep launcher: "%SWEEPSCRIPT%"
    if not "%NO_PAUSE%"=="1" pause
    exit /b 1
)

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
if defined WAYPOINT_X    set "PS_CMD=%PS_CMD% -WaypointX %WAYPOINT_X%"
if defined KEEP_RUN_LOGS set "PS_CMD=%PS_CMD% -KeepRunLogs %KEEP_RUN_LOGS%"
if defined COLD          set "PS_CMD=%PS_CMD% %COLD%"
if defined KEEP_WARM     set "PS_CMD=%PS_CMD% %KEEP_WARM%"
if defined WINDOWED      set "PS_CMD=%PS_CMD% %WINDOWED%"
if defined NO_PRES       set "PS_CMD=%PS_CMD% %NO_PRES%"

echo.
echo run_stair_sweep : launching stair-height sweep
echo.

powershell.exe -ExecutionPolicy Bypass -NoProfile -Command "%PS_CMD%"
set "SWEEP_EXIT=%ERRORLEVEL%"

echo.
if not "%SWEEP_EXIT%"=="0" (
    echo run_stair_sweep failed with exit code %SWEEP_EXIT%.
) else (
    echo run_stair_sweep finished.
)
if not "%NO_PAUSE%"=="1" pause
exit /b %SWEEP_EXIT%
