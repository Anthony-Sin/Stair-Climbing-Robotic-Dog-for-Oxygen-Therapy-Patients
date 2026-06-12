@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0"
set "LAUNCHER=%REPO_ROOT%run_sim.ps1"
set "PS_ARGS="

if not exist "%LAUNCHER%" (
    echo ERROR: Missing launcher script: "%LAUNCHER%"
    if not "%NO_PAUSE%"=="1" pause
    exit /b 1
)

:parse_args
if "%~1"=="" goto run_launcher
set "ARG=%~1"
if /I "%ARG%"=="--skip-build" set "ARG=-SkipBuild"
if /I "%ARG%"=="--force-build" set "ARG=-ForceBuild"
if /I "%ARG%"=="--dry-run" set "ARG=-DryRun"
if /I "%ARG%"=="--no-pause-after-isaac" set "ARG=-NoPauseAfterIsaac"
if /I "%ARG%"=="--pause-after-isaac" set "ARG=-PauseAfterIsaac"
if /I "%ARG%"=="--no-isaac" set "ARG=-NoIsaac"
if /I "%ARG%"=="--no-docker-run" set "ARG=-NoDockerRun"
if /I "%ARG%"=="--no-isaac-ready-wait" set "ARG=-NoIsaacReadyWait"
if /I "%ARG%"=="--isaac-ready-timeout-sec" set "ARG=-IsaacReadyTimeoutSec"
if /I "%ARG%"=="--keep-run-logs" set "ARG=-KeepRunLogs"
if /I "%ARG%"=="--vision-preview" set "ARG=-VisionPreview"
if /I "%ARG%"=="--isaacsim-dir" set "ARG=-IsaacSimDir"
if /I "%ARG%"=="--image" set "ARG=-Image"
if /I "%ARG%"=="--frame-host" set "ARG=-FrameHost"
if /I "%ARG%"=="--cmd-host" set "ARG=-CmdHost"
if /I "%ARG%"=="--cmd-port" set "ARG=-CmdPort"
if /I "%ARG%"=="--frame-port" set "ARG=-FramePort"
if /I "%ARG%"=="--follow-backend" set "ARG=-FollowBackend"
if /I "%ARG%"=="--trt-engine" set "ARG=-TrtEngine"
if /I "%ARG%"=="--osnet-trt-engine" set "ARG=-OsnetTrtEngine"
if /I "%ARG%"=="--sim-frame-timeout-exit-sec" set "ARG=-SimFrameTimeoutExitSec"
if /I "%ARG%"=="--no-model-preflight" set "ARG=-NoModelPreflight"
if /I "%ARG%"=="--max-run-time-sec" set "ARG=-MaxRunTimeSec"
set "PS_ARGS=%PS_ARGS% "%ARG%""
shift
goto parse_args

:run_launcher
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" %PS_ARGS%
set "RUN_SIM_EXIT=%ERRORLEVEL%"

echo.
if not "%RUN_SIM_EXIT%"=="0" (
    echo run_sim failed with exit code %RUN_SIM_EXIT%.
) else (
    echo run_sim finished.
)
if not "%NO_PAUSE%"=="1" pause
exit /b %RUN_SIM_EXIT%
