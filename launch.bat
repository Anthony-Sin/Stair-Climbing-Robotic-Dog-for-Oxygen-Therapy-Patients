@echo off
REM Preset launcher for the go2 sim/real stack (Claude-Code styled).
REM   launch.bat            interactive menu: arrow to a preset, Enter runs, e edits the flags
REM   launch.bat --preview  static UI preview (no GPU/TTY)
REM   launch.bat --dashboard launches into the live telemetry dashboard
REM You can still run the underlying command directly: src\sim\run_sim.bat ...
python "%~dp0src\launcher.py" %*
