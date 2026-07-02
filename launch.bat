@echo off
REM btop-style launcher for the go2 sim/real stack.
REM   launch.bat            interactive start screen
REM   launch.bat --preview  static UI preview (no GPU)
REM You can still run the underlying command directly: src\sim\run_sim.bat ...
python "%~dp0src\launcher.py" %*
