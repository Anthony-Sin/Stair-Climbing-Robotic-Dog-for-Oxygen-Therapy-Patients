@echo off
REM Check whether THIS Windows machine can run the blind-RL stair fine-tune locally.
REM Prefers Python 3.11 (Isaac Sim 5.1's version) but runs on any 3.x.
setlocal
set "HERE=%~dp0"
where py >nul 2>&1 && (
  py -3.11 "%HERE%check_local_machine.py" %* || py "%HERE%check_local_machine.py" %*
) || python "%HERE%check_local_machine.py" %*
if not "%NO_PAUSE%"=="1" pause
