@echo off
REM ===========================================================================
REM  run_living_room.bat -- the proven stair-climbing sim, restaged in a
REM  furnished LIVING ROOM.
REM
REM  Spawns collidable household furniture (sofa, coffee table, bookshelf,
REM  armchair, TV unit, ...) on the flat approach and makes the patient walk a
REM  realistic WINDING route that weaves AROUND the furniture before rejoining
REM  the stair centreline and climbing -- instead of a straight line / simple
REM  left-right zigzag. Same robot / patient / stair-climb stack as run_sim.bat.
REM
REM  It just sets SIM_LIVING_ROOM=1 (read by isaac_args --living-room) and hands
REM  off to run_sim.bat, so every run_sim flag still works, e.g.:
REM      run_living_room.bat                 (windowed, records video)
REM      run_living_room.bat --headless      (faster, no viewport)
REM ===========================================================================
setlocal EnableExtensions

set "SIM_LIVING_ROOM=1"
call "%~dp0run_sim.bat" %*
exit /b %ERRORLEVEL%
