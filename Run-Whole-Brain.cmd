@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install -e . || goto :fail
".venv\Scripts\python.exe" -m fruit_fly_stuff.whole_brain download || goto :fail
if not exist "data\fafb783\processed\metadata.json" (
  ".venv\Scripts\python.exe" -m fruit_fly_stuff.whole_brain prepare || goto :fail
)
echo.
echo Running full FAFB mathematical connectome. Stop with Ctrl+C.
".venv\Scripts\python.exe" -m fruit_fly_stuff.whole_brain run --forever --checkpoint-every 500 || goto :fail
pause
exit /b 0
:fail
echo An error occurred. See docs\WHOLE_BRAIN.md for manual download help.
pause
exit /b 1
