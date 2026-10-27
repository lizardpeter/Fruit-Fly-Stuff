@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install -e ".[plots]" || goto :fail
".venv\Scripts\python.exe" -m fruit_fly_stuff.biophysical --plot || goto :fail
echo.
echo Created results\biophysical\hh_neurons.png and hh_neurons.csv
pause
exit /b 0
:fail
echo An error occurred. Check that Python 3.11+ is installed.
pause
exit /b 1
