@echo off
setlocal
cd /d "%~dp0"
echo EXPERIMENTAL RETINA - REAL NEURON IDS, UNCALIBRATED RETINAL ALIGNMENT
echo This launcher explicitly selects --approximate.
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install -e . || goto :fail
".venv\Scripts\python.exe" -m fruit_fly_stuff.whole_brain download --annotations --retinotopy || goto :fail
if not exist "data\fafb783\processed\metadata.json" (
  ".venv\Scripts\python.exe" -m fruit_fly_stuff.whole_brain prepare || goto :fail
)
".venv\Scripts\python.exe" -m fruit_fly_stuff.retinotopy --approximate || goto :fail
set "FRUIT_FLY_NEURAL_MAPPING=retinotopic"
call Run-Neural-Fly.cmd full
exit /b %errorlevel%
:fail
echo Consult docs\RETINA.md if preparation fails.
pause
exit /b 1
