@echo off
setlocal
cd /d "%~dp0"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=toy"
if /I not "%MODE%"=="toy" if /I not "%MODE%"=="full" (
  echo Usage: Run-Neural-Fly.cmd [toy ^| full]
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install -e . || goto :fail
if /I "%MODE%"=="full" (
  ".venv\Scripts\python.exe" -m fruit_fly_stuff.whole_brain download --annotations || goto :fail
  if not exist "data\fafb783\processed\metadata.json" (
    echo Preparing full FlyWire graph. This may require multiple GB of RAM.
    ".venv\Scripts\python.exe" -m fruit_fly_stuff.whole_brain download || goto :fail
    ".venv\Scripts\python.exe" -m fruit_fly_stuff.whole_brain prepare || goto :fail
  )
)
set "FRUIT_FLY_BRAIN_MODE=%MODE%"
set "FRUIT_FLY_ROOT=%CD%"
set "FRUIT_FLY_PYTHON=%CD%\.venv\Scripts\python.exe"
echo Rust 3D engine with the %MODE% brain controller.
echo Current input/output neuron mappings are artificial; this is not biological fly behavior.
if exist "engine\target\release\fruit-fly-engine.exe" (
  "engine\target\release\fruit-fly-engine.exe"
  goto :end
)
if exist "fruit-fly-engine.exe" (
  "fruit-fly-engine.exe"
  goto :end
)
where cargo >nul 2>&1
if not errorlevel 1 (
  cargo run --release --manifest-path engine\Cargo.toml
  goto :end
)
echo Download the fruit-fly-engine.exe artifact from Rust Actions, extract it
echo into this repository folder, and run this script again.
goto :fail
:end
if errorlevel 1 goto :fail
exit /b 0
:fail
echo Error: check Python 3.11+, Rust binary and console output.
pause
exit /b 1
