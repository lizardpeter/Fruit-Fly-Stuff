@echo off
setlocal
cd /d "%~dp0"
if exist "engine\target\release\fruit-fly-engine.exe" (
  "engine\target\release\fruit-fly-engine.exe"
  goto :end
)
where cargo >nul 2>&1 || (
  echo Rust was not found. Install rustup from https://rustup.rs or download the prebuilt .exe from GitHub Actions.
  pause
  exit /b 1
)
cargo run --release --manifest-path engine\Cargo.toml
:end
if errorlevel 1 pause
