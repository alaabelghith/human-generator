@echo off
rem Human Generator server (port 8777). Houdini starts it automatically; you can also run it by hand to see its log.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. Run install.ps1 first.
  pause
  exit /b 1
)
set PYTHONHOME=
set PYTHONPATH=
rem Llama 3 (Kimodo's text encoder) on the CPU: ~3 GB of VRAM instead of ~17 GB. Set to cuda on a 24 GB card for faster prompts.
if "%TEXT_ENCODER_DEVICE%"=="" set TEXT_ENCODER_DEVICE=cpu
rem after the first successful start every model is cached: no network calls
if exist "cache\.models_ready" set HF_HUB_OFFLINE=1
".venv\Scripts\python.exe" -u server\hg_server.py %*
pause
