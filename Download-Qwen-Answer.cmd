@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Set up the project virtual environment first. See README.md.
  pause
  exit /b 1
)
echo Downloads the public 2.1 GB answer model; course files are not uploaded.
.venv\Scripts\python.exe scripts\setup_answer_model.py
if errorlevel 1 (
  echo Setup failed. Install huggingface-hub in the virtual environment and see README.md.
  pause
  exit /b 1
)
echo The answer model is ready. Review the README for how to start the workspace.
pause
endlocal
