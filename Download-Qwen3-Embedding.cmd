@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Set up the project virtual environment first. See README.md.
  pause
  exit /b 1
)
echo Downloads the public 639 MB Qwen3 embedding model; course files are not uploaded.
.venv\Scripts\python.exe scripts\setup_qwen_embedding.py
if errorlevel 1 (
  echo Setup failed. Install huggingface-hub in the virtual environment and see README.md.
  pause
  exit /b 1
)
echo The model is ready. You can now run Start-RewindAI-Semantic.cmd.
pause
endlocal
