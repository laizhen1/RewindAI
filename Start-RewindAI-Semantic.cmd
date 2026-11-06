@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo RewindAI needs a project virtual environment first. See README.md.
    pause
    exit /b 1
)
echo Qwen3 semantic search runs locally on CPU.
echo It needs optional semantic dependencies and locally cached model files.
echo Lecture recordings and transcripts should remain private and local.
echo.
".venv\Scripts\python.exe" scripts\run_semantic_workspace.py
if errorlevel 1 (
    echo.
    echo The workspace could not start. Check the messages above and README.md.
    pause
    exit /b 1
)
endlocal
