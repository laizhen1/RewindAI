@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo RewindAI needs a project virtual environment first.
    echo See README.md for installation instructions.
    pause
    exit /b 1
)
set "PYTHONPATH=%CD%\src"
echo Starting the local CPU demo at http://127.0.0.1:5000
echo This uses example text, not a real video AI model.
echo Open the address in your browser. Close this window to stop the server.
echo.
".venv\Scripts\python.exe" scripts\run_demo.py
if errorlevel 1 (
    echo.
    echo The demo could not start. Check dependencies and whether port 5000 is in use.
    pause
    exit /b 1
)
endlocal

