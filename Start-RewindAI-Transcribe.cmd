@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo RewindAI needs a project virtual environment first. See README.md.
    pause
    exit /b 1
)
echo This tool processes the selected video locally on CPU.
echo It requires the optional ASR dependency and a cached Whisper base model.
echo See docs\LOCAL_TRANSCRIPTION.md for setup and privacy details.
echo.
".venv\Scripts\python.exe" scripts\choose_video.py %*
if errorlevel 1 (
    echo.
    echo Transcription or startup failed. Check the messages above.
    pause
    exit /b 1
)
endlocal
