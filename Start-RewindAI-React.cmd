@echo off
setlocal
cd /d "%~dp0"
if not exist "frontend\dist\index.html" (
  echo The React workspace has not been built yet.
  echo From the frontend folder run: npm ci ^&^& npm run build
  echo See README.md for full setup instructions.
  pause
  exit /b 1
)
echo Starting RewindAI at http://127.0.0.1:5000/
call "%~dp0Start-RewindAI-Semantic-Answers.cmd"
