@echo off
cd /d "%~dp0"
if not defined REWINDAI_ANSWER_MODEL set "REWINDAI_ANSWER_MODEL=artifacts\hf-cache\hub\models--Qwen--Qwen2.5-3B-Instruct-GGUF\snapshots\7dabda4d13d513e3e842b20f0d435c732f172cbe\qwen2.5-3b-instruct-q4_k_m.gguf"
if not exist "%REWINDAI_ANSWER_MODEL%" (
  echo Local GGUF model not found. Set REWINDAI_ANSWER_MODEL or see README.
  pause
  exit /b 1
)
.venv\Scripts\python.exe scripts\run_transcript.py --answer-model "%REWINDAI_ANSWER_MODEL%"
pause
