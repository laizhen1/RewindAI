@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo RewindAI needs a project virtual environment first. See README.md.
  pause
  exit /b 1
)
if not defined REWINDAI_ANSWER_MODEL set "REWINDAI_ANSWER_MODEL=artifacts\hf-cache\hub\models--Qwen--Qwen2.5-3B-Instruct-GGUF\snapshots\7dabda4d13d513e3e842b20f0d435c732f172cbe\qwen2.5-3b-instruct-q4_k_m.gguf"
if not defined REWINDAI_SEMANTIC_MODEL_DIR set "REWINDAI_SEMANTIC_MODEL_DIR=artifacts\hf-cache\hub\models--Xenova--all-MiniLM-L6-v2\snapshots\08a308f628bc9d6774b7922f319eb1b65afa1a82"
if not exist "%REWINDAI_ANSWER_MODEL%" (
  echo Local answer model missing. Set REWINDAI_ANSWER_MODEL or see README.md.
  pause
  exit /b 1
)
if not exist "%REWINDAI_SEMANTIC_MODEL_DIR%\tokenizer.json" (
  echo Local semantic model missing. Set REWINDAI_SEMANTIC_MODEL_DIR or see README.md.
  pause
  exit /b 1
)
if not exist "%REWINDAI_SEMANTIC_MODEL_DIR%\onnx\model_quantized.onnx" (
  echo Local semantic model weights missing. See README.md.
  pause
  exit /b 1
)
echo Experimental CPU-only semantic search and local answers. Verify all answers against the video.
echo Processing and startup can take time; course material stays local.
.venv\Scripts\python.exe scripts\run_transcript.py --semantic-model "%REWINDAI_SEMANTIC_MODEL_DIR%" --answer-model "%REWINDAI_ANSWER_MODEL%"
if errorlevel 1 (
  echo The workspace could not start. Check the messages above and README.md.
  pause
  exit /b 1
)
endlocal
