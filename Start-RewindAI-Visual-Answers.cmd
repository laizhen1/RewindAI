@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo RewindAI needs a project virtual environment first. See README.md.
  pause
  exit /b 1
)
if not defined REWINDAI_ANSWER_MODEL set "REWINDAI_ANSWER_MODEL=artifacts\hf-cache\hub\models--Qwen--Qwen2.5-3B-Instruct-GGUF\snapshots\7dabda4d13d513e3e842b20f0d435c732f172cbe\qwen2.5-3b-instruct-q4_k_m.gguf"
if not defined REWINDAI_SEMANTIC_MODEL set "REWINDAI_SEMANTIC_MODEL=artifacts\hf-cache\hub\models--Qwen--Qwen3-Embedding-0.6B-GGUF\snapshots\370f27d7550e0def9b39c1f16d3fbaa13aa67728\Qwen3-Embedding-0.6B-Q8_0.gguf"
if not defined REWINDAI_VISUAL_MODEL_DIR set "REWINDAI_VISUAL_MODEL_DIR=artifacts\hf-cache\hub\models--Xenova--clip-vit-base-patch32\snapshots\d15189d7028b43f1d3e65039190477f6af591c2a"
if not exist "%REWINDAI_ANSWER_MODEL%" goto missing
if not exist "%REWINDAI_SEMANTIC_MODEL%" goto missing
if not exist "%REWINDAI_VISUAL_MODEL_DIR%\tokenizer.json" goto missing
if not exist "%REWINDAI_VISUAL_MODEL_DIR%\onnx\model_quantized.onnx" goto missing
echo Experimental local CPU models. Visual fusion returns candidates, not generated answers.
echo First visual search may take time. Verify all results against the recording.
set "REWINDAI_VISUAL_OPTION="
if "%REWINDAI_VISUAL_COVER_GAPS%"=="1" set "REWINDAI_VISUAL_OPTION=--visual-cover-gaps"
.venv\Scripts\python.exe scripts\run_transcript.py --semantic-model "%REWINDAI_SEMANTIC_MODEL%" --visual-model "%REWINDAI_VISUAL_MODEL_DIR%" --answer-model "%REWINDAI_ANSWER_MODEL%" %REWINDAI_VISUAL_OPTION%
if errorlevel 1 (
  echo Workspace failed to start. Check dependencies and README.md.
  pause
  exit /b 1
)
endlocal
exit /b 0
:missing
echo One or more local model files are missing. Check the README and model path environment variables.
pause
exit /b 1
