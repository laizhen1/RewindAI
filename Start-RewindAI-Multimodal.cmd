@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment missing. See docs\REACT_WORKSPACE.md.
  pause
  exit /b 1
)
if not exist "frontend\dist\index.html" (
  echo React workspace not built. Run npm ci and npm run build in frontend.
  pause
  exit /b 1
)
if not defined REWINDAI_ANSWER_MODEL set "REWINDAI_ANSWER_MODEL=artifacts\hf-cache\hub\models--Qwen--Qwen2.5-3B-Instruct-GGUF\snapshots\7dabda4d13d513e3e842b20f0d435c732f172cbe\qwen2.5-3b-instruct-q4_k_m.gguf"
if not defined REWINDAI_SEMANTIC_MODEL set "REWINDAI_SEMANTIC_MODEL=artifacts\hf-cache\hub\models--Qwen--Qwen3-Embedding-0.6B-GGUF\snapshots\370f27d7550e0def9b39c1f16d3fbaa13aa67728\Qwen3-Embedding-0.6B-Q8_0.gguf"
if not defined REWINDAI_VISUAL_MODEL_DIR set "REWINDAI_VISUAL_MODEL_DIR=artifacts\hf-cache\hub\models--Xenova--clip-vit-base-patch32\snapshots\d15189d7028b43f1d3e65039190477f6af591c2a"
if not exist "%REWINDAI_ANSWER_MODEL%" (
  echo Answer model missing. Run python scripts\setup_answer_model.py.
  pause
  exit /b 1
)
if not exist "%REWINDAI_SEMANTIC_MODEL%" (
  echo Embedding model missing. Run python scripts\setup_qwen_embedding.py.
  pause
  exit /b 1
)
if not exist "%REWINDAI_VISUAL_MODEL_DIR%\tokenizer.json" (
  echo CLIP model missing. Run python scripts\setup_visual_model.py.
  pause
  exit /b 1
)
if not exist "%REWINDAI_VISUAL_MODEL_DIR%\onnx\model_quantized.onnx" (
  echo CLIP model incomplete. Run python scripts\setup_visual_model.py.
  pause
  exit /b 1
)
echo Starting local CPU workspace with experimental frame search.
echo This launcher does not download model files. Verify results against the video.
.venv\Scripts\python.exe scripts\run_transcript.py --semantic-model "%REWINDAI_SEMANTIC_MODEL%" --answer-model "%REWINDAI_ANSWER_MODEL%" --visual-model "%REWINDAI_VISUAL_MODEL_DIR%"
if errorlevel 1 (
  echo The workspace could not start. Check the messages above and the setup guide.
  pause
  exit /b 1
)
endlocal
