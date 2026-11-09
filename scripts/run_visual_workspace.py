"""Open the experimental visual-search workspace using local model files."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"
DEFAULT_MODEL = ROOT / "artifacts" / "hf-cache" / "hub" / "models--Xenova--clip-vit-base-patch32" / "snapshots" / REVISION


def main() -> int:
    model = Path(os.environ.get("REWINDAI_VISUAL_MODEL_DIR", str(DEFAULT_MODEL))).expanduser().resolve()
    if not (model / "tokenizer.json").is_file() or not (model / "onnx" / "model_quantized.onnx").is_file():
        print("Local CLIP model files were not found. See docs/LOCAL_VISUAL_EXPERIMENT.md.", file=sys.stderr)
        print("Set REWINDAI_VISUAL_MODEL_DIR to the directory containing tokenizer.json and onnx/.", file=sys.stderr)
        return 2
    print("Starting experimental English visual search locally on CPU; review every candidate.", flush=True)
    print("No model download is performed. Open http://127.0.0.1:5000/ when ready.", flush=True)
    command = [sys.executable, str(ROOT / "scripts" / "run_transcript.py"), "--visual-model", str(model)]
    if os.environ.get("REWINDAI_VISUAL_COVER_GAPS") == "1":
        command.append("--visual-cover-gaps")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
