"""Open the experimental English semantic workspace with a local model."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "08a308f628bc9d6774b7922f319eb1b65afa1a82"
DEFAULT_MODEL = ROOT / "artifacts" / "hf-cache" / "hub" / "models--Xenova--all-MiniLM-L6-v2" / "snapshots" / REVISION


def main() -> int:
    model = Path(os.environ.get("REWINDAI_SEMANTIC_MODEL_DIR", str(DEFAULT_MODEL))).expanduser().resolve()
    if not (model / "tokenizer.json").is_file() or not (model / "onnx" / "model_quantized.onnx").is_file():
        print("Local semantic model files were not found. See README.md for setup.", file=sys.stderr)
        print("Set REWINDAI_SEMANTIC_MODEL_DIR to the directory containing tokenizer.json and onnx/.", file=sys.stderr)
        return 2
    print("Starting experimental English semantic search locally on CPU; results need review.", flush=True)
    print("No model download is performed. Open http://127.0.0.1:5000/ when ready.", flush=True)
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "run_transcript.py"),
                            "--semantic-model", str(model)], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
