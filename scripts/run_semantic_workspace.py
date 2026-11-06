"""Open the Qwen3 semantic workspace with a local model."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "370f27d7550e0def9b39c1f16d3fbaa13aa67728"
DEFAULT_MODEL = ROOT / "artifacts" / "hf-cache" / "hub" / "models--Qwen--Qwen3-Embedding-0.6B-GGUF" / "snapshots" / REVISION / "Qwen3-Embedding-0.6B-Q8_0.gguf"


def main() -> int:
    model = Path(os.environ.get("REWINDAI_SEMANTIC_MODEL", str(DEFAULT_MODEL))).expanduser().absolute()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        print("Local semantic model files were not found. See README.md for setup.", file=sys.stderr)
        print("Set REWINDAI_SEMANTIC_MODEL to a local Qwen3-Embedding GGUF file.", file=sys.stderr)
        return 2
    print("Starting Qwen3 semantic search locally on CPU; results need review.", flush=True)
    print("No model download is performed. Open http://127.0.0.1:5000/ when ready.", flush=True)
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "run_transcript.py"),
                            "--semantic-model", str(model)], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
