from __future__ import annotations

import argparse
from pathlib import Path

from video_rag.config import load_config
from video_rag.index_manifest import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an existing manifest without rebinding indexes.")
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--index-dir", default=Path("artifacts/indexes"), type=Path)
    parser.add_argument("--config", default=Path("config.toml"), type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_manifest(
        segments_path=args.segments,
        index_dir=args.index_dir,
        text_model=config.models.text_embedding,
        clip_model=config.models.clip,
    )
    print("Existing index manifest matches the selected files and configuration. "
          "No manifest was rewritten; this does not prove how the indexes were built.")


if __name__ == "__main__":
    main()
