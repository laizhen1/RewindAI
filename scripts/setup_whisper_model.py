"""Prepare the public faster-whisper base files without using lecture data."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'artifacts' / 'hf-cache' / 'hub'
MODEL = 'Systran/faster-whisper-base'
FILES = ['config.json', 'model.bin', 'tokenizer.json', 'vocabulary.txt']


def setup(*, check_only: bool = False) -> Path:
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('Install huggingface-hub first') from exc
    try:
        directory = Path(snapshot_download(repo_id=MODEL, cache_dir=str(CACHE),
                                           allow_patterns=FILES, local_files_only=check_only))
    except Exception as exc:
        raise RuntimeError('Could not find the public Whisper model in the local cache' if check_only
                           else 'Could not download the public Whisper model') from exc
    if any(not (directory / filename).is_file() for filename in FILES):
        raise ValueError('The cached Whisper model is incomplete')
    return directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Check the cache without network access')
    args = parser.parse_args()
    try:
        setup(check_only=args.check_only)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Whisper setup failed: {exc}', file=sys.stderr)
        return 2
    print('Public Whisper base model is ready in the local project cache.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
