"""Explicitly download or check the public local-answer model."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct-GGUF'
REVISION = '7dabda4d13d513e3e842b20f0d435c732f172cbe'
FILENAME = 'qwen2.5-3b-instruct-q4_k_m.gguf'
SHA256 = '626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d'
CACHE = ROOT / 'artifacts' / 'hf-cache' / 'hub'
DEFAULT_MODEL = CACHE / 'models--Qwen--Qwen2.5-3B-Instruct-GGUF' / 'snapshots' / REVISION / FILENAME


def verify_model(path: Path) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == SHA256


def setup(*, check_only: bool = False) -> Path:
    if verify_model(DEFAULT_MODEL):
        return DEFAULT_MODEL
    if check_only:
        raise ValueError('The pinned local answer model is missing or has the wrong checksum')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError('Install the download dependency: python -m pip install huggingface-hub') from exc
    try:
        path = Path(hf_hub_download(repo_id=MODEL_ID, filename=FILENAME, revision=REVISION, cache_dir=str(CACHE)))
    except Exception as exc:
        raise RuntimeError('Could not download the public model; check the network and cache') from exc
    if not verify_model(path) or not verify_model(DEFAULT_MODEL):
        raise ValueError('Model checksum verification failed; inspect your local cache and retry')
    return DEFAULT_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify the local cache without network access')
    args = parser.parse_args()
    if not args.check_only:
        print('Preparing the public 2.1 GB answer model. No course files are uploaded.', flush=True)
    try:
        path = setup(check_only=args.check_only)
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        print(f'Model setup failed: {exc}', file=sys.stderr)
        return 2
    print('Pinned answer model verified:', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
