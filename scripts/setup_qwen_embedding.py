"""Explicitly download or check the public Qwen3 embedding model."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'Qwen/Qwen3-Embedding-0.6B-GGUF'
REVISION = '370f27d7550e0def9b39c1f16d3fbaa13aa67728'
FILENAME = 'Qwen3-Embedding-0.6B-Q8_0.gguf'
SHA256 = '06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439'
CACHE = ROOT / 'artifacts' / 'hf-cache' / 'hub'
DEFAULT_MODEL = CACHE / 'models--Qwen--Qwen3-Embedding-0.6B-GGUF' / 'snapshots' / REVISION / FILENAME


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
        raise ValueError('The pinned local Qwen3 model is missing or has the wrong checksum')
    # This action downloads public model weights only. Model inference never calls it.
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
        print('Preparing the public 639 MB Qwen3 embedding model. No course files are uploaded.', flush=True)
    try:
        path = setup(check_only=args.check_only)
    except (OSError, ImportError, RuntimeError, ValueError) as exc:
        print(f'Model setup failed: {exc}', file=sys.stderr)
        return 2
    print('Pinned Qwen3 model verified:', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
