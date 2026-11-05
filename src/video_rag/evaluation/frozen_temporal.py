"""Local provenance and configuration checks for a declared frozen temporal test."""
from __future__ import annotations

import importlib.metadata
import json
import math
import platform
from pathlib import Path

from video_rag.evaluation.provenance import fingerprint

ROOT = Path(__file__).resolve().parents[3]
CODE_FILES = (
    'scripts/compare_temporal_multimodal.py', 'scripts/diagnose_transcript.py', 'scripts/freeze_temporal_test.py',
    'scripts/validate_frozen_labels.py',
    'scripts/prepare_temporal_questions.py', 'src/video_rag/evaluation/frozen_temporal.py',
    'src/video_rag/evaluation/provenance.py', 'src/video_rag/evaluation/private_output.py',
    'src/video_rag/api/visual_sampling.py', 'src/video_rag/api/frames.py',
    'src/video_rag/transcript_baseline.py', 'src/video_rag/pipeline.py',
    'src/video_rag/schemas.py', 'src/video_rag/retrieval/fusion.py',
    'src/video_rag/retrieval/in_memory.py', 'src/video_rag/retrieval/base.py',
    'src/video_rag/adapters/local.py',
    'src/video_rag/retrieval/onnx_clip.py', 'src/video_rag/retrieval/onnx_semantic.py',
)


def input_paths(video, subtitles, questions, visual_model, semantic_model=None):
    if Path(questions).suffix.lower() != '.csv':
        raise ValueError('The frozen test requires a video-bound labeler CSV')
    paths = {'video': Path(video), 'subtitles': Path(subtitles), 'questions': Path(questions),
             'annotation_binding': Path(questions).with_suffix(Path(questions).suffix + '.source.json'),
             'visual_tokenizer': Path(visual_model) / 'tokenizer.json',
             'visual_model': Path(visual_model) / 'onnx' / 'model_quantized.onnx'}
    if semantic_model is not None:
        paths.update({'semantic_tokenizer': Path(semantic_model) / 'tokenizer.json',
                      'semantic_model': Path(semantic_model) / 'onnx' / 'model_quantized.onnx'})
    return paths


def config(video_id, semantic_model, semantic_threshold, visual_min_score, min_coverage):
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError('A test video ID is required')
    if not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError('Coverage threshold must be in (0, 1]')
    if semantic_model is not None and (not math.isfinite(semantic_threshold) or not -1 <= semantic_threshold <= 1):
        raise ValueError('Semantic threshold must be finite and in [-1, 1]')
    if visual_min_score is not None and (not math.isfinite(visual_min_score) or not -1 <= visual_min_score <= 1):
        raise ValueError('Visual threshold must be finite and in [-1, 1]')
    return {'video_id': video_id, 'split': 'test',
            'transcript_retrieval': 'semantic_experimental' if semantic_model is not None else 'keyword',
            'semantic_threshold': semantic_threshold if semantic_model is not None else None,
            'visual_min_score': visual_min_score, 'min_coverage': min_coverage,
            'rrf_k': 60, 'top_k': 3, 'visual_max_segments': 100}


def runtime_versions():
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'onnxruntime', 'tokenizers', 'av', 'faiss-cpu', 'flask'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def snapshot(paths: dict[str, Path], settings: dict) -> dict:
    return {'input_sha256': {key: fingerprint(path) for key, path in paths.items()},
            'code_sha256': {path: fingerprint(ROOT / path) for path in CODE_FILES},
            'runtime_versions': runtime_versions(), 'settings': settings}


def read_and_verify(manifest_path: Path, paths: dict[str, Path], settings: dict) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, dict) or manifest.get('status') != 'frozen_test_declared_not_independently_verified':
        raise ValueError('Expected a declared frozen-test manifest')
    current = snapshot(paths, settings)
    if any(manifest.get(key) != value for key, value in current.items()):
        raise ValueError('Frozen test inputs, configuration, code, or runtime differ from the manifest')
    return manifest
