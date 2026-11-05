"""Provenance checks for a declared frozen, locally reviewed answer test."""
from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path

from video_rag.evaluation.frozen_temporal import ROOT, runtime_versions
from video_rag.evaluation.provenance import fingerprint

CODE_FILES = (
    'scripts/freeze_answer_test.py', 'scripts/validate_frozen_labels.py', 'scripts/compare_temporal_multimodal.py',
    'scripts/start_answer_review.py', 'scripts/evaluate_local_answers.py',
    'scripts/prepare_temporal_questions.py', 'scripts/review_answers.py', 'scripts/score_answer_review.py',
    'src/video_rag/api/templates/answer_review.html', 'src/video_rag/adapters/local_answer.py',
    'src/video_rag/adapters/local.py', 'src/video_rag/transcript_baseline.py', 'src/video_rag/pipeline.py',
    'src/video_rag/schemas.py', 'src/video_rag/retrieval/in_memory.py', 'src/video_rag/retrieval/fusion.py',
    'src/video_rag/retrieval/onnx_semantic.py', 'src/video_rag/evaluation/frozen_answers.py',
    'src/video_rag/evaluation/frozen_temporal.py', 'src/video_rag/evaluation/provenance.py',
    'src/video_rag/evaluation/private_output.py',
)


def input_paths(video, subtitles, labels, model, semantic_model=None):
    labels = Path(labels)
    if labels.suffix.lower() != '.csv':
        raise ValueError('A frozen answer test requires a video-bound labeler CSV')
    paths = {'video': Path(video), 'subtitles': Path(subtitles), 'annotations': labels,
             'annotation_binding': labels.with_suffix(labels.suffix + '.source.json'), 'model': Path(model)}
    if semantic_model is not None:
        paths.update({'semantic_tokenizer': Path(semantic_model) / 'tokenizer.json',
                      'semantic_model': Path(semantic_model) / 'onnx' / 'model_quantized.onnx'})
    return paths


def config(video_id, semantic_model, semantic_threshold):
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError('A test video ID is required')
    if semantic_model is not None and (not math.isfinite(semantic_threshold) or not -1 <= semantic_threshold <= 1):
        raise ValueError('Semantic threshold must be finite and in [-1, 1]')
    return {'video_id': video_id, 'split': 'test',
            'transcript_retrieval': 'semantic_experimental' if semantic_model is not None else 'keyword',
            'semantic_threshold': semantic_threshold if semantic_model is not None else None,
            'prompt_version': 'explicit-evidence-v2', 'generator_device': 'cpu'}


def answer_runtime_versions() -> dict:
    versions = runtime_versions()
    try:
        versions['llama-cpp-python'] = importlib.metadata.version('llama-cpp-python')
    except importlib.metadata.PackageNotFoundError:
        versions['llama-cpp-python'] = None
    return versions


def snapshot(paths: dict[str, Path], settings: dict) -> dict:
    return {'input_sha256': {key: fingerprint(path) for key, path in paths.items()},
            'code_sha256': {path: fingerprint(ROOT / path) for path in CODE_FILES},
            'runtime_versions': answer_runtime_versions(), 'settings': settings}


def read_and_verify(manifest_path: Path, paths: dict[str, Path], settings: dict) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, dict) or manifest.get('status') != 'frozen_answer_test_declared_not_independently_verified':
        raise ValueError('Expected a declared frozen answer-test manifest')
    if any(manifest.get(key) != value for key, value in snapshot(paths, settings).items()):
        raise ValueError('Frozen answer-test inputs, configuration, code, or runtime differ from the manifest')
    return manifest


def verify_review_manifest(report: dict, manifest_path: Path) -> None:
    hashes = report.get('input_sha256')
    if not isinstance(hashes, dict) or hashes.get('frozen_manifest') != fingerprint(manifest_path):
        raise ValueError('Review does not match the frozen answer-test manifest')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    if not isinstance(manifest, dict) or manifest.get('status') != 'frozen_answer_test_declared_not_independently_verified':
        raise ValueError('Expected a declared frozen answer-test manifest')
    if manifest.get('input_sha256') != {key: value for key, value in hashes.items() if key != 'frozen_manifest'}:
        raise ValueError('Review input fingerprints do not match the frozen manifest')
    settings, retrieval, generation = manifest.get('settings'), report.get('retrieval_config'), report.get('generation_config')
    if (not isinstance(settings, dict) or not isinstance(retrieval, dict) or not isinstance(generation, dict) or
            retrieval.get('type') != settings.get('transcript_retrieval') or
            retrieval.get('threshold') != settings.get('semantic_threshold') or
            generation.get('prompt_version') != settings.get('prompt_version')):
        raise ValueError('Review configuration does not match the frozen manifest')
    if (manifest.get('code_sha256') != {path: fingerprint(ROOT / path) for path in CODE_FILES} or
            manifest.get('runtime_versions') != answer_runtime_versions()):
        raise ValueError('Code or runtime changed since the answer test was frozen')
