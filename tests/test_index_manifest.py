import hashlib
import json
from pathlib import Path

import pytest

from video_rag.index_manifest import SIMILARITY, file_sha256, segment_count, validate_manifest


def test_manifest_scans_files_without_loading_them_whole(tmp_path, monkeypatch):
    data = b'a' * (4 * 1024 * 1024 + 17)
    binary = tmp_path / 'index.faiss'
    binary.write_bytes(data)
    segments = tmp_path / 'segments.jsonl'
    segments.write_text('{"segment_id":"a"}\n\n{"segment_id":"b"}\n', encoding='utf-8')
    def reject_whole_file(*args, **kwargs):
        raise AssertionError('Do not load the whole file')
    monkeypatch.setattr(Path, 'read_bytes', reject_whole_file)
    monkeypatch.setattr(Path, 'read_text', reject_whole_file)
    assert file_sha256(binary) == hashlib.sha256(data).hexdigest()
    assert segment_count(segments) == 2


def base_manifest(segments_path, model="clip-model"):
    import hashlib
    return {
        "schema_version": 2,
        "segment_count": 1,
        "segments_sha256": hashlib.sha256(segments_path.read_bytes()).hexdigest(),
        "models": {"text_embedding": "text-model", "clip": model},
        "similarity": SIMILARITY,
        "indexes": {},
    }


def test_rejects_wrong_model_before_loading_indexes(tmp_path):
    segments = tmp_path / "segments.jsonl"
    segments.write_text('{"segment_id":"s1"}\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(base_manifest(segments, model="wrong")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="model mismatch"):
        validate_manifest(
            segments_path=segments,
            index_dir=tmp_path,
            text_model="text-model",
            clip_model="clip-model",
        )


def test_rejects_wrong_segments_sha_before_loading_indexes(tmp_path):
    segments = tmp_path / "segments.jsonl"
    segments.write_text('{"segment_id":"s1"}\n', encoding="utf-8")
    manifest = base_manifest(segments)
    manifest["segments_sha256"] = "bad"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_manifest(
            segments_path=segments,
            index_dir=tmp_path,
            text_model="text-model",
            clip_model="clip-model",
        )


def test_rejects_legacy_manifest(tmp_path):
    segments = tmp_path / "segments.jsonl"
    segments.write_text('{"segment_id":"s1"}\n', encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        validate_manifest(
            segments_path=segments,
            index_dir=tmp_path,
            text_model="text-model",
            clip_model="clip-model",
        )
