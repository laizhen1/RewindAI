import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import refresh_index_manifest
from video_rag import index_manifest


def test_refresh_verifies_without_rebinding_changed_inputs(tmp_path, monkeypatch, capsys):
    segments = tmp_path / 'segments.jsonl'
    segments.write_text('{"segment_id":"a"}\n', encoding='utf-8')
    spec = {'dimension': 4, 'count': 1, 'index_sha256': 'index', 'metadata_sha256': 'metadata'}
    manifest = {'schema_version': 2, 'segment_count': 1,
                'segments_sha256': hashlib.sha256(segments.read_bytes()).hexdigest(),
                'models': {'text_embedding': 'text', 'clip': 'clip'},
                'similarity': index_manifest.SIMILARITY,
                'indexes': {'text_dense': spec, 'vision_dense': spec}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    original = path.read_bytes()
    monkeypatch.setattr(index_manifest, 'index_spec', lambda *args: spec)
    monkeypatch.setattr(refresh_index_manifest, 'load_config', lambda path: SimpleNamespace(
        models=SimpleNamespace(text_embedding='text', clip='clip')))
    monkeypatch.setattr(sys, 'argv', ['refresh', '--segments', str(segments), '--index-dir', str(tmp_path)])
    refresh_index_manifest.main()
    assert 'No manifest was rewritten' in capsys.readouterr().out
    assert path.read_bytes() == original
    monkeypatch.setattr(index_manifest, 'index_spec', lambda *args: {**spec, 'index_sha256': 'changed'})
    with pytest.raises(ValueError, match='index_sha256'):
        refresh_index_manifest.main()
    assert path.read_bytes() == original
    monkeypatch.setattr(index_manifest, 'index_spec', lambda *args: spec)
    segments.write_text('{"segment_id":"changed"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256'):
        refresh_index_manifest.main()
    assert path.read_bytes() == original


def test_refresh_will_not_create_a_manifest_for_unbound_indexes(tmp_path, monkeypatch):
    segments = tmp_path / 'segments.jsonl'
    segments.write_text('{"segment_id":"a"}\n', encoding='utf-8')
    monkeypatch.setattr(refresh_index_manifest, 'load_config', lambda path: SimpleNamespace(
        models=SimpleNamespace(text_embedding='text', clip='clip')))
    monkeypatch.setattr(sys, 'argv', ['refresh', '--segments', str(segments), '--index-dir', str(tmp_path)])
    with pytest.raises(ValueError, match='rebuild indexes'):
        refresh_index_manifest.main()
    assert not (tmp_path / 'manifest.json').exists()
