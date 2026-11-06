import hashlib
import sys
import types

import pytest

from scripts import setup_answer_model as setup


def test_check_only_is_offline_and_checks_bytes(tmp_path, monkeypatch):
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'public weights')
    monkeypatch.setattr(setup, 'DEFAULT_MODEL', model)
    monkeypatch.setattr(setup, 'SHA256', hashlib.sha256(b'public weights').hexdigest())
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    assert setup.setup(check_only=True) == model
    model.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        setup.setup(check_only=True)


def test_download_is_pinned_and_verified(tmp_path, monkeypatch):
    model = tmp_path / 'model.gguf'
    monkeypatch.setattr(setup, 'DEFAULT_MODEL', model)
    monkeypatch.setattr(setup, 'CACHE', tmp_path)
    monkeypatch.setattr(setup, 'SHA256', hashlib.sha256(b'public weights').hexdigest())
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        model.write_bytes(b'public weights')
        return str(model)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=download))
    assert setup.setup() == model
    assert calls == [{'repo_id': setup.MODEL_ID, 'filename': setup.FILENAME,
                      'revision': setup.REVISION, 'cache_dir': str(tmp_path)}]
    assert setup.setup() == model and len(calls) == 1


def test_setup_path_matches_all_answer_launchers():
    expected = str(setup.DEFAULT_MODEL.relative_to(setup.ROOT)).replace('/', '\\')
    for name in ('Start-RewindAI-Local-Answers.cmd', 'Start-RewindAI-Semantic-Answers.cmd',
                 'Start-RewindAI-Visual-Answers.cmd'):
        assert expected in (setup.ROOT / name).read_text(encoding='utf-8')
