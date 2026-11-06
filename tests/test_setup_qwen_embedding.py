import hashlib
import sys
import types

import pytest

from scripts import run_semantic_workspace
from scripts import setup_qwen_embedding as setup


def test_check_only_uses_no_network_and_validates_checksum(tmp_path, monkeypatch):
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'public model')
    monkeypatch.setattr(setup, 'DEFAULT_MODEL', model)
    monkeypatch.setattr(setup, 'SHA256', hashlib.sha256(b'public model').hexdigest())
    monkeypatch.setitem(sys.modules, 'huggingface_hub', None)
    assert setup.setup(check_only=True) == model
    model.write_bytes(b'bad data')
    with pytest.raises(ValueError, match='checksum'):
        setup.setup(check_only=True)


def test_downloads_only_the_pinned_public_model_and_checks_bytes(tmp_path, monkeypatch):
    model = tmp_path / 'model.gguf'
    monkeypatch.setattr(setup, 'DEFAULT_MODEL', model)
    monkeypatch.setattr(setup, 'CACHE', tmp_path)
    monkeypatch.setattr(setup, 'SHA256', hashlib.sha256(b'public model').hexdigest())
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        model.write_bytes(b'public model')
        return str(model)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=download))
    assert setup.setup() == model
    assert calls == [{'repo_id': setup.MODEL_ID, 'filename': setup.FILENAME,
                      'revision': setup.REVISION, 'cache_dir': str(tmp_path)}]
    assert setup.setup() == model and len(calls) == 1


def test_default_download_path_matches_launcher():
    assert setup.DEFAULT_MODEL == run_semantic_workspace.DEFAULT_MODEL
