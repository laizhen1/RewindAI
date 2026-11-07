import sys
from types import SimpleNamespace

import pytest

from scripts import setup_whisper_model as module


def test_offline_check_requires_complete_cached_model(tmp_path, monkeypatch):
    model = tmp_path / 'snapshot'
    model.mkdir()
    for filename in module.FILES:
        (model / filename).write_bytes(b'synthetic')
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(model)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=snapshot_download))
    assert module.setup(check_only=True) == model
    assert calls[0]['local_files_only'] is True and calls[0]['repo_id'] == module.MODEL
    (model / 'model.bin').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        module.setup(check_only=True)
