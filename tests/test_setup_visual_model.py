import sys
from types import SimpleNamespace

import pytest

from scripts import setup_visual_model as module


def test_visual_setup_uses_pinned_public_files_and_offline_check(tmp_path, monkeypatch):
    model = tmp_path / 'snapshot'
    for filename in module.FILES:
        path = model / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic')
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(model)

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=snapshot_download))
    assert module.setup(check_only=True) == model
    assert calls[0]['local_files_only'] is True
    assert calls[0]['revision'] == module.REVISION
    assert calls[0]['allow_patterns'] == module.FILES
    (model / 'onnx/model_quantized.onnx').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        module.setup(check_only=True)
