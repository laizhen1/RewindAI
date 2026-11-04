from scripts import run_visual_workspace


def test_missing_visual_model_does_not_start_server(tmp_path, monkeypatch):
    monkeypatch.setenv("REWINDAI_VISUAL_MODEL_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(run_visual_workspace.subprocess, "call", lambda *a, **kw: calls.append(a) or 0)
    assert run_visual_workspace.main() == 2
    assert not calls


def test_visual_launcher_uses_local_model_without_download(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").touch()
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_quantized.onnx").touch()
    monkeypatch.setenv("REWINDAI_VISUAL_MODEL_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(run_visual_workspace.subprocess, "call", lambda cmd, **kw: calls.append((cmd, kw)) or 0)
    assert run_visual_workspace.main() == 0
    command, options = calls[0]
    assert command[-2:] == ["--visual-model", str(tmp_path.resolve())]
    assert options["cwd"] == run_visual_workspace.ROOT
    assert "--allow-model-download" not in command
