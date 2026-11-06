from scripts import run_semantic_workspace


def test_missing_model_does_not_start_server(tmp_path, monkeypatch):
    monkeypatch.setenv("REWINDAI_SEMANTIC_MODEL", str(tmp_path))
    calls = []
    monkeypatch.setattr(run_semantic_workspace.subprocess, "call", lambda *a, **kw: calls.append(a) or 0)
    assert run_semantic_workspace.main() == 2
    assert not calls


def test_local_model_is_passed_to_workspace_without_download(tmp_path, monkeypatch):
    model = tmp_path / "embedding.gguf"
    model.touch()
    monkeypatch.setenv("REWINDAI_SEMANTIC_MODEL", str(model))
    calls = []
    monkeypatch.setattr(run_semantic_workspace.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0)
    assert run_semantic_workspace.main() == 0
    assert calls[0][-2:] == ["--semantic-model", str(model.resolve())]
    assert "--allow-model-download" not in calls[0]
