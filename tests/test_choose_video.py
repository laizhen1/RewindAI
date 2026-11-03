from scripts import choose_video


def test_supplied_path_is_passed_as_one_argument_without_download(tmp_path, monkeypatch):
    video = tmp_path / "lecture with spaces.mp4"
    calls = []
    monkeypatch.setattr(choose_video.subprocess, "call", lambda command, **kwargs: calls.append((command, kwargs)) or 0)
    assert choose_video.main([str(video)]) == 0
    command, kwargs = calls[0]
    assert command[-1] == str(video)
    assert command[1].endswith("rewind_local.py")
    assert "--allow-model-download" not in command
    assert kwargs["cwd"] == choose_video.ROOT


def test_rejects_multiple_videos():
    assert choose_video.main(["one.mp4", "two.mp4"]) == 2


def test_project_cache_is_used_unless_user_selects_a_cache(tmp_path, monkeypatch):
    reference = tmp_path / "artifacts/hf-cache/hub/models--Systran--faster-whisper-base/refs/main"
    reference.parent.mkdir(parents=True)
    reference.write_text("revision")
    monkeypatch.setattr(choose_video, "ROOT", tmp_path)
    for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(choose_video.subprocess, "call", lambda command, **kwargs: calls.append(kwargs) or 0)
    choose_video.main(["lecture.mp4"])
    assert calls[-1]["env"]["HF_HOME"] == str(tmp_path / "artifacts/hf-cache")
    monkeypatch.setenv("HF_HUB_CACHE", "chosen-cache")
    choose_video.main(["lecture.mp4"])
    assert "HF_HOME" not in calls[-1]["env"]
    assert calls[-1]["env"]["HF_HUB_CACHE"] == "chosen-cache"
