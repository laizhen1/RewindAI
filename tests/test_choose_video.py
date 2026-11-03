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
