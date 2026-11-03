from scripts import rewind_local


def test_existing_subtitles_skip_transcription(tmp_path, monkeypatch):
    video, subtitle = tmp_path / "video.mp4", tmp_path / "video.srt"
    video.write_bytes(b"placeholder")
    subtitle.write_text("placeholder")
    calls = []
    monkeypatch.setattr(rewind_local.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0)
    assert rewind_local.main([str(video), "--subtitles", str(subtitle)]) == 0
    assert len(calls) == 1 and calls[0][1].endswith("run_transcript.py")
    assert "--save-startup-video" in calls[0]


def test_transcription_failure_does_not_start_server(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    calls = []
    monkeypatch.setattr(rewind_local.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 7)
    assert rewind_local.main([str(video), "--language", "en"]) == 7
    assert len(calls) == 1 and calls[0][1].endswith("transcribe_local.py")


def test_generated_subtitles_passed_to_server(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    calls = []
    monkeypatch.setattr(rewind_local.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0)
    assert rewind_local.main([str(video), "--allow-model-download"]) == 0
    assert len(calls) == 2
    assert "--allow-model-download" in calls[0]
    assert calls[0][calls[0].index("--output") + 1] == calls[1][calls[1].index("--subtitles") + 1]
