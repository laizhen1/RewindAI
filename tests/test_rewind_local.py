import sqlite3

import pytest

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


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(rewind_local, "WORKSPACE_DATABASE", tmp_path / "library.sqlite3")


def save_video(database, video):
    import hashlib
    stat = video.stat()
    video_id = "video_" + hashlib.sha256(str(video.resolve()).encode("utf-8")).hexdigest()[:16]
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE local_transcript_videos (video_id TEXT, video_path TEXT, video_size INTEGER, video_mtime_ns INTEGER)")
        conn.execute("INSERT INTO local_transcript_videos VALUES (?, ?, ?, ?)",
                     (video_id, str(video.resolve()), stat.st_size, stat.st_mtime_ns))


def test_existing_video_opens_workspace_without_transcription(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    save_video(rewind_local.WORKSPACE_DATABASE, video)
    calls = []
    monkeypatch.setattr(rewind_local.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0)
    assert rewind_local.main([str(video), "--port", "5200"]) == 0
    assert len(calls) == 1 and calls[0][1].endswith("run_transcript.py")
    assert "--video" not in calls[0] and calls[0][-1] == "5200"
    assert rewind_local.main([str(video), "--subtitles", str(tmp_path / "replacement.srt")]) == 1
    assert len(calls) == 1


def test_changed_saved_video_does_not_transcribe_or_start(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    save_video(rewind_local.WORKSPACE_DATABASE, video)
    video.write_bytes(b"modified recording")
    calls = []
    monkeypatch.setattr(rewind_local.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0)
    assert rewind_local.main([str(video)]) == 1
    assert not calls


def test_corrupt_library_stops_before_transcription(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    rewind_local.WORKSPACE_DATABASE.write_bytes(b"not sqlite")
    calls = []
    monkeypatch.setattr(rewind_local.subprocess, "call", lambda cmd, **kw: calls.append(cmd) or 0)
    assert rewind_local.main([str(video)]) == 1
    assert not calls
