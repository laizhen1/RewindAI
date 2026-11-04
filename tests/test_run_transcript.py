import sys

import pytest
from flask import Flask

from scripts import run_transcript
from video_rag.api import create_app
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


def test_startup_video_is_saved_for_restart_and_duplicate_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"video")
    subtitles = tmp_path / "lesson.srt"
    subtitles.write_text("1\n00:00:01,000 --> 00:00:04,000\nOriginal topic.\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_transcript.py", "--video", str(video), "--subtitles", str(subtitles),
                                      "--save-startup-video"])
    monkeypatch.setattr(Flask, "run", lambda self, **kwargs: None)
    run_transcript.main()
    client = create_app(build_transcript_segments([]), mode="transcript").test_client()
    assert len(client.get("/api/library").get_json()["videos"]) == 1
    assert not client.post("/api/ask", json={"question": "original"}).get_json()["abstained"]
    subtitles.write_text("1\n00:00:01,000 --> 00:00:04,000\nReplacement topic.\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_transcript.main()
    reopened = create_app(build_transcript_segments([]), mode="transcript").test_client()
    assert not reopened.post("/api/ask", json={"question": "original"}).get_json()["abstained"]
    assert reopened.post("/api/ask", json={"question": "replacement"}).get_json()["abstained"]


def test_save_flag_requires_one_video(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_transcript.py", "--save-startup-video"])
    with pytest.raises(SystemExit):
        run_transcript.main()


def test_changed_video_rejects_new_bookmark_but_preserves_existing_note(tmp_path):
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"original")
    subtitles = tmp_path / "lesson.srt"
    subtitles.write_text("1\n00:00:01,000 --> 00:00:04,000\nRelevant topic.\n", encoding="utf-8")
    segments = segments_from_subtitles(video, subtitles)
    app = create_app(build_transcript_segments(segments), database_path=tmp_path / "library.sqlite3", mode="transcript")
    client = app.test_client()
    payload = {"video_id": segments[0].video_id, "segment_id": segments[0].segment_id, "note": "keep"}
    assert client.post("/api/bookmarks", json=payload).status_code == 201
    video.write_bytes(b"a different video")
    assert client.post("/api/bookmarks", json=payload).status_code == 409
    saved = client.get("/api/bookmarks").get_json()["bookmarks"]
    assert len(saved) == 1 and saved[0]["note"] == "keep" and not saved[0]["available"]
