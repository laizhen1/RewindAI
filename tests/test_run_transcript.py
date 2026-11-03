import sys

import pytest
from flask import Flask

from scripts import run_transcript
from video_rag.api import create_app
from video_rag.transcript_baseline import build_transcript_segments


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
