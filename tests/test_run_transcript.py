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


def test_search_discards_results_if_source_changes_during_query(tmp_path, monkeypatch):
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"original")
    subtitles = tmp_path / "lesson.srt"
    subtitles.write_text("1\n00:00:01,000 --> 00:00:04,000\nRelevant topic.\n", encoding="utf-8")
    segments = segments_from_subtitles(video, subtitles)
    pipeline = build_transcript_segments(segments)
    original_ask = pipeline.ask

    def change_during_search(*args, **kwargs):
        result = original_ask(*args, **kwargs)
        video.write_bytes(b"different video contents")
        return result

    monkeypatch.setattr(pipeline, "ask", change_during_search)
    app = create_app(pipeline, database_path=tmp_path / "library.sqlite3", mode="transcript")
    response = app.test_client().post("/api/ask", json={"question": "relevant"})
    assert response.status_code == 409
    assert "evidence" not in response.get_json()


def test_visual_search_discards_results_if_source_changes_during_query(tmp_path):
    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"original")
    subtitles = tmp_path / "lesson.srt"
    subtitles.write_text("1\n00:00:01,000 --> 00:00:04,000\nRelevant topic.\n", encoding="utf-8")
    segments = segments_from_subtitles(video, subtitles)

    class Visual:
        def search(self, question, limit):
            video.write_bytes(b"different video contents")
            return []

    app = create_app(build_transcript_segments(segments), database_path=tmp_path / "library.sqlite3",
                     mode="transcript", visual_builder=lambda _: Visual())
    client = app.test_client()
    response = client.post("/api/visual-search", json={"question": "diagram"})
    assert response.status_code == 409 and "evidence" not in response.get_json()
    video.write_bytes(b"original")
    fresh = create_app(build_transcript_segments(segments), database_path=tmp_path / "second.sqlite3",
                       mode="transcript", visual_builder=lambda _: Visual()).test_client()
    response = fresh.post("/api/compare-search", json={"question": "relevant"})
    assert response.status_code == 409 and "transcript" not in response.get_json()
    video.write_bytes(b"original")
    fresh = create_app(build_transcript_segments(segments), database_path=tmp_path / "third.sqlite3",
                       mode="transcript", visual_builder=lambda _: Visual()).test_client()
    response = fresh.post("/api/hybrid-search", json={"question": "relevant"})
    assert response.status_code == 409 and "evidence" not in response.get_json()


def test_frame_preview_discards_image_if_source_changes_during_decode(tmp_path, monkeypatch):
    pytest.importorskip("av")
    from video_rag.api import frames

    video = tmp_path / "lesson.mp4"
    video.write_bytes(b"original")
    subtitles = tmp_path / "lesson.srt"
    subtitles.write_text("1\n00:00:01,000 --> 00:00:04,000\nRelevant topic.\n", encoding="utf-8")
    segments = segments_from_subtitles(video, subtitles)

    def change_during_decode(*args):
        video.write_bytes(b"different video contents")
        return b"private image", 2.5

    monkeypatch.setattr(frames, "extract_frame_png", change_during_decode)
    app = create_app(build_transcript_segments(segments), database_path=tmp_path / "library.sqlite3", mode="transcript")
    response = app.test_client().get(f"/api/segments/{segments[0].segment_id}/frame")
    assert response.status_code == 409 and b"private image" not in response.data
