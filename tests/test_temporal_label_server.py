import csv

import pytest

from scripts.prepare_temporal_questions import prepare
from scripts.temporal_label_server import create_label_app


@pytest.fixture
def labeler(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"example video")
    output = tmp_path / "private" / "labels.csv"
    app = create_label_app(video, output, "sample", "development", 100, allowed_roots=(tmp_path / "private",))
    return app.test_client(), video, output, tmp_path


def test_saves_draft_that_can_be_converted_and_detects_stale_revision(labeler):
    client, video, output, _ = labeler
    initial = client.get("/api/entries").get_json()
    payload = {"revision": initial["revision"], "question": "What is discussed?", "question_type": "audio",
               "answerable": True, "answer": "A topic", "start": 3.0, "end": 8.0}
    result = client.post("/api/entries", json=payload)
    assert result.status_code == 201 and result.get_json()["count"] == 1
    assert client.post("/api/entries", json=payload).status_code == 409
    with output.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["annotation_status"] == "draft_unverified"
    prepared = prepare(rows, "sample", "development")
    assert prepared["status"] == "csv_import_unverified_draft"
    assert prepared["questions"][0]["evidence_start_seconds"] == 3
    assert client.get("/video", headers={"Range": "bytes=0-3"}).status_code == 206
    video.write_bytes(b"changed video")
    assert client.get("/video").status_code == 409
    assert client.post("/api/entries", json={**payload, "revision": result.get_json()["revision"]}).status_code == 409


def test_validates_missing_scope_ranges_types_and_local_origin(labeler):
    client, _, _, _ = labeler
    rev = client.get("/api/entries").get_json()["revision"]
    base = {"revision": rev, "question": "Question?", "question_type": "audio", "answerable": True,
            "answer": "Answer", "start": 1, "end": 2}
    for updates in ({"end": 101}, {"start": True}, {"question_type": []}, {"answer": ""},
                    {"answerable": False, "answer": "", "checked_scope": ""}):
        assert client.post("/api/entries", json={**base, **updates}).status_code == 400
    assert client.post("/api/entries", json={**base, "answerable": False, "answer": "", "checked_scope": "partial"}).status_code == 400
    negative = {**base, "answerable": False, "answer": "", "checked_scope": "Entire video watched", "checked_entire_video": True}
    assert client.post("/api/entries", json=negative).status_code == 201
    assert client.get("/api/entries", headers={"Origin": "https://outside.example"}).status_code == 403
    assert client.get("/api/entries", headers={"Host": "outside.example"}).status_code == 403
    assert client.get("/").headers["Cache-Control"] == "no-store"


def test_reopening_rejects_different_video_and_unbound_existing_file(labeler):
    _, video, output, tmp = labeler
    assert create_label_app(video, output, "sample", "development", 100, allowed_roots=(tmp / "private",))
    other = tmp / "other.mp4"
    other.write_bytes(b"different")
    with pytest.raises(ValueError, match="different source"):
        create_label_app(other, output, "sample", "development", 100, allowed_roots=(tmp / "private",))
    unbound = tmp / "private" / "unbound.csv"
    unbound.write_text("not a bound file", encoding="utf-8")
    with pytest.raises(ValueError, match="no source binding"):
        create_label_app(video, unbound, "sample", "development", 100, allowed_roots=(tmp / "private",))
    with pytest.raises(ValueError, match="under artifacts"):
        create_label_app(video, tmp / "outside.csv", "sample", "development", 100, allowed_roots=(tmp / "private",))


def test_local_diagnostic_uses_drafts_without_exposing_question_text(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    subs = tmp_path / "sample.srt"
    subs.write_text("1\n00:00:03,000 --> 00:00:08,000\nAttention mechanism.\n", encoding="utf-8")
    app = create_label_app(video, tmp_path / "private" / "draft.csv", "sample", "development", 100,
                           allowed_roots=(tmp_path / "private",), subtitles=subs)
    client = app.test_client()
    entry = client.get("/api/entries").get_json()
    assert entry["evaluation_available"]
    assert client.post("/api/evaluate", json={"revision": entry["revision"]}).status_code == 400
    result = client.post("/api/entries", json={"revision": entry["revision"], "question": "attention",
                         "question_type": "audio", "answerable": True, "answer": "private answer",
                         "start": 3, "end": 8}).get_json()
    assert client.post("/api/evaluate", json={"revision": entry["revision"]}).status_code == 409
    response = client.post("/api/evaluate", json={"revision": result["revision"]})
    assert response.status_code == 200
    data = response.get_json()
    assert data["metrics"]["tp"] == 1 and data["metrics"]["f1"] == 1
    assert data["metrics"]["question_count"] == 1 and data["subtitle_last_end_seconds"] == 8
    assert "private answer" not in response.get_data(as_text=True)
    assert "attention" not in response.get_data(as_text=True)
    negative = client.post("/api/entries", json={"revision": result["revision"], "question": "quantum entanglement",
                           "question_type": "audio", "answerable": False, "answer": "", "checked_scope": "all",
                           "checked_entire_video": True}).get_json()
    result2 = client.post("/api/evaluate", json={"revision": negative["revision"]}).get_json()
    assert result2["metrics"]["tn"] == 1 and result2["metrics"]["question_count"] == 2
    reserved = create_label_app(video, tmp_path / "private" / "test.csv", "reserved", "test", 100,
                                allowed_roots=(tmp_path / "private",), subtitles=subs).test_client()
    assert not reserved.get("/api/entries").get_json()["evaluation_available"]
    assert reserved.post("/api/evaluate", json={}).status_code == 409


def test_local_diagnostic_rejects_subtitle_mutation(tmp_path, monkeypatch):
    from scripts import temporal_label_server as module

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    subs = tmp_path / "sample.srt"
    subs.write_text("1\n00:00:03,000 --> 00:00:08,000\nAttention mechanism.\n", encoding="utf-8")
    app = create_label_app(video, tmp_path / "private" / "draft.csv", "sample", "validation", 100,
                           allowed_roots=(tmp_path / "private",), subtitles=subs)
    client = app.test_client()
    revision = client.get("/api/entries").get_json()["revision"]
    result = client.post("/api/entries", json={"revision": revision, "question": "attention", "question_type": "audio",
                         "answerable": True, "answer": "answer", "start": 3, "end": 8}).get_json()
    original = module.build_transcript_segments
    def mutate(segments):
        pipeline = original(segments)
        subs.write_text("changed", encoding="utf-8")
        return pipeline
    monkeypatch.setattr(module, "build_transcript_segments", mutate)
    assert client.post("/api/evaluate", json={"revision": result["revision"]}).status_code == 409
