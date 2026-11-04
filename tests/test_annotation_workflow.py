import json

from scripts.annotation_server import create_app


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")


def fixture(tmp_path):
    segment = {
        "segment_id": "video_0000", "video_id": "video", "source_path": str(tmp_path / "video.mp4"),
        "start_time": 0.0, "end_time": 20.0, "transcript": "播报内容", "visual_caption": "画面内容", "keyframes": [],
    }
    question = {
        "question_id": "q1", "question": "谁在播报？", "answer": "记者", "answer_aliases": ["记者"],
        "question_type": "audio", "video_id": "video", "relevant_segment_ids": ["video_0000"],
        "evidence_start": 1.0, "evidence_end": 4.0, "answerable": True,
        "verification_status": "generated_candidate", "annotation_source": "model_generated_pending_review",
        "modality_evidence": {"asr": {"quote": "播报内容", "segment_ids": ["video_0000"], "human_verified": True}},
    }
    candidates, segments, events = tmp_path / "candidates.jsonl", tmp_path / "segments.jsonl", tmp_path / "events.jsonl"
    write_jsonl(candidates, [question])
    write_jsonl(segments, [segment])
    return candidates, segments, events, question


def test_accept_appends_auditable_human_review_event(tmp_path):
    candidates, segments, events, question = fixture(tmp_path)
    client = create_app(candidates, segments, events, "owner", allowed_video_root=tmp_path, allowed_frame_root=tmp_path).test_client()
    response = client.post("/api/review", json={
        "question_id": "q1", "base_event_id": None, "action": "accept", "reason": "played evidence", "edited": question,
        "review_behavior": {"video_played": True, "playback_ranges": [[0, 5]], "review_duration_seconds": 12},
    })
    assert response.status_code == 200
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["annotation_after"]["verification_status"] == "verified"
    assert rows[0]["annotation_after"]["review_event_id"] == rows[0]["event_id"]
    assert rows[0]["reviewer_id"] == "owner"


def test_reject_requires_reason_and_never_creates_verified_annotation(tmp_path):
    candidates, segments, events, question = fixture(tmp_path)
    client = create_app(candidates, segments, events, "owner", allowed_video_root=tmp_path, allowed_frame_root=tmp_path).test_client()
    assert client.post("/api/review", json={
        "question_id": "q1", "base_event_id": None, "action": "reject", "reason": "", "edited": question,
    }).status_code == 400
    response = client.post("/api/review", json={
        "question_id": "q1", "base_event_id": None, "action": "reject", "reason": "ambiguous_answer", "edited": question,
    })
    assert response.status_code == 200
    row = json.loads(events.read_text(encoding="utf-8").strip())
    assert row["annotation_after"] is None


def test_reopen_makes_item_pending_and_stale_tab_conflicts(tmp_path):
    candidates, segments, events, question = fixture(tmp_path)
    client = create_app(candidates, segments, events, "owner", allowed_video_root=tmp_path, allowed_frame_root=tmp_path).test_client()
    accepted = client.post("/api/review", json={
        "question_id": "q1", "base_event_id": None, "action": "accept", "reason": "first pass",
        "edited": question, "review_behavior": {"video_played": True, "playback_ranges": [[0, 5]]},
    }).get_json()
    stale = client.post("/api/review", json={
        "question_id": "q1", "base_event_id": None, "action": "reject", "reason": "stale tab", "edited": question,
    })
    assert stale.status_code == 409
    reopened = client.post("/api/reopen", json={
        "question_id": "q1", "base_event_id": accepted["event_id"], "reason": "second review required",
    })
    assert reopened.status_code == 200
    state = client.get("/api/next").get_json()
    assert state["progress"]["pending"] == 1
    assert state["item"]["revision"] == reopened.get_json()["event_id"]


def test_review_blocks_cross_origin_and_untrusted_host_without_writing(tmp_path):
    candidates, segments, events, question = fixture(tmp_path)
    client = create_app(candidates, segments, events, "owner", allowed_video_root=tmp_path, allowed_frame_root=tmp_path).test_client()
    payload = {"question_id": "q1", "base_event_id": None, "action": "reject", "reason": "test", "edited": question}
    assert client.get("/api/next", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/api/next", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/review", json=payload, headers={"Origin": "https://evil.example"}).status_code == 403
    assert not events.exists()
    response = client.get("/api/next", headers={"Origin": "http://localhost"})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert client.post("/api/review", json=payload, headers={"Origin": "http://localhost"}).status_code == 200


def test_verified_export_keeps_existing_files_and_accepts_identical_retry(tmp_path, monkeypatch):
    import sys
    import pytest
    from scripts import export_verified_questions as exporter
    from video_rag.evaluation.private_output import private_output
    candidates, segments, events, question = fixture(tmp_path)
    client = create_app(candidates, segments, events, "owner", allowed_video_root=tmp_path, allowed_frame_root=tmp_path).test_client()
    response = client.post("/api/review", json={"question_id": "q1", "base_event_id": None, "action": "accept",
        "reason": "played", "edited": question, "review_behavior": {"video_played": True, "playback_ranges": [[0, 5]]}})
    assert response.status_code == 200
    output, report = tmp_path / "private" / "verified.jsonl", tmp_path / "private" / "report.json"
    monkeypatch.setattr(exporter, "private_output", lambda path: private_output(path, (tmp_path,)))
    monkeypatch.setattr(sys, "argv", ["export", "--candidates", str(candidates), "--segments", str(segments),
                                     "--events", str(events), "--output", str(output), "--report", str(report)])
    exporter.main()
    original = output.read_text(encoding="utf-8")
    assert json.loads(report.read_text(encoding="utf-8"))["formal_metric_eligible"] is False
    exporter.main()
    assert output.read_text(encoding="utf-8") == original
    report.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        exporter.main()
    assert output.read_text(encoding="utf-8") == original
    assert report.read_text(encoding="utf-8") == "do not overwrite"
