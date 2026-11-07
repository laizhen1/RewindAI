import csv
import json

import pytest

from scripts.prepare_temporal_questions import prepare
from scripts.temporal_label_server import case_result, create_label_app


@pytest.fixture
def labeler(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"example video")
    output = tmp_path / "private" / "labels.csv"
    app = create_label_app(video, output, "sample", "development", 100, allowed_roots=(tmp_path / "private",))
    return app.test_client(), video, output, tmp_path


def test_case_status_distinguishes_time_miss_and_unanswerable_candidate():
    prediction = {"abstained": False, "first_hit_reference_coverage": 0.2, "returned_ranges": [[10, 15]],
                  "question": "private text"}
    assert case_result({"expected_answerable": True}, prediction) == {"status": "time_mismatch", "returned_ranges": [[10, 15]]}
    assert case_result({"expected_answerable": False}, prediction)["status"] == "candidate_on_unanswerable"


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


def test_review_and_edit_keep_id_validate_and_reject_stale_updates(labeler):
    client, _, output, _ = labeler
    revision = client.get("/api/entries").get_json()["revision"]
    payload = {"revision": revision, "question": "Original question", "question_type": "audio",
               "answerable": True, "answer": "Original answer", "start": 3, "end": 8}
    created = client.post("/api/entries", json=payload).get_json()
    listing = client.get("/api/entries").get_json()
    entry = listing["entries"][0]
    assert entry["question"] == "Original question" and entry["annotation_status"] == "draft_unverified"
    endpoint = f"/api/entries/{entry['question_id']}"
    changed = {**payload, "revision": created["revision"], "question": "Corrected question", "start": 4}
    updated = client.put(endpoint, json=changed)
    assert updated.status_code == 200 and updated.get_json()["count"] == 1
    assert client.put(endpoint, json=changed).status_code == 409
    latest = client.get("/api/entries").get_json()
    assert latest["entries"][0]["question_id"] == entry["question_id"]
    assert latest["entries"][0]["question"] == "Corrected question"
    assert client.put("/api/entries/missing", json={**changed, "revision": latest["revision"]}).status_code == 404
    assert client.put(endpoint, json={**changed, "revision": latest["revision"], "end": 200}).status_code == 400
    with output.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f)); fields = list(rows[0])
    rows[0]["evidence_frame_seconds"] = "5"
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    latest = client.get("/api/entries").get_json()
    assert client.put(endpoint, json={**changed, "revision": latest["revision"]}).status_code == 409
    with output.open(encoding="utf-8", newline="") as f:
        assert next(csv.DictReader(f))["evidence_frame_seconds"] == "5"


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
    assert data["coverage_feasibility"] == {"answerable_questions": 1, "structurally_unattainable": 0,
                                             "reference_coverage_threshold": 0.5}
    assert "cases" not in data
    assert not (tmp_path / "private" / "reports").exists()
    assert "private answer" not in response.get_data(as_text=True)
    assert "attention" not in response.get_data(as_text=True)
    negative = client.post("/api/entries", json={"revision": result["revision"], "question": "quantum entanglement",
                           "question_type": "audio", "answerable": False, "answer": "", "checked_scope": "all",
                           "checked_entire_video": True}).get_json()
    result2 = client.post("/api/evaluate", json={"revision": negative["revision"]}).get_json()
    assert result2["metrics"]["tn"] == 1 and result2["metrics"]["question_count"] == 2
    cases = client.post("/api/evaluate", json={"revision": negative["revision"], "include_cases": True}).get_json()["cases"]
    assert cases[0]["keyword"]["status"] == "time_match" and cases[0]["reference_range"] == [3, 8]
    assert cases[0]["keyword"]["returned_ranges"] == [[3, 8]] and cases[0]["semantic"] is None
    assert cases[1]["keyword"]["status"] == "abstained" and cases[1]["reference_range"] is None
    assert client.post("/api/evaluate", json={"revision": negative["revision"], "include_cases": 1}).status_code == 400
    assert client.post("/api/evaluate", json={"revision": negative["revision"], "save_report": 1}).status_code == 400
    saved_report = client.post("/api/evaluate", json={"revision": negative["revision"], "include_cases": True,
                                                     "save_report": True}).get_json()
    report_path = tmp_path / "private" / "reports" / saved_report["saved_report_filename"]
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["status"] == "unverified_draft" and report["summary"]["metrics"]["question_count"] == 2
    assert report["source_sha256"]["annotation_csv"] == negative["revision"]
    assert "cases" not in report["summary"] and "attention" not in report_text and "private answer" not in report_text
    reserved = create_label_app(video, tmp_path / "private" / "test.csv", "reserved", "test", 100,
                                allowed_roots=(tmp_path / "private",), subtitles=subs).test_client()
    assert not reserved.get("/api/entries").get_json()["evaluation_available"]
    assert reserved.post("/api/evaluate", json={}).status_code == 409


def test_local_diagnostic_warns_when_reference_cannot_be_covered(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    subs = tmp_path / "sample.srt"
    subs.write_text("1\n00:00:03,000 --> 00:00:08,000\nAttention mechanism.\n", encoding="utf-8")
    client = create_label_app(video, tmp_path / "private" / "draft.csv", "sample", "development", 100,
                              allowed_roots=(tmp_path / "private",), subtitles=subs).test_client()
    revision = client.get("/api/entries").get_json()["revision"]
    saved = client.post("/api/entries", json={"revision": revision, "question": "attention",
                        "question_type": "audio", "answerable": True, "answer": "private answer",
                        "start": 0, "end": 50}).get_json()
    response = client.post("/api/evaluate", json={"revision": saved["revision"]})
    assert response.status_code == 200
    assert response.get_json()["coverage_feasibility"]["structurally_unattainable"] == 1
    assert "private answer" not in response.get_data(as_text=True)


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
    assert client.post("/api/evaluate", json={"revision": result["revision"], "save_report": True}).status_code == 409
    assert not (tmp_path / "private" / "reports").exists()


def test_local_paired_comparison_keeps_queries_private_and_checks_model_revision(tmp_path, monkeypatch):
    from video_rag.retrieval import onnx_semantic
    from video_rag.transcript_baseline import build_transcript_segments

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    subs = tmp_path / "sample.srt"
    subs.write_text("1\n00:00:03,000 --> 00:00:08,000\nAttention mechanism.\n", encoding="utf-8")
    model = tmp_path / "model"
    (model / "onnx").mkdir(parents=True)
    (model / "tokenizer.json").write_bytes(b"tokenizer")
    (model / "onnx" / "model_quantized.onnx").write_bytes(b"model")
    monkeypatch.setattr(onnx_semantic, "OnnxSentenceEncoder", lambda directory: object())
    monkeypatch.setattr(onnx_semantic, "build_semantic_pipeline", lambda segments, encoder, threshold: build_transcript_segments(segments))
    app = create_label_app(video, tmp_path / "private" / "draft.csv", "sample", "development", 100,
                           allowed_roots=(tmp_path / "private",), subtitles=subs, semantic_model=model)
    client = app.test_client()
    revision = client.get("/api/entries").get_json()["revision"]
    saved = client.post("/api/entries", json={"revision": revision, "question": "attention", "question_type": "audio",
                        "answerable": True, "answer": "private answer", "start": 3, "end": 8}).get_json()
    response = client.post("/api/evaluate", json={"revision": saved["revision"]})
    assert response.status_code == 200
    result = response.get_json()
    assert result["metrics"]["f1"] == result["semantic"]["metrics"]["f1"] == 1
    assert result["paired"]["answerable"]["both_hit"] == 1 and result["semantic_threshold"] == 0.4
    assert "attention" not in response.get_data(as_text=True) and "private answer" not in response.get_data(as_text=True)
    cases = client.post("/api/evaluate", json={"revision": saved["revision"], "include_cases": True}).get_json()["cases"]
    assert cases[0]["keyword"]["status"] == cases[0]["semantic"]["status"] == "time_match"
    def mutate(segments, encoder, threshold):
        (model / "tokenizer.json").write_bytes(b"changed")
        return build_transcript_segments(segments)
    monkeypatch.setattr(onnx_semantic, "build_semantic_pipeline", mutate)
    assert client.post("/api/evaluate", json={"revision": saved["revision"]}).status_code == 409


def test_development_excerpts_are_bounded_and_do_not_create_labels(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    subs = tmp_path / "sample.srt"
    subs.write_text("\n".join(f"{i + 1}\n00:{i // 60:02}:{i % 60:02},000 --> 00:{(i + 1) // 60:02}:{(i + 1) % 60:02},000\nExample topic {i}.\n"
                              for i in range(0, 99, 7)), encoding="utf-8")
    app = create_label_app(video, tmp_path / "private" / "dev.csv", "sample", "development", 100,
                           allowed_roots=(tmp_path / "private",), subtitles=subs)
    client = app.test_client()
    assert client.get("/api/entries").get_json()["suggestions_available"]
    result = client.get("/api/development-excerpts")
    assert result.status_code == 200
    excerpts = result.get_json()["excerpts"]
    assert len(excerpts) == 12
    assert len({item["start"] for item in excerpts}) == len(excerpts)
    assert client.get("/api/entries").get_json()["count"] == 0
    assert client.get("/api/development-excerpts", headers={"Origin": "https://outside.example"}).status_code == 403
    reserved = create_label_app(video, tmp_path / "private" / "test.csv", "reserved", "test", 100,
                                allowed_roots=(tmp_path / "private",), subtitles=subs).test_client()
    assert not reserved.get("/api/entries").get_json()["suggestions_available"]
    assert reserved.get("/api/development-excerpts").status_code == 409
    validation = create_label_app(video, tmp_path / "private" / "validation.csv", "validation", "validation", 100,
                                  allowed_roots=(tmp_path / "private",), subtitles=subs).test_client()
    assert not validation.get("/api/entries").get_json()["suggestions_available"]
    assert validation.get("/api/development-excerpts").status_code == 409


def test_development_excerpts_reject_source_mutation(tmp_path, monkeypatch):
    from scripts import temporal_label_server as module

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    subs = tmp_path / "sample.srt"
    subs.write_text("1\n00:00:03,000 --> 00:00:08,000\nAttention mechanism.\n", encoding="utf-8")
    app = create_label_app(video, tmp_path / "private" / "dev.csv", "sample", "development", 100,
                           allowed_roots=(tmp_path / "private",), subtitles=subs)
    original = module.segments_from_subtitles
    def mutate(*args):
        segments = original(*args)
        subs.write_text("changed", encoding="utf-8")
        return segments
    monkeypatch.setattr(module, "segments_from_subtitles", mutate)
    assert app.test_client().get("/api/development-excerpts").status_code == 409


def test_model_suggestion_origin_is_preserved_in_development_only(labeler):
    client, video, output, tmp = labeler
    initial = client.get('/api/entries').get_json()
    payload = {'revision': initial['revision'], 'question': 'What is the topic?', 'question_type': 'audio',
               'answerable': True, 'answer': 'A topic', 'start': 3.0, 'end': 8.0,
               'model_generated_candidate': True}
    assert client.post('/api/entries', json=payload).status_code == 201
    with output.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    prepared = prepare(rows, 'sample', 'development')
    assert prepared['questions'][0]['annotation_source'] == 'model_generated_candidate_unverified'
    entry = client.get('/api/entries').get_json()['entries'][0]
    assert entry['notes'] == 'model_generated_candidate'
    validation = create_label_app(video, tmp / 'private' / 'validation.csv', 'sample', 'validation', 100,
                                  allowed_roots=(tmp / 'private',)).test_client()
    payload['revision'] = validation.get('/api/entries').get_json()['revision']
    assert validation.post('/api/entries', json=payload).status_code == 400


def test_candidates_are_source_bound_development_only_and_never_auto_saved(labeler):
    import hashlib

    _client, video, output, tmp = labeler
    subtitle = tmp / 'sample.srt'
    subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nA sample\n', encoding='utf-8')
    candidates = tmp / 'private' / 'suggestions.json'
    suggestions = {'status': 'model_generated_unverified', 'split': 'development',
                   'input_sha256': {'video': hashlib.sha256(video.read_bytes()).hexdigest(),
                                    'subtitles': hashlib.sha256(subtitle.read_bytes()).hexdigest()},
                   'suggestions': [{'question': 'What?', 'answer': 'Sample', 'start_seconds': 1,
                                    'end_seconds': 3, 'status': 'model_generated_unverified'}]}
    candidates.write_text(json.dumps(suggestions), encoding='utf-8')
    app = create_label_app(video, output, 'sample', 'development', 100, allowed_roots=(tmp / 'private',),
                           subtitles=subtitle, candidates=candidates)
    client = app.test_client()
    assert client.get('/api/entries').json['model_candidates_available'] is True
    assert client.get('/api/development-candidates').json['candidates'][0]['question'] == 'What?'
    assert client.get('/api/entries').json['count'] == 0
    subtitle.write_text('changed', encoding='utf-8')
    assert client.get('/api/development-candidates').status_code == 400
    with pytest.raises(ValueError, match='development'):
        create_label_app(video, tmp / 'private' / 'validation.csv', 'sample', 'validation', 100,
                         allowed_roots=(tmp / 'private',), subtitles=subtitle, candidates=candidates)


def test_candidate_origin_survives_into_answer_review(labeler):
    from scripts.evaluate_local_answers import prepare_review
    from video_rag.schemas import Answer

    client, _video, output, _tmp = labeler
    revision = client.get('/api/entries').json['revision']
    payload = {'revision': revision, 'question': 'What is the topic?', 'question_type': 'audio',
               'answerable': True, 'answer': 'A topic', 'start': 3.0, 'end': 8.0,
               'model_generated_candidate': True}
    assert client.post('/api/entries', json=payload).status_code == 201
    with output.open(encoding='utf-8', newline='') as f:
        data = prepare(list(csv.DictReader(f)), 'sample', 'development')

    class StubPipeline:
        def ask(self, question):
            return Answer(answer='No answer', evidence=(), abstained=True)

    row = prepare_review(StubPipeline(), data)[0]
    assert row['annotation_source'] == 'model_generated_candidate_unverified'
    assert row['human_answer_verdict'] is None


def test_labeler_semantic_only_does_not_run_keyword_pipeline(tmp_path, monkeypatch):
    from scripts import temporal_label_server as module
    from video_rag.retrieval import onnx_semantic, semantic_models
    from video_rag.transcript_baseline import build_transcript_segments

    video = tmp_path / 'sample.mp4'
    video.write_bytes(b'video')
    subs = tmp_path / 'sample.srt'
    subs.write_text('1\n00:00:03,000 --> 00:00:08,000\nAttention mechanism.\n', encoding='utf-8')
    model = tmp_path / 'embedding.gguf'
    model.write_bytes(b'model')
    monkeypatch.setattr(semantic_models, 'load_semantic_encoder', lambda path: object())
    monkeypatch.setattr(onnx_semantic, 'build_semantic_pipeline',
                        lambda segments, encoder, threshold: build_transcript_segments(segments))
    monkeypatch.setattr(module, 'build_transcript_segments', lambda segments: pytest.fail('keyword pipeline ran'))
    app = create_label_app(video, tmp_path / 'private' / 'draft.csv', 'sample', 'development', 100,
                           allowed_roots=(tmp_path / 'private',), subtitles=subs, semantic_model=model)
    client = app.test_client()
    entries = client.get('/api/entries').get_json()
    assert entries['semantic_available'] is True
    saved = client.post('/api/entries', json={'revision': entries['revision'], 'question': 'attention',
                        'question_type': 'audio', 'answerable': True, 'answer': 'answer',
                        'start': 3, 'end': 8}).get_json()
    response = client.post('/api/evaluate', json={'revision': saved['revision'], 'semantic_only': True,
                                                   'include_cases': True, 'save_report': True})
    assert response.status_code == 200
    report = response.get_json()
    assert report['retrieval_mode'] == 'semantic' and report['metrics']['f1'] == 1
    assert report['paired'] is None and report['semantic_threshold'] == 0.4
    assert report['cases'][0]['keyword'] is None and report['cases'][0]['semantic']['status'] == 'time_match'
