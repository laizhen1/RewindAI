import pytest

from scripts.compare_time_retrieval import compare


class Response:
    def __init__(self, evidence):
        self.evidence = evidence
    def to_dict(self):
        return {"abstained": not self.evidence, "evidence": self.evidence}


class Pipeline:
    def __init__(self, answers):
        self.answers = answers
    def ask(self, question):
        return Response(self.answers[question])


def test_paired_counts_keep_wrong_evidence_and_absences_separate():
    hit = [{"start_time": 0, "end_time": 5}]
    wrong = [{"start_time": 10, "end_time": 15}]
    questions = [{"question": q, "evidence_start_seconds": 0, "evidence_end_seconds": 5} for q in ("a", "b", "c", "d")]
    questions += [{"question": q, "expected_answerable": False} for q in ("e", "f", "g", "h")]
    keyword = Pipeline({"a": hit, "b": hit, "c": wrong, "d": [], "e": [], "f": [], "g": hit, "h": hit})
    semantic = Pipeline({"a": hit, "b": wrong, "c": hit, "d": [], "e": [], "f": hit, "g": [], "h": hit})
    result = compare(questions, keyword, semantic)
    assert result["paired_answerable"] == {"both_hit": 1, "keyword_only_hit": 1, "semantic_only_hit": 1, "neither_hit": 1}
    assert result["paired_unanswerable"] == {"both_abstain": 1, "keyword_only_abstains": 1, "semantic_only_abstains": 1, "neither_abstains": 1}
    assert result["keyword"]["metrics"]["fp"] == 3
    with pytest.raises(ValueError):
        compare([], keyword, semantic)


def test_local_aggregate_report_omits_questions_and_refuses_overwrite(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import compare_time_retrieval as module
    from video_rag.retrieval import onnx_semantic
    root = tmp_path
    video, subtitles, questions = root / "video.mp4", root / "video.srt", root / "questions.json"
    video.write_bytes(b"video")
    subtitles.write_bytes(b"subtitle")
    questions.write_text(json.dumps({"questions": [{"question": "PRIVATE QUESTION"}]}), encoding="utf-8")
    model = root / "model"
    (model / "onnx").mkdir(parents=True)
    (model / "onnx" / "model_quantized.onnx").write_bytes(b"model")
    (model / "tokenizer.json").write_bytes(b"tokenizer")
    output = root / "artifacts" / "report.json"
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "segments_from_subtitles", lambda *args: ["segment"])
    monkeypatch.setattr(module, "build_transcript_segments", lambda segments: object())
    monkeypatch.setattr(onnx_semantic, "OnnxSentenceEncoder", lambda path: object())
    monkeypatch.setattr(onnx_semantic, "build_semantic_pipeline", lambda *args: object())
    monkeypatch.setattr(module, "compare", lambda *args: {"results": [{"question": "PRIVATE QUESTION"}], "total": 1})
    monkeypatch.setattr(sys, "argv", ["compare", "--video", str(video), "--subtitles", str(subtitles),
                                    "--questions", str(questions), "--model-dir", str(model),
                                    "--output", str(output), "--aggregate-only", "--record-hashes"])
    assert module.main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["aggregate_only"] and "results" not in report["comparison"]
    assert "PRIVATE QUESTION" not in output.read_text(encoding="utf-8")
    assert len(report["input_sha256"]) == 5
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
