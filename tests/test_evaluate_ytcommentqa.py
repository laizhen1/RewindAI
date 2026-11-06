import pytest

from scripts.evaluate_ytcommentqa import compare, evaluate, evidence_segments, partition


def row(labels, question="apple", video="v"):
    return {"videoID": video, "question": question, "answerability": labels, "evidence": [1, 2], "caption": [
        {"id": 0, "start": 0, "end": 1, "text": "outside context"},
        {"id": 1, "start": 1, "end": 2, "text": "apple fruit"},
        {"id": 2, "start": 2, "end": 3, "text": "orange fruit"}]}


def test_snippet_proxy_and_exclusion_do_not_include_outside_captions():
    dataset = {"1": row([2]), "2": row([0]), "3": row([2, 1], question="missing"),
               "4": row([0], question="missing"), "5": row([1]), "6": row([3])}
    metrics = evaluate(dataset, sample="all")
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["f1"] == .5
    assert metrics["excluded_visual_or_combined_only"] == 2
    assert [s.segment_id for s in evidence_segments(dataset["1"])] == ["1", "2"]
    assert partition("v") == partition("v")


def test_bad_caption_rejected_and_counted():
    invalid = row([2])
    invalid["caption"][1]["end"] = 0
    with pytest.raises(ValueError):
        evidence_segments(invalid)
    assert evaluate({"1": invalid}, sample="all")["invalid_evidence_captions"] == 1


def test_paired_comparison_counts_same_questions_without_exposing_text():
    from video_rag.transcript_baseline import build_transcript_segments
    dataset = {"1": row([2]), "2": row([0]), "3": row([2], question="missing"), "4": row([0], question="missing")}
    result = compare(dataset, build_transcript_segments, sample="all")
    assert result["keyword"] == result["semantic"]
    assert result["script_answerable_pairs"] == {"both_return": 1, "keyword_only": 0, "semantic_only": 0, "neither_return": 1}
    assert result["unanswerable_pairs"] == {"both_return": 1, "keyword_only": 0, "semantic_only": 0, "neither_return": 1}
    assert "_predictions" not in result["keyword"]
    assert "apple fruit" not in str(result)


def test_paired_comparison_distinguishes_each_method():
    from types import SimpleNamespace
    dataset = {"1": row([2]), "2": row([0]), "3": row([2], question="missing"), "4": row([0], question="missing")}
    class FakeSemantic:
        def ask(self, question):
            return SimpleNamespace(evidence=[1] if question == "missing" else [])
    result = compare(dataset, lambda segments: FakeSemantic(), sample="all")
    assert result["script_answerable_pairs"] == {"both_return": 0, "keyword_only": 1, "semantic_only": 1, "neither_return": 0}
    assert result["unanswerable_pairs"] == {"both_return": 0, "keyword_only": 1, "semantic_only": 1, "neither_return": 0}


def test_boolean_labels_are_not_treated_as_integer_categories():
    assert evaluate({"1": row([False])}, sample="all")["invalid_rows"] == 1


def test_cli_single_semantic_model_does_not_run_comparison(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from scripts import evaluate_ytcommentqa as module
    from video_rag.retrieval import onnx_semantic, semantic_models

    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps({'1': row([2])}), encoding='utf-8')
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'synthetic model')
    monkeypatch.setattr(sys, 'argv', ['eval', '--dataset', str(dataset), '--sample', 'all',
                                      '--semantic-model', str(model)])
    monkeypatch.setattr(semantic_models, 'load_semantic_encoder', lambda path: object())
    from video_rag.transcript_baseline import build_transcript_segments
    monkeypatch.setattr(onnx_semantic, 'build_semantic_pipeline',
                        lambda segments, encoder, threshold: build_transcript_segments(segments))
    monkeypatch.setattr(module, 'compare', lambda *args, **kwargs: pytest.fail('comparison was requested'))
    module.main()
    result = json.loads(capsys.readouterr().out)
    assert result['metrics']['tp'] == 1
    assert len(result['input_sha256']['semantic_model']) == 64


def test_single_model_cli_rejects_model_change_during_evaluation(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from scripts import evaluate_ytcommentqa as module
    from video_rag.retrieval import semantic_models

    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps({'1': row([2])}), encoding='utf-8')
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'initial')
    monkeypatch.setattr(sys, 'argv', ['eval', '--dataset', str(dataset), '--semantic-model', str(model)])
    monkeypatch.setattr(semantic_models, 'load_semantic_encoder', lambda path: object())

    def change_model(*args, **kwargs):
        model.write_bytes(b'changed')
        return {'tp': 1}

    monkeypatch.setattr(module, 'evaluate', change_model)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert 'changed during evaluation' in capsys.readouterr().err


def test_single_model_cli_detects_dataset_change_while_loading(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from pathlib import Path

    from scripts import evaluate_ytcommentqa as module
    from video_rag.retrieval import semantic_models

    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps({'1': row([2])}), encoding='utf-8')
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'synthetic')
    monkeypatch.setattr(sys, 'argv', ['eval', '--dataset', str(dataset), '--semantic-model', str(model)])
    monkeypatch.setattr(semantic_models, 'load_semantic_encoder', lambda path: object())
    monkeypatch.setattr(module, 'evaluate', lambda *args, **kwargs: {'tp': 1})
    read_text = Path.read_text

    def changed_after_read(path, *args, **kwargs):
        text = read_text(path, *args, **kwargs)
        if path == dataset:
            dataset.write_text('{}', encoding='utf-8')
        return text

    monkeypatch.setattr(Path, 'read_text', changed_after_read)
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2
    assert 'changed during evaluation' in capsys.readouterr().err
