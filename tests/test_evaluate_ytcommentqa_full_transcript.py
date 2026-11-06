from types import SimpleNamespace

from scripts.evaluate_ytcommentqa import partition
from scripts.evaluate_ytcommentqa_full_transcript import (
    compare,
    evaluate,
    full_transcript,
    overlaps,
    reference_coverage,
    union_duration,
)
from video_rag.transcript_baseline import build_transcript_segments


def row(labels, question="apple"):
    return {"videoID": "v", "question": question, "answerability": labels, "evidence": [1, 2], "caption": [
        {"id": 0, "start": 0, "end": 1, "text": "orange fruit"},
        {"id": 1, "start": 1, "end": 2, "text": "apple fruit"},
        {"id": 2, "start": 2, "end": 3, "text": "other context"},
        {"id": 3, "start": 3, "end": 4, "text": "more fruit"}]}


def test_full_transcript_overlap_uses_only_script_positive_labels():
    data = {"1": row([2]), "2": row([2, 1], "missing"), "3": row([0]), "4": row([1]), "5": row([3])}
    result = evaluate(data, sample="all")
    assert result["evaluated_questions"] == 2
    assert result["excluded_other_labels"] == 3
    assert result["top1_overlap_count"] == result["top3_overlap_count"] == 1
    assert result["top3_reference_half_covered_count"] == 1
    assert result["mean_top3_reference_coverage"] == .25
    assert result["returned_segment_window_overlap_fraction"] == 1
    assert result["reference_window_time_micro"] == {
        "matched_seconds": 1, "returned_seconds": 1, "reference_seconds": 4,
        "precision": 1, "recall": .25, "f1": .4}
    assert result["abstained"] == 1
    assert result["first_three_caption_overlap_count"] == 2
    segments, ref, skipped = full_transcript(data["1"])
    assert skipped == 0 and ref == (1, 3) and not overlaps(segments[0], ref) and not overlaps(segments[3], ref)
    assert partition("v") == partition("v")


def test_paired_metrics_use_identical_questions():
    data = {"1": row([2]), "2": row([2], "missing")}
    result = compare(data, build_transcript_segments, sample="all")
    assert result["keyword"] == result["semantic"]
    assert result["paired_top3"] == {"both_overlap": 1, "keyword_only": 0, "semantic_only": 0, "neither_overlap": 1}
    assert "_outcomes" not in result["keyword"]


def test_invalid_full_caption_is_counted_not_silently_skipped():
    item = row([2])
    item["caption"][0]["end"] = float("nan")
    result = evaluate({"1": item}, sample="all")
    assert result["invalid_transcripts"] == 1 and result["evaluated_questions"] == 0


def test_allow_incomplete_drops_only_invalid_nonreference_cues():
    item = row([2])
    item["caption"][0]["end"] = float("nan")
    result = evaluate({"1": item}, sample="all", allow_incomplete=True)
    assert result["evaluated_questions"] == 1
    assert result["incomplete_transcripts"] == 1 and result["skipped_invalid_cues"] == 1
    item["caption"][1]["end"] = float("nan")
    result = evaluate({"1": item}, sample="all", allow_incomplete=True)
    assert result["evaluated_questions"] == 0 and result["invalid_transcripts"] == 1


def test_paired_results_are_attributed_to_the_correct_method():
    dataset = {"1": row([2])}
    class OutsidePipeline:
        def __init__(self, segments):
            self.segments = segments
        def ask(self, question):
            return SimpleNamespace(evidence=[SimpleNamespace(segment=self.segments[0])])
    result = compare(dataset, OutsidePipeline, sample="all")
    assert result["paired_top3"] == {"both_overlap": 0, "keyword_only": 1, "semantic_only": 0, "neither_overlap": 0}


def test_boolean_labels_are_rejected():
    assert evaluate({"1": row([True])}, sample="all")["invalid_rows"] == 1


def test_reference_coverage_merges_overlaps_and_clips_to_window():
    from video_rag.schemas import VideoSegment
    segments = [VideoSegment(str(i), "v", "local", start, end) for i, (start, end) in enumerate([(8, 12), (11, 15), (18, 25)])]
    assert reference_coverage(segments, (10, 20)) == .7
    assert reference_coverage([], (10, 20)) == 0
    assert reference_coverage(segments[:2], (10, 20)) == .5
    assert union_duration([(8, 12), (11, 15), (18, 25)]) == 14
    assert union_duration([]) == 0


def test_workspace_segmentation_matches_real_subtitle_grouping():
    item = row([2])
    item["caption"][3].update(start=30, end=31)
    segments, ref, _ = full_transcript(item, segmentation="workspace")
    assert [(s.start_time, s.end_time, s.transcript) for s in segments] == [
        (0, 3, "orange fruit apple fruit other context"), (30, 31, "more fruit")]
    assert ref == (1, 3)
    result = compare({"1": item}, build_transcript_segments, sample="all", segmentation="workspace")
    assert result["keyword"]["top3_reference_half_covered_count"] == 1


def test_cli_discards_result_if_dataset_changes(tmp_path, monkeypatch, capsys):
    import json
    import sys

    import pytest

    from scripts import evaluate_ytcommentqa_full_transcript as module

    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"1": row([2])}), encoding="utf-8")
    values = iter(["before", "after"])
    monkeypatch.setattr(module, "fingerprint", lambda path: next(values))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--dataset", str(dataset), "--sample", "all"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "changed during evaluation" in output.err and "metrics" not in output.out


def test_limited_sample_is_deterministic_and_eligible():
    from scripts.evaluate_ytcommentqa_full_transcript import select_eligible

    invalid = row([2])
    invalid['caption'][0]['end'] = -1
    dataset = {'a': row([2]), 'b': row([2, 1]), 'c': row([0]), 'd': invalid, 'e': row([2])}
    sample, eligible = select_eligible(dataset, 2, sample='all', segmentation='workspace')
    reversed_sample, _ = select_eligible(dict(reversed(list(dataset.items()))), 2, sample='all',
                                         segmentation='workspace')
    assert eligible == 3 and len(sample) == 2
    assert list(sample) == list(reversed_sample)
    assert all(key in {'a', 'b', 'e'} for key in sample)


def test_single_semantic_cli_uses_one_model_and_reports_sampling(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from scripts import evaluate_ytcommentqa_full_transcript as module
    from video_rag.retrieval import onnx_semantic, semantic_models

    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps({'1': row([2]), '2': row([2]), '3': row([0])}), encoding='utf-8')
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'synthetic model')
    monkeypatch.setattr(sys, 'argv', ['eval', '--dataset', str(dataset), '--sample', 'all',
                                      '--semantic-model', str(model), '--max-questions', '1'])
    monkeypatch.setattr(semantic_models, 'load_semantic_encoder', lambda path: object())
    monkeypatch.setattr(onnx_semantic, 'build_semantic_pipeline',
                        lambda segments, encoder, threshold: build_transcript_segments(segments))
    module.main()
    result = json.loads(capsys.readouterr().out)
    assert result['eligible_before_limit'] == 2 and result['metrics']['evaluated_questions'] == 1
    assert 'keyword' not in result['metrics'] and 'first_three_caption_overlap_rate' not in result['metrics']
    assert len(result['source_sha256']['semantic_model']) == 64
