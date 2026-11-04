import pytest

from scripts.evaluate_ytcommentqa import evaluate, evidence_segments, partition


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
