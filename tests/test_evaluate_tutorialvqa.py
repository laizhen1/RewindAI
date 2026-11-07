import pytest

from scripts.evaluate_tutorialvqa import evaluate, partition_for_video


def test_sentence_counts_and_inclusive_end():
    videos = [{"video_id": "1", "transcript": ["apple fruit", "orange fruit", "pear fruit"]}]
    questions = [{"video_id": "1", "question": "apple", "answer_start": 0, "answer_end": 1}]
    result = evaluate(videos, questions)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)
    assert result["precision"] == 1
    assert result["recall"] == 0.5
    assert result["top1_overlap_rate"] == 1


def test_invalid_label_rejected():
    with pytest.raises(ValueError):
        evaluate([{"video_id": "1", "transcript": ["apple"]}],
                 [{"video_id": "1", "question": "apple", "answer_start": 0, "answer_end": 1}])


def test_context_expands_without_duplicates_and_respects_transcript_boundaries():
    videos = [{"video_id": "1", "transcript": ["apple fruit", "orange fruit", "pear fruit"]}]
    questions = [{"video_id": "1", "question": "apple", "answer_start": 0, "answer_end": 1}]
    result = evaluate(videos, questions, context_sentences=1)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["f1"] == 1.0
    with pytest.raises(ValueError):
        evaluate(videos, questions, context_sentences=-1)


def test_video_partition_is_stable_and_video_wise():
    assignments = {str(i): partition_for_video(str(i)) for i in range(100)}
    assert set(assignments.values()) == {"development", "heldout"}
    assert assignments == {key: partition_for_video(key) for key in assignments}


def test_custom_pipeline_builder_is_used():
    from video_rag.transcript_baseline import build_transcript_segments

    calls = []
    def builder(segments):
        calls.append(segments)
        return build_transcript_segments(segments)
    videos = [{"video_id": "1", "transcript": ["apple fruit", "orange fruit"]}]
    questions = [{"video_id": "1", "question": "apple", "answer_start": 0, "answer_end": 0}] * 2
    result = evaluate(videos, questions, builder=builder)
    assert len(calls) == 1
    assert result["tp"] == 2


def test_video_bootstrap_is_reproducible_and_requires_multiple_videos():
    videos = [{"video_id": "1", "transcript": ["apple"]}, {"video_id": "2", "transcript": ["orange"]}]
    questions = [{"video_id": "1", "question": "apple", "answer_start": 0, "answer_end": 0},
                 {"video_id": "2", "question": "missing", "answer_start": 0, "answer_end": 0}]
    a = evaluate(videos, questions, bootstrap_samples=100, seed=7)
    b = evaluate(videos, questions, bootstrap_samples=100, seed=7)
    assert a["video_bootstrap"] == b["video_bootstrap"]
    assert a["video_bootstrap"]["percentile_95"]["recall"] == [0.0, 1.0]
    with pytest.raises(ValueError):
        evaluate(videos, questions[:1], bootstrap_samples=10)


def test_no_predictions_has_undefined_precision_and_zero_f1():
    videos = [{"video_id": "1", "transcript": ["apple"]}, {"video_id": "2", "transcript": ["orange"]}]
    questions = [{"video_id": "1", "question": "missing", "answer_start": 0, "answer_end": 0},
                 {"video_id": "2", "question": "missing", "answer_start": 0, "answer_end": 0}]
    result = evaluate(videos, questions, bootstrap_samples=20, seed=7)
    assert result["precision"] is None and result["recall"] == result["f1"] == 0
    assert result["video_bootstrap"]["percentile_95"]["precision"] is None
    assert result["video_bootstrap"]["defined_samples"]["precision"] == 0
    assert result["video_bootstrap"]["defined_samples"]["f1"] == 20


def test_bootstrap_excludes_undefined_precision_samples():
    videos = [{"video_id": "1", "transcript": ["apple"]}, {"video_id": "2", "transcript": ["orange"]}]
    questions = [{"video_id": "1", "question": "apple", "answer_start": 0, "answer_end": 0},
                 {"video_id": "2", "question": "missing", "answer_start": 0, "answer_end": 0}]
    result = evaluate(videos, questions, bootstrap_samples=100, seed=7)
    bootstrap = result["video_bootstrap"]
    assert 0 < bootstrap["defined_samples"]["precision"] < 100
    assert bootstrap["percentile_95"]["precision"] == [1.0, 1.0]
