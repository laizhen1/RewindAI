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
