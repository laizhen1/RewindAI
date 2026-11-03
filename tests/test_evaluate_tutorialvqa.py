import pytest

from scripts.evaluate_tutorialvqa import evaluate


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
