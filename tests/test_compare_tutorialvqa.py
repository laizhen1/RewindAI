import pytest

from scripts.compare_tutorialvqa import compare, paired_bootstrap
from video_rag.transcript_baseline import build_transcript_segments


def test_paired_resampling_identical_systems_have_zero_difference():
    units = [([1, 2, 3], [1, 2, 3]), ([2, 3, 4], [2, 3, 4])]
    a = paired_bootstrap(units, samples=100, seed=4)
    assert a == paired_bootstrap(units, samples=100, seed=4)
    assert all(v == 0 for v in a["semantic_minus_keyword"].values())
    assert all(v == [0, 0] for v in a["percentile_95"].values())
    with pytest.raises(ValueError):
        paired_bootstrap(units[:1])
    with pytest.raises(ValueError):
        paired_bootstrap([([1, -1, 0], [1, 0, 0])] * 2)


def test_comparison_aggregates_paired_counts():
    videos = [{"video_id": "1", "transcript": ["apple"]}, {"video_id": "2", "transcript": ["orange"]}]
    questions = [{"video_id": "1", "question": "apple", "answer_start": 0, "answer_end": 0},
                 {"video_id": "2", "question": "missing", "answer_start": 0, "answer_end": 0}]
    result = compare(videos, questions, build_transcript_segments, samples=10)
    assert result["questions"] == result["videos"] == 2
    assert result["keyword"] == result["semantic"]
    assert result["keyword"]["tp"] == 1
    assert result["keyword"]["fn"] == 1
