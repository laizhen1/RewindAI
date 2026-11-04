import pytest

from scripts.compare_frame_retrieval import compare
from video_rag.schemas import SearchHit, VideoSegment


def test_compare_uses_same_frame_labels_and_preserves_absent_queries():
    segments = [VideoSegment("0", "v", "local", 0, 2, transcript="apple fruit"),
                VideoSegment("1", "v", "local", 2, 4, transcript="orange fruit")]
    labels = {"review_status": "draft", "segment_midpoints_seconds": [1, 3], "queries": [
        {"query": "apple", "relevant_segment_indices": [0]},
        {"query": "cat", "relevant_segment_indices": []},
    ]}
    class Visual:
        def search(self, query, count):
            return [SearchHit("1", .5, "visual", 1)]
    with pytest.raises(ValueError, match="not reviewed"):
        compare(segments, labels, Visual())
    result = compare(segments, labels, Visual(), allow_unreviewed=True)
    assert (result["transcript"]["tp"], result["transcript"]["fp"]) == (1, 0)
    assert (result["visual"]["tp"], result["visual"]["fp"], result["visual"]["fn"]) == (0, 2, 1)
    assert result["visual"]["absent_queries_returning_candidates"] == 1
    assert result["transcript"]["absent_queries_returning_candidates"] == 0


def test_no_predictions_with_positive_labels_has_zero_f1():
    segments = [VideoSegment("0", "v", "local", 0, 2, transcript="apple")]
    labels = {"review_status": "reviewed", "segment_midpoints_seconds": [1], "queries": [
        {"query": "dog", "relevant_segment_indices": [0]}]}
    class Empty:
        def search(self, query, count):
            return []
    result = compare(segments, labels, Empty())
    assert result["visual"]["precision"] is None
    assert result["visual"]["recall"] == 0
    assert result["visual"]["f1"] == 0
