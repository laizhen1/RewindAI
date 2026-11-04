import pytest

from scripts.evaluate_clip_frames import evaluate, sha256_file, validate_sources
from video_rag.schemas import SearchHit, VideoSegment


def test_visual_metrics_include_absent_queries_and_require_matching_frames():
    segments = [VideoSegment(str(i), "v", "local", i * 2, i * 2 + 2) for i in range(4)]
    class FakeRetriever:
        def search(self, query, count):
            return [SearchHit("0", 0.4, "visual", 1), SearchHit("1", 0.3, "visual", 2)]
    labels = {"review_status": "draft", "segment_midpoints_seconds": [1, 3, 5, 7], "queries": [
        {"query": "present", "relevant_segment_indices": [0, 2]},
        {"query": "absent", "relevant_segment_indices": []},
    ]}
    with pytest.raises(ValueError, match="not reviewed"):
        evaluate(FakeRetriever(), segments, labels)
    result = evaluate(FakeRetriever(), segments, labels, allow_unreviewed=True)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 3, 1, 3)
    assert result["absent_queries_returning_candidates"] == 1
    labels["segment_midpoints_seconds"][0] = 2
    with pytest.raises(ValueError, match="frame times"):
        evaluate(FakeRetriever(), segments, labels, allow_unreviewed=True)


def test_source_hashes_bind_annotations_to_video_and_captions(tmp_path):
    video, captions = tmp_path / "video.mp4", tmp_path / "captions.srt"
    video.write_bytes(b"video")
    captions.write_bytes(b"subtitles")
    labels = {"video_sha256": sha256_file(video), "subtitles_sha256": sha256_file(captions)}
    validate_sources(labels, video, captions)
    video.write_bytes(b"other")
    with pytest.raises(ValueError, match="source hashes"):
        validate_sources(labels, video, captions)
