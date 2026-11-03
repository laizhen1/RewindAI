import numpy as np
import pytest

from video_rag.api import create_app
from video_rag.retrieval.onnx_semantic import SemanticRetriever, build_semantic_pipeline
from video_rag.schemas import VideoSegment


class FakeEncoder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] if "apple" in text else [0.0, 1.0] for text in texts], dtype=np.float32)


def test_semantic_search_ranks_scores_and_is_deterministic():
    retriever = SemanticRetriever(FakeEncoder())
    with pytest.raises(RuntimeError):
        retriever.search("apple", 1)
    retriever.build([VideoSegment("b", "v", "v.mp4", 0, 1, transcript="orange"),
                     VideoSegment("a", "v", "v.mp4", 1, 2, transcript="apple")])
    hits = retriever.search("apple", 2)
    assert [hit.segment_id for hit in hits] == ["a", "b"]
    assert hits[0].score == 1.0
    assert retriever.search("", 2) == []


def test_semantic_index_survives_import_and_restart(tmp_path):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    video.with_suffix(".srt").write_text("1\n00:00:01,000 --> 00:00:04,000\napple example\n", encoding="utf-8")
    builder = lambda segments: build_semantic_pipeline(segments, FakeEncoder(), 0.4)
    db = tmp_path / "workspace.sqlite3"
    client = create_app(builder([]), database_path=db, mode="transcript", transcript_builder=builder,
                        retrieval_type="semantic_experimental").test_client()
    assert client.get("/api/health").get_json()["retrieval_type"] == "semantic_experimental"
    assert client.post("/api/library/local-video", json={"video_path": str(video)}).status_code == 201
    assert not client.post("/api/ask", json={"question": "apple"}).get_json()["abstained"]
    reopened = create_app(builder([]), database_path=db, mode="transcript", transcript_builder=builder,
                          retrieval_type="semantic_experimental").test_client()
    assert not reopened.post("/api/ask", json={"question": "apple"}).get_json()["abstained"]


def test_semantic_threshold_must_be_finite():
    with pytest.raises(ValueError):
        build_semantic_pipeline([], FakeEncoder(), float("nan"))
