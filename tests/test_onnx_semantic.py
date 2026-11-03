import numpy as np
import pytest

from video_rag.retrieval.onnx_semantic import SemanticRetriever
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

