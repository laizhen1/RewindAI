import numpy as np
import pytest

from video_rag.retrieval.onnx_clip import ClipFrameRetriever, _normalize
from video_rag.schemas import VideoSegment


def test_clip_ranking_and_deterministic_ties(monkeypatch):
    pytest.importorskip("av")
    class FakeEncoder:
        def encode_images(self, images):
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        def encode_query(self, query):
            return np.asarray([1.0, 0.0], dtype=np.float32)
    monkeypatch.setattr("video_rag.retrieval.onnx_clip.extract_frame_png", lambda *args: (b"frame", 0))
    monkeypatch.setattr("video_rag.retrieval.onnx_clip.preprocess_png", lambda data: np.zeros((3, 224, 224)))
    retriever = ClipFrameRetriever(FakeEncoder())
    with pytest.raises(RuntimeError):
        retriever.search("picture")
    retriever.build([VideoSegment("a", "v", "local", 0, 1), VideoSegment("b", "v", "local", 1, 2)])
    assert [hit.segment_id for hit in retriever.search("picture")] == ["a", "b"]
    assert retriever.search("", 2) == []


def test_clip_invalid_vectors_are_rejected():
    with pytest.raises(ValueError):
        _normalize(np.zeros((1, 3), dtype=np.float32))
