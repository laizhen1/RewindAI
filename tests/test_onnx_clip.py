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
    monkeypatch.setattr("video_rag.retrieval.onnx_clip.extract_frame_png", lambda _path, time: (b"frame", time + 0.1))
    monkeypatch.setattr("video_rag.retrieval.onnx_clip.preprocess_png", lambda data: np.zeros((3, 224, 224)))
    retriever = ClipFrameRetriever(FakeEncoder())
    with pytest.raises(RuntimeError):
        retriever.search("picture")
    retriever.build([VideoSegment("a", "v", "local", 0, 1), VideoSegment("b", "v", "local", 1, 2)])
    hits = retriever.search("picture")
    assert [hit.segment_id for hit in hits] == ["a", "b"]
    assert [hit.metadata["frame_time"] for hit in hits] == pytest.approx([0.6, 1.6])
    assert retriever.search("", 2) == []


def test_clip_invalid_vectors_are_rejected():
    with pytest.raises(ValueError):
        _normalize(np.zeros((1, 3), dtype=np.float32))


def test_failed_rebuild_preserves_the_previous_visual_index(monkeypatch):
    pytest.importorskip('av')
    class Encoder:
        invalid = False
        def encode_images(self, images):
            return np.ones((0 if self.invalid else len(images), 2), dtype=np.float32)
        def encode_query(self, _query):
            return np.array([1, 0], dtype=np.float32)
    monkeypatch.setattr('video_rag.retrieval.onnx_clip.extract_frame_png', lambda _path, time: (b'frame', time))
    monkeypatch.setattr('video_rag.retrieval.onnx_clip.preprocess_png', lambda _data: np.zeros((3, 224, 224)))
    encoder = Encoder()
    retriever = ClipFrameRetriever(encoder)
    original = VideoSegment('original', 'v', 'local', 0, 1)
    retriever.build([original])
    encoder.invalid = True
    with pytest.raises(ValueError, match='invalid image embeddings'):
        retriever.build([VideoSegment('new', 'v', 'local', 1, 2)])
    hit = retriever.search('question')[0]
    assert hit.segment_id == 'original' and hit.metadata['frame_time'] == .5
    with pytest.raises(ValueError, match='unique'):
        retriever.build([original, original])
    assert retriever.search('question')[0].segment_id == 'original'
