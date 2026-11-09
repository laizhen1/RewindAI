import numpy as np
import pytest

from video_rag.api.visual_timeline import cover_subtitle_gaps, video_duration_seconds
from video_rag.retrieval.onnx_semantic import SemanticRetriever
from video_rag.schemas import VideoSegment


def segment(identifier, start, end, *, video='v', path='synthetic.mp4'):
    return VideoSegment(identifier, video, path, start, end, transcript='spoken words')


def test_gaps_cover_head_middle_and_tail_without_changing_subtitles():
    original = [segment('b', 30, 40), segment('a', 5, 10), segment('overlap', 8, 12)]
    result = cover_subtitle_gaps(original, 65, window_seconds=10)
    gaps = [item for item in result if not item.transcript]
    assert [(s.start_time, s.end_time) for s in gaps] == [(0, 5), (12, 22), (22, 30),
                                                          (40, 50), (50, 60), (60, 65)]
    assert all(next(item for item in result if item.segment_id == s.segment_id) is s for s in original)
    assert [s.segment_id for s in result] == [s.segment_id for s in cover_subtitle_gaps(original, 65, window_seconds=10)]
    assert [s.segment_id for s in result] == [s.segment_id for s in cover_subtitle_gaps(original, 65.0, window_seconds=10.0)]
    assert all(not s.visual_caption and not s.keyframes for s in gaps)


def test_empty_gap_segments_do_not_enter_the_transcript_vector_index():
    class Encoder:
        def encode(self, texts):
            assert texts == ['spoken words']
            return np.array([[1.0, 0.0]], dtype=np.float32)

    original = segment('a', 10, 20)
    result = cover_subtitle_gaps([original], 30)
    retriever = SemanticRetriever(Encoder())
    retriever.build(result)
    assert retriever.ids == ['a']


def test_gap_ids_change_when_the_underlying_subtitle_revision_changes():
    first = cover_subtitle_gaps([segment('revision-a', 10, 20)], 30)
    second = cover_subtitle_gaps([segment('revision-b', 10, 20)], 30)
    first_ids = {item.segment_id for item in first if not item.transcript}
    second_ids = {item.segment_id for item in second if not item.transcript}
    assert first_ids and first_ids.isdisjoint(second_ids)


def test_gap_windows_validate_bounds_limits_and_video_identity():
    valid = [segment('a', 5, 10)]
    for duration, window in ((0, 20), (float('nan'), 20), (20, 0), (20, float('inf')), (20, 1e-320)):
        with pytest.raises(ValueError):
            cover_subtitle_gaps(valid, duration, window_seconds=window)
    with pytest.raises(ValueError, match='same video'):
        cover_subtitle_gaps(valid + [segment('b', 12, 13, video='other')], 20)
    with pytest.raises(ValueError, match='same video'):
        cover_subtitle_gaps(valid + [segment('a', 12, 13)], 20)
    with pytest.raises(ValueError, match='limit'):
        cover_subtitle_gaps(valid, 100, window_seconds=1, max_gap_segments=10)


def test_video_duration_errors_hide_local_paths(monkeypatch):
    av = pytest.importorskip('av')

    def fail(_path):
        raise av.error.InvalidDataError(22, 'synthetic error', 'private/path.mp4')

    monkeypatch.setattr(av, 'open', fail)
    with pytest.raises(ValueError, match='Could not read the local video duration') as exc:
        video_duration_seconds('private/path.mp4')
    assert 'private/path.mp4' not in str(exc.value)


def test_duration_prefers_video_stream_over_audio_tail(monkeypatch):
    from types import SimpleNamespace

    av = pytest.importorskip('av')
    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(duration=40, time_base=1)])
        duration = 50 * av.time_base
        def __enter__(self): return self
        def __exit__(self, *_args): pass
    monkeypatch.setattr(av, 'open', lambda _path: Container())
    assert video_duration_seconds('synthetic.mp4') == 40
    Container.streams.video = [SimpleNamespace(duration=None, time_base=None)]
    assert video_duration_seconds('synthetic.mp4') == 50
    Container.streams.video = []
    with pytest.raises(ValueError, match='no video stream'):
        video_duration_seconds('synthetic.mp4')
