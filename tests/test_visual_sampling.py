import pytest

from video_rag.api.visual_sampling import sample_visual_segments
from video_rag.schemas import VideoSegment


def segment(video, index):
    return VideoSegment(f"{video}-{index}", video, "local", float(index), float(index + 1))


def test_sampling_is_bounded_even_and_keeps_small_videos():
    data = tuple([segment("long", i) for i in range(201)] + [segment("short", i) for i in range(2)])
    sampled = sample_visual_segments(data, limit=10)
    assert len(sampled) == 10
    assert any(x.video_id == "short" for x in sampled)
    indices = [int(x.segment_id.split("-")[1]) for x in sampled if x.video_id == "long"]
    assert indices[0] == 0 and indices[-1] == 200
    assert len({x.segment_id for x in sampled}) == 10
    assert sampled == sample_visual_segments(tuple(reversed(data)), limit=10)


def test_sampling_fails_when_too_many_videos():
    with pytest.raises(ValueError, match="too many videos"):
        sample_visual_segments(tuple(segment(str(i), 0) for i in range(11)), limit=10)
