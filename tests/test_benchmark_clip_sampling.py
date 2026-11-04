import pytest

from scripts.benchmark_clip_sampling import synthetic_segments


def test_synthetic_segments_are_bounded_and_reject_invalid_inputs(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    segments = synthetic_segments(video, 3, 2.0)
    assert [(s.start_time, s.end_time) for s in segments] == [(0, 2), (2, 4), (4, 6)]
    for count, step in ((0, 1), (5001, 1), (1, 0), (1, float("nan"))):
        with pytest.raises(ValueError):
            synthetic_segments(video, count, step)
