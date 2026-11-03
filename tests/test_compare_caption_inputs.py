import pytest

from scripts.compare_caption_inputs import fixed_windows
from video_rag.schemas import TimedText


def test_fixed_windows_preserve_boundaries_and_assign_by_midpoint(tmp_path):
    cues = [TimedText(0, 2, "first"), TimedText(18, 22, "boundary"), TimedText(41, 43, "last")]
    windows = fixed_windows(cues, 3, 20, tmp_path / "sample.mp4")
    assert [(x.start_time, x.end_time, x.transcript) for x in windows] == [
        (0, 20, "first"), (20, 40, "boundary"), (40, 60, "last")]
    assert [x.segment_id for x in windows] == ["window_0", "window_1", "window_2"]


def test_fixed_windows_reject_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):
        fixed_windows([TimedText(30, 32, "late")], 1, 20, tmp_path)
    with pytest.raises(ValueError):
        fixed_windows([], 1, float("nan"), tmp_path)
