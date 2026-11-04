import json

import pytest

from scripts.prepare_clip_labels import prepare


def test_prepares_private_draft_without_overwriting_or_escaping(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    subtitles = tmp_path / "video.srt"
    subtitles.write_text("1\n00:00:01,000 --> 00:00:03,000\nhello\n", encoding="utf-8")
    local = tmp_path / "artifacts"
    target = local / "review"
    monkeypatch.setattr("scripts.prepare_clip_labels.extract_frame_png", lambda *args: (b"fake-png", 2.0))
    assert prepare(video, subtitles, target, (local,)) == 1
    assert (target / "frame-000.png").read_bytes() == b"fake-png"
    manifest = json.loads((target / "labels.json").read_text(encoding="utf-8"))
    assert manifest["review_status"] == "draft" and manifest["segment_midpoints_seconds"] == [2.0]
    with pytest.raises(ValueError, match="already exists"):
        prepare(video, subtitles, target, (local,))
    with pytest.raises(ValueError, match="subdirectory"):
        prepare(video, subtitles, tmp_path / "outside", (local,))
