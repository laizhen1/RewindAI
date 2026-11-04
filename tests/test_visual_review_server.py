import json

import pytest

from scripts.evaluate_clip_frames import sha256_file
from scripts.visual_review_server import create_review_app


@pytest.fixture
def review_workspace(tmp_path):
    root = tmp_path / "artifacts"
    review = root / "review"
    review.mkdir(parents=True)
    video, subs = tmp_path / "video.mp4", tmp_path / "video.srt"
    video.write_bytes(b"video")
    subs.write_bytes(b"captions")
    (review / "frame-000.png").write_bytes(b"fake")
    manifest = {"video_sha256": sha256_file(video), "subtitles_sha256": sha256_file(subs),
                "segment_midpoints_seconds": [2.0], "decoded_frame_seconds": [2.0],
                "review_status": "draft", "queries": []}
    (review / "labels.json").write_text(json.dumps(manifest), encoding="utf-8")
    return review, video, subs, root


def test_review_saves_locally_rejects_stale_revisions_and_preserves_source_binding(review_workspace):
    review, video, subs, root = review_workspace
    client = create_review_app(review, video, subs, allowed_roots=(root,)).test_client()
    initial = client.get("/api/review").get_json()
    assert client.get("/api/frames/0").status_code == 200
    assert client.get("/api/frames/1").status_code == 404
    payload = {"revision": initial["revision"], "reviewed": True,
               "queries": [{"query": "a chart", "relevant_segment_indices": [0]}]}
    assert client.put("/api/review", json=payload, headers={"Origin": "https://attacker.example"}).status_code == 403
    saved = client.put("/api/review", json=payload)
    assert saved.status_code == 200
    assert client.put("/api/review", json=payload).status_code == 409
    labels = json.loads((review / "labels.json").read_text(encoding="utf-8"))
    assert labels["review_status"] == "reviewed" and labels["video_sha256"] == sha256_file(video)
    assert labels["queries"][0]["relevant_segment_indices"] == [0]
    payload["revision"] = saved.get_json()["revision"]
    payload["reviewed"] = False
    payload["queries"] = [{"query": "absent", "relevant_segment_indices": []}]
    assert client.put("/api/review", json=payload).status_code == 200
    assert json.loads((review / "labels.json").read_text(encoding="utf-8"))["review_status"] == "draft"


def test_review_refuses_mismatched_sources_and_output_outside_local_roots(review_workspace):
    review, video, subs, root = review_workspace
    with pytest.raises(ValueError, match="inside"):
        create_review_app(review, video, subs, allowed_roots=(root / "other",))
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source hashes"):
        create_review_app(review, video, subs, allowed_roots=(root,))


def test_bad_frame_labels_are_rejected(review_workspace):
    review, video, subs, root = review_workspace
    client = create_review_app(review, video, subs, allowed_roots=(root,)).test_client()
    revision = client.get("/api/review").get_json()["revision"]
    for indices in ([1], [0, 0], [[0]], [True]):
        result = client.put("/api/review", json={"revision": revision,
                            "queries": [{"query": "x", "relevant_segment_indices": indices}]})
        assert result.status_code == 400
