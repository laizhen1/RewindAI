import hashlib
import json

import pytest

from scripts.review_fusion_comparison import create_app


@pytest.fixture
def sources(tmp_path):
    video = tmp_path / 'sample.mp4'; video.write_bytes(b'fake public video')
    root = tmp_path / 'private'; root.mkdir()
    path = root / 'report.json'
    report = {'split': 'development', 'input_sha256': {'video': hashlib.sha256(video.read_bytes()).hexdigest()},
              'transcript_retrieval': 'semantic_experimental', 'comparison': {'results': [
                  {'question': 'Where?', 'reference_range': [1, 3],
                   'transcript': {'returned_ranges': [[1, 3]]},
                   'fused': {'returned_ranges': [[4, 6]], 'returned_sources': [['visual']]}}]}}
    path.write_text(json.dumps(report), encoding='utf-8')
    return video, root, path, report


def test_local_comparison_is_video_bound_and_serves_ranges(sources):
    video, root, path, report = sources
    client = create_app(video, path, allowed_roots=(root,)).test_client()
    response = client.get('/api/comparison')
    assert response.status_code == 200 and response.json['rows'][0]['reference_range'] == [1, 3]
    assert response.json['rows'][0]['fused']['returned_sources'] == [['visual']]
    assert response.headers['Cache-Control'] == 'no-store'
    assert client.get('/video', headers={'Range': 'bytes=0-3'}).status_code == 206
    assert client.get('/api/comparison', headers={'Host': 'evil.example'}).status_code == 403
    assert client.get('/api/comparison', headers={'Origin': 'https://evil.example'}).status_code == 403
    path.write_text(json.dumps({**report, 'split': 'validation'}), encoding='utf-8')
    assert client.get('/api/comparison').status_code == 409


def test_rejects_aggregate_reports_wrong_video_and_test_split(sources):
    video, root, path, report = sources
    path.write_text(json.dumps({**report, 'comparison': {}}), encoding='utf-8')
    with pytest.raises(ValueError, match='detailed'):
        create_app(video, path, allowed_roots=(root,))
    path.write_text(json.dumps({**report, 'split': 'test'}), encoding='utf-8')
    with pytest.raises(ValueError, match='frozen-test'):
        create_app(video, path, allowed_roots=(root,))
    frozen = {**report, 'split': 'test', 'test_declaration': 'frozen_test_declared_not_independently_verified'}
    path.write_text(json.dumps(frozen), encoding='utf-8')
    assert create_app(video, path, allowed_roots=(root,)).test_client().get('/api/comparison').json['split'] == 'test'
    path.write_text(json.dumps(report), encoding='utf-8')
    video.write_bytes(b'changed')
    with pytest.raises(ValueError, match='does not match'):
        create_app(video, path, allowed_roots=(root,))
