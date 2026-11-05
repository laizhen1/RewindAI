import pytest

from scripts.review_subtitles import create_review_app
from video_rag.transcript_baseline import parse_subtitles


@pytest.fixture
def review(tmp_path):
    video = tmp_path / 'sample.mp4'
    video.write_bytes(b'public synthetic video')
    subtitles = tmp_path / 'sample.srt'
    subtitles.write_text('1\n00:00:01,000 --> 00:00:03,000\nA model\n\n2\n00:00:04,000 --> 00:00:06,000\nOriginal phrase\n', encoding='utf-8')
    output = tmp_path / 'private' / 'corrections'
    app = create_review_app(video, subtitles, output, allowed_roots=(tmp_path / 'private',))
    return app.test_client(), video, subtitles, output


def test_local_review_and_immutable_exports(review):
    client, _video, subtitles, output = review
    original = subtitles.read_bytes()
    cues = client.get('/api/cues').get_json()['cues']
    assert len(cues) == 2 and cues[0]['text'] == 'A model'
    assert client.get('/video', headers={'Range': 'bytes=0-3'}).status_code == 206
    first = client.post('/api/export', json={'edits': {'1': 'Corrected <term>'}})
    second = client.post('/api/export', json={'edits': {'1': 'Corrected <term>'}})
    assert first.status_code == second.status_code == 201
    assert first.json['filename'] != second.json['filename']
    corrected = output / first.json['filename']
    parsed = parse_subtitles(corrected.read_text(encoding='utf-8'))
    assert [c.text for c in parsed] == ['A model', 'Corrected <term>']
    assert [(c.start_time, c.end_time) for c in parsed] == [(1, 3), (4, 6)]
    assert subtitles.read_bytes() == original
    assert client.get('/').headers['Cache-Control'] == 'no-store'


def test_rejects_invalid_edits_origins_and_changed_sources(review):
    client, _video, subtitles, output = review
    for edits in ({}, {'-1': 'x'}, {'01': 'x'}, {'2': 'x'}, {'0': ''}, {'0': 'x --> y'}, {'0': 3}):
        assert client.post('/api/export', json={'edits': edits}).status_code == 400
    assert client.get('/api/cues', headers={'Origin': 'https://outside.example'}).status_code == 403
    assert client.get('/api/cues', headers={'Host': 'outside.example'}).status_code == 403
    subtitles.write_text('changed', encoding='utf-8')
    assert client.post('/api/export', json={'edits': {'0': 'changed'}}).status_code == 409
    assert not output.exists()


def test_refuses_output_outside_private_root(tmp_path):
    video = tmp_path / 'sample.mp4'; video.write_bytes(b'video')
    subtitle = tmp_path / 'sample.srt'; subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nx\n')
    with pytest.raises(ValueError, match='Private evaluation outputs'):
        create_review_app(video, subtitle, tmp_path / 'public', allowed_roots=(tmp_path / 'private',))
