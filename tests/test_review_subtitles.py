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


def test_metrics_only_use_explicitly_reviewed_cues(review):
    client, _video, subtitles, output = review
    before = subtitles.read_bytes()
    payload = {'confirm_reviewed': True, 'reviewed': [0],
               'edits': {'0': 'A large model', '1': 'A completely different sentence'}}
    result = client.post('/api/metrics', json=payload)
    assert result.status_code == 200
    metrics = result.json['metrics']
    assert metrics == {'reviewed_cues': 1, 'reference_tokens': 3, 'substitutions': 0,
                       'deletions': 1, 'insertions': 0, 'token_errors': 1, 'token_error_rate': 1 / 3}
    assert subtitles.read_bytes() == before and not output.exists()
    assert 'not independently verified' in result.json['warning']
    assert client.post('/api/metrics', json={**payload, 'confirm_reviewed': False}).status_code == 400
    assert client.post('/api/metrics', json={**payload, 'reviewed': [0, 0]}).status_code == 400
    assert client.post('/api/metrics', json={**payload, 'reviewed': [True]}).status_code == 400
    assert client.post('/api/metrics', json=payload, headers={'Origin': 'https://outside.example'}).status_code == 403
    subtitles.write_text('changed', encoding='utf-8')
    assert client.post('/api/metrics', json=payload).status_code == 409


def test_metrics_handle_insertions_substitutions_and_cjk_tokens():
    from video_rag.evaluation.subtitle_metrics import edit_counts, tokens

    assert edit_counts(tokens('a model'), tokens('a large model')) == (0, 0, 1)
    assert edit_counts(tokens('red model'), tokens('blue model')) == (1, 0, 0)
    assert tokens('ＡＢＣ，中文！') == ['abc', '中', '文']
    with pytest.raises(ValueError, match='too many'):
        edit_counts(['x'] * 201, ['y'])


def test_metrics_can_save_private_summary_without_subtitle_text(review):
    import json

    client, _video, _subtitles, output = review
    payload = {'confirm_reviewed': True, 'reviewed': [0], 'edits': {'0': 'A large model'}, 'save_report': True}
    response = client.post('/api/metrics', json=payload)
    assert response.status_code == 200
    target = output / response.json['filename']
    raw = target.read_text(encoding='utf-8')
    report = json.loads(raw)
    assert report['reviewed_cue_indices'] == [0]
    assert report['reference_text_included'] is False
    assert report['metrics']['token_error_rate'] == 1 / 3
    assert len(report['source_sha256']) == len(report['reference_sha256']) == 64
    assert 'A large model' not in raw and 'A model' not in raw
    assert client.post('/api/metrics', json={**payload, 'save_report': 'yes'}).status_code == 400
    assert len(list(output.glob('subtitle-metrics-*.json'))) == 1


def test_review_progress_can_be_resumed_only_for_matching_sources(review, tmp_path):
    client, video, subtitles, output = review
    payload = {'edits': {'0': 'A corrected\n  model'}, 'reviewed': [0, 1]}
    response = client.post('/api/progress', json=payload)
    assert response.status_code == 201 and (output / response.json['filename']).is_file()
    reopened = create_review_app(video, subtitles, output, allowed_roots=(tmp_path / 'private',)).test_client()
    saved = reopened.get('/api/progress/latest')
    assert saved.status_code == 200 and saved.json == payload
    other = tmp_path / 'other.mp4'
    other.write_bytes(b'a different video')
    other_client = create_review_app(other, subtitles, output, allowed_roots=(tmp_path / 'private',)).test_client()
    assert other_client.get('/api/progress/latest').status_code == 404
    subtitles.write_text('1\n00:00:01,000 --> 00:00:03,000\nChanged\n', encoding='utf-8')
    assert reopened.get('/api/progress/latest').status_code == 409
    changed = create_review_app(video, subtitles, output, allowed_roots=(tmp_path / 'private',)).test_client()
    assert changed.get('/api/progress/latest').status_code == 404


def test_review_progress_rejects_bad_or_empty_input_and_cross_origin(review):
    client, _video, _subtitles, output = review
    for payload in ({'edits': {}, 'reviewed': []}, {'edits': {'-1': 'x'}, 'reviewed': []},
                    {'edits': {}, 'reviewed': [True]}, {'edits': {}, 'reviewed': [0, 0]}):
        assert client.post('/api/progress', json=payload).status_code == 400
    assert client.post('/api/progress', json={'edits': {}, 'reviewed': [0]},
                       headers={'Origin': 'https://outside.example'}).status_code == 403
    assert not output.exists()


def test_progress_requires_matching_optional_alternative(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'synthetic video')
    subtitles = tmp_path / 'sub.srt'
    subtitles.write_text('1\n00:00:01,000 --> 00:00:03,000\nA model\n', encoding='utf-8')
    alternative = tmp_path / 'alternative.txt'
    alternative.write_text('A model', encoding='utf-8')
    output = tmp_path / 'private' / 'corrections'
    client = create_review_app(video, subtitles, output, alternative=alternative,
                               allowed_roots=(tmp_path / 'private',)).test_client()
    assert client.post('/api/progress', json={'edits': {}, 'reviewed': [0]}).status_code == 201
    same = create_review_app(video, subtitles, output, alternative=alternative,
                             allowed_roots=(tmp_path / 'private',)).test_client()
    assert same.get('/api/progress/latest').status_code == 200
    alternative.write_text('A changed model', encoding='utf-8')
    changed = create_review_app(video, subtitles, output, alternative=alternative,
                                allowed_roots=(tmp_path / 'private',)).test_client()
    assert changed.get('/api/progress/latest').status_code == 404
