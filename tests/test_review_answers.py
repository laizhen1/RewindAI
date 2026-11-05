import hashlib
import json

import pytest

from scripts.review_answers import create_review_app
from scripts.score_answer_review import summarize


@pytest.fixture
def review_app(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'synthetic video')
    review = tmp_path / 'private' / 'draft.json'
    review.parent.mkdir()
    report = {'status': 'unreviewed_answer_drafts', 'split': 'development', 'binding': 'matched',
              'input_sha256': {'video': hashlib.sha256(video.read_bytes()).hexdigest()}, 'rows': [
                  {'question': 'Color?', 'expected_answerable': True, 'expected_answer': 'red',
                   'answer': 'red', 'evidence': [], 'human_answer_verdict': None, 'human_evidence_verdict': None},
                  {'question': 'Year?', 'expected_answerable': False, 'answer': 'unknown', 'evidence': [],
                   'human_answer_verdict': None, 'human_evidence_verdict': None}]}
    review.write_text(json.dumps(report), encoding='utf-8')
    app = create_review_app(video, review, allowed_roots=(review.parent,))
    return app.test_client(), video, review


def test_review_saves_new_copy_and_local_metrics(review_app):
    client, _video, review = review_app
    original = review.read_bytes()
    assert len(client.get('/api/review').json['rows']) == 2
    assert client.get('/video', headers={'Range': 'bytes=0-3'}).status_code == 206
    decisions = [{'answer': 'correct', 'evidence': 'supported'},
                 {'answer': 'abstain', 'evidence': 'not_applicable'}]
    result = client.post('/api/save', json={'confirm_human_review': True, 'judgments': decisions})
    assert result.status_code == 201 and result.json['metrics']['f1'] == 1.0
    saved = review.parent / result.json['filename']
    assert summarize(json.loads(saved.read_text(encoding='utf-8')))['tn'] == 1
    assert review.read_bytes() == original
    assert client.get('/').headers['Cache-Control'] == 'no-store'


def test_requires_human_declaration_complete_consistent_verdicts_and_local_origin(review_app):
    client, _video, review = review_app
    decisions = [{'answer': 'correct', 'evidence': 'supported'},
                 {'answer': 'correct', 'evidence': 'supported'}]
    assert client.post('/api/save', json={'judgments': decisions}).status_code == 400
    assert client.post('/api/save', json={'confirm_human_review': True, 'judgments': decisions}).status_code == 400
    assert client.get('/api/review', headers={'Origin': 'https://outside.example'}).status_code == 403
    assert client.get('/api/review', headers={'Host': 'outside.example'}).status_code == 403
    assert len(list(review.parent.glob('reviewed-*.json'))) == 0


def test_rejects_changed_sources_and_wrong_video(review_app, tmp_path):
    client, video, review = review_app
    other = tmp_path / 'other.mp4'; other.write_bytes(b'wrong video')
    with pytest.raises(ValueError, match='bound'):
        create_review_app(other, review, allowed_roots=(review.parent,))
    video.write_bytes(b'changed video')
    assert client.get('/api/review').status_code == 409
    assert client.post('/api/save', json={'confirm_human_review': True, 'judgments': []}).status_code == 409
