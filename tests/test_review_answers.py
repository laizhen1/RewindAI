import hashlib
import json

import pytest

from scripts.review_answers import create_review_app, reusable_judgments
from scripts.score_answer_review import summarize


@pytest.fixture
def review_app(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'synthetic video')
    review = tmp_path / 'private' / 'draft.json'
    review.parent.mkdir()
    report = {'status': 'unreviewed_answer_drafts', 'split': 'development', 'binding': 'matched',
              'retrieval_config': {'type': 'semantic_experimental', 'threshold': 0.4},
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
    data = client.get('/api/review').json
    assert len(data['rows']) == 2 and data['retrieval_config']['type'] == 'semantic_experimental'
    assert client.get('/video', headers={'Range': 'bytes=0-3'}).status_code == 206
    decisions = [{'answer': 'correct', 'evidence': 'supported'},
                 {'answer': 'abstain', 'evidence': 'not_applicable'}]
    result = client.post('/api/save', json={'confirm_human_review': True, 'judgments': decisions})
    assert result.status_code == 201 and result.json['metrics']['f1'] == 1.0
    assert result.json['metrics']['evidence_supported_answers'] == 1
    assert result.json['metrics']['evidence_support_rate_among_answers'] == 1
    saved = review.parent / result.json['filename']
    reviewed = json.loads(saved.read_text(encoding='utf-8'))
    assert summarize(reviewed)['tn'] == 1 and reviewed['retrieval_config']['threshold'] == 0.4
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


def test_failed_generation_allows_progress_but_cannot_be_scored(review_app):
    _client, video, review = review_app
    report = json.loads(review.read_text(encoding='utf-8'))
    report['rows'][0]['generation_error'] = 'Context limit exceeded'
    failed = review.parent / 'failed-draft.json'
    failed.write_text(json.dumps(report), encoding='utf-8')
    client = create_review_app(video, failed, allowed_roots=(review.parent,)).test_client()
    judgments = [{'answer': 'abstain', 'evidence': 'not_applicable'},
                 {'answer': 'abstain', 'evidence': 'not_applicable'}]
    assert client.post('/api/progress', json={'judgments': judgments}).status_code == 201
    response = client.post('/api/save', json={'confirm_human_review': True, 'judgments': judgments})
    assert response.status_code == 400 and 'Generation failed' in response.json['error']
    assert not list(review.parent.glob('reviewed-*.json'))


def test_reuse_suggests_only_identical_bound_outputs_and_never_changes_files(review_app):
    _client, video, review = review_app
    current = json.loads(review.read_text(encoding='utf-8'))
    current['input_sha256'].update(subtitles='a' * 64, model='b' * 64, annotations='c' * 64,
                                   annotation_binding='d' * 64)
    current['generation_config'] = {'prompt_version': 'v1'}
    for index, item in enumerate(current['rows']):
        item.update(question_id=f'q{index}', retrieval_abstained=False, model_question=item['question'],
                    evidence=[{'text': 'synthetic', 'model_excerpt': 'synthetic'}], latency_ms=5)
    previous = json.loads(json.dumps(current))
    previous['human_review_state'] = 'in_progress_not_verified'
    previous['rows'][0].update(human_answer_verdict='correct', human_evidence_verdict='supported', latency_ms=99)
    previous['rows'][1].update(human_answer_verdict='incorrect', human_evidence_verdict='unsupported')
    current['rows'][1]['answer'] = 'changed answer'
    expected = {'q0': {'answer': 'correct', 'evidence': 'supported'}}
    assert reusable_judgments(current, previous) == expected
    for key in ('generation_config', 'retrieval_config', 'input_sha256', 'split'):
        changed = json.loads(json.dumps(previous)); changed[key] = None
        assert reusable_judgments(current, changed) == {}
    missing = json.loads(json.dumps(current)); del missing['rows'][0]['model_question']
    assert reusable_judgments(missing, previous) == {}
    changed_evidence = json.loads(json.dumps(current)); changed_evidence['rows'][0]['evidence'][0]['model_excerpt'] = 'different'
    assert reusable_judgments(changed_evidence, previous) == {}
    failed = json.loads(json.dumps(current)); failed['rows'][0]['generation_error'] = 'failed'
    assert reusable_judgments(failed, previous) == {}
    duplicate = json.loads(json.dumps(current)); duplicate['rows'][1]['question_id'] = 'q0'
    assert reusable_judgments(duplicate, previous) == {}
    final_test = json.loads(json.dumps(current)); final_test['split'] = 'test'
    prior_test = json.loads(json.dumps(previous)); prior_test['split'] = 'test'
    assert reusable_judgments(final_test, prior_test) == {}
    review.write_text(json.dumps(current), encoding='utf-8')
    progress = review.parent / 'progress-synthetic.json'
    progress.write_text(json.dumps(previous), encoding='utf-8')
    before = (review.read_bytes(), progress.read_bytes())
    client = create_review_app(video, review, allowed_roots=(review.parent,)).test_client()
    response = client.get('/api/progress/reusable')
    assert response.status_code == 200
    assert response.json['judgments'] == [expected['q0'], None]
    assert (review.read_bytes(), progress.read_bytes()) == before


def test_rejects_changed_sources_and_wrong_video(review_app, tmp_path):
    client, video, review = review_app
    other = tmp_path / 'other.mp4'; other.write_bytes(b'wrong video')
    with pytest.raises(ValueError, match='bound'):
        create_review_app(other, review, allowed_roots=(review.parent,))
    video.write_bytes(b'changed video')
    assert client.get('/api/review').status_code == 409
    assert client.post('/api/save', json={'confirm_human_review': True, 'judgments': []}).status_code == 409


def test_partial_progress_can_be_reopened_without_scoring(review_app):
    client, video, review = review_app
    original = review.read_bytes()
    judgments = [{'answer': 'correct', 'evidence': 'supported'}, {'answer': '', 'evidence': ''}]
    result = client.post('/api/progress', json={'judgments': judgments})
    assert result.status_code == 201 and 'metrics' not in result.json
    progress = review.parent / result.json['filename']
    saved = json.loads(progress.read_text(encoding='utf-8'))
    assert saved['human_review_state'] == 'in_progress_not_verified'
    assert saved['rows'][0]['human_answer_verdict'] == 'correct'
    assert saved['rows'][1]['human_answer_verdict'] is None
    assert 'human_review_declaration' not in saved and review.read_bytes() == original
    reopened = create_review_app(video, progress, allowed_roots=(review.parent,)).test_client()
    assert reopened.get('/api/review').json['rows'][0]['human_answer_verdict'] == 'correct'
    with pytest.raises(ValueError, match='verdicts'):
        summarize(saved)


def test_progress_rejects_invalid_verdict_and_changed_input(review_app):
    client, video, review = review_app
    invalid = [{'answer': [], 'evidence': ''}, {'answer': '', 'evidence': ''}]
    assert client.post('/api/progress', json={'judgments': invalid}).status_code == 400
    assert not list(review.parent.glob('progress-*.json'))
    video.write_bytes(b'changed')
    assert client.post('/api/progress', json={'judgments': []}).status_code == 409


def test_restore_latest_matching_progress_without_changing_draft(review_app):
    client, _video, review = review_app
    original = review.read_bytes()
    assert client.get('/api/progress/latest').status_code == 404
    first = [{'answer': 'incorrect', 'evidence': 'unsupported'}, {'answer': '', 'evidence': ''}]
    second = [{'answer': 'correct', 'evidence': 'supported'}, {'answer': '', 'evidence': ''}]
    assert client.post('/api/progress', json={'judgments': first}).status_code == 201
    assert client.post('/api/progress', json={'judgments': second}).status_code == 201
    assert client.get('/api/progress/latest').json['judgments'] == second
    assert review.read_bytes() == original
    assert client.get('/api/progress/latest', headers={'Origin': 'https://outside.example'}).status_code == 403


def test_restore_ignores_unrelated_or_malformed_progress(review_app):
    client, _video, review = review_app
    judgments = [{'answer': 'correct', 'evidence': 'supported'}, {'answer': '', 'evidence': ''}]
    response = client.post('/api/progress', json={'judgments': judgments})
    matching = review.parent / response.json['filename']
    data = json.loads(matching.read_text(encoding='utf-8'))
    data['rows'][0]['question'] = 'different question'
    (review.parent / 'progress-unrelated.json').write_text(json.dumps(data), encoding='utf-8')
    (review.parent / 'progress-malformed.json').write_text('{', encoding='utf-8')
    assert client.get('/api/progress/latest').json['judgments'] == judgments
    matching.unlink()
    assert client.get('/api/progress/latest').status_code == 404


def test_resume_finds_progress_for_matching_video_only(review_app, tmp_path):
    from scripts.resume_answer_review import find_review

    client, video, review = review_app
    judgments = [{'answer': 'correct', 'evidence': 'supported'}, {'answer': '', 'evidence': ''}]
    saved = client.post('/api/progress', json={'judgments': judgments}).json['filename']
    other = tmp_path / 'other.mp4'
    other.write_bytes(b'other video')
    assert find_review(video, review.parent, allowed_roots=(review.parent,)).name == saved
    with pytest.raises(ValueError, match='No matching'):
        find_review(other, review.parent, allowed_roots=(review.parent,))


def test_resume_prefers_newer_regenerated_draft_over_older_progress(review_app):
    import os

    from scripts.resume_answer_review import find_review

    client, video, review = review_app
    judgments = [{'answer': 'correct', 'evidence': 'supported'}, {'answer': '', 'evidence': ''}]
    progress = review.parent / client.post('/api/progress', json={'judgments': judgments}).json['filename']
    draft = review.parent / 'answer-draft-new.json'
    draft.write_bytes(review.read_bytes())
    os.utime(progress, ns=(1_000_000_000, 1_000_000_000))
    os.utime(draft, ns=(2_000_000_000, 2_000_000_000))
    assert find_review(video, review.parent, allowed_roots=(review.parent,)) == draft


def test_final_review_rejects_not_applicable_evidence_for_a_wrong_answer(review_app):
    client, _video, review = review_app
    judgments = [{'answer': 'incorrect', 'evidence': 'not_applicable'},
                 {'answer': 'abstain', 'evidence': 'not_applicable'}]
    response = client.post('/api/save', json={'confirm_human_review': True, 'judgments': judgments})
    assert response.status_code == 400
    assert 'only valid for an abstention' in response.json['error']
    assert not list(review.parent.glob('reviewed-*.json'))
