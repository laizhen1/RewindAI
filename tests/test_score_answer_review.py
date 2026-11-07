import pytest

from scripts.score_answer_review import summarize


def row(answerable, verdict, evidence='not_applicable'):
    return {'expected_answerable': answerable, 'human_answer_verdict': verdict,
            'human_evidence_verdict': evidence}


def test_answer_metrics_count_wrong_answer_as_fp_and_fn():
    report = {'status': 'unreviewed_answer_drafts', 'rows': [
        row(True, 'correct', 'supported'), row(True, 'incorrect', 'unsupported'),
        row(True, 'abstain'), row(False, 'incorrect', 'unsupported'), row(False, 'abstain'),
    ]}
    result = summarize(report)
    assert (result['tp'], result['fp'], result['fn'], result['tn']) == (1, 2, 2, 1)
    assert result['precision'] == result['recall'] == result['f1'] == pytest.approx(1/3)
    assert result['question_origin_not_recorded'] == 5 and result['model_generated_questions'] == 0
    assert result['answerable_questions'] == 3 and result['unanswerable_questions'] == 2
    assert result['answerable_abstentions'] == 1 and result['unanswerable_abstention_rate'] == .5
    assert result['answerable_abstention_rate'] == pytest.approx(1/3)
    assert result['evidence_supported_answers'] == 1 and result['evidence_unsupported_answers'] == 2
    assert result['evidence_support_rate_among_answers'] == pytest.approx(1/3)


def test_requires_complete_consistent_reviews_and_reserves_test():
    base = {'status': 'unreviewed_answer_drafts', 'rows': [row(True, 'correct', 'supported')]}
    assert summarize(base)['f1'] == 1
    for invalid in (row(True, None), row(False, 'correct', 'supported'), row(True, 'correct', 'unsupported'),
                    row(True, 'abstain', 'supported'), row('true', 'correct', 'supported')):
        with pytest.raises(ValueError):
            summarize({**base, 'rows': [invalid]})
    with pytest.raises(ValueError, match='reserved'):
        summarize({**base, 'split': 'test'})
    scores = summarize({**base, 'rows': [row(False, 'abstain')]})
    assert scores['precision'] is None and scores['recall'] is None and scores['f1'] is None
    assert scores['answerable_questions'] == 0 and scores['unanswerable_abstention_rate'] == 1
    assert scores['answerable_abstention_rate'] is None
    assert scores['evidence_support_rate_among_answers'] is None


def test_failed_generation_cannot_be_scored_as_an_abstention():
    failed = {**row(True, 'abstain'), 'generation_error': 'Context limit'}
    with pytest.raises(ValueError, match='Generation failed'):
        summarize({'status': 'unreviewed_answer_drafts', 'rows': [failed]})


def test_supported_evidence_does_not_imply_a_correct_answer():
    result = summarize({'status': 'unreviewed_answer_drafts',
                        'rows': [row(True, 'incorrect', 'supported')]})
    assert result['evidence_support_rate_among_answers'] == 1
    assert result['f1'] == 0


def test_question_type_metrics_keep_unlabeled_drafts_separate():
    result = summarize({'status': 'unreviewed_answer_drafts', 'rows': [
        {**row(True, 'correct', 'supported'), 'question_type': 'audio'},
        {**row(True, 'abstain'), 'question_type': 'visual'},
        row(False, 'abstain'),
    ]})
    groups = result['by_question_type']
    assert groups['audio']['f1'] == 1 and groups['audio']['question_count'] == 1
    assert groups['visual']['f1'] == 0 and groups['visual']['answerable_abstentions'] == 1
    assert groups['unlabeled']['f1'] is None and groups['unlabeled']['question_count'] == 1
    assert 'by_question_type' not in groups['audio']


def test_query_latency_reports_missing_and_nearest_rank_p95():
    rows = [{**row(True, 'correct', 'supported'), 'latency_ms': value} for value in range(1, 21)]
    rows.append(row(False, 'abstain'))
    latency = summarize({'status': 'unreviewed_answer_drafts', 'rows': rows})['query_latency']
    assert latency == {'count': 20, 'missing': 1, 'median_ms': 10.5, 'p95_ms': 19.0}
    empty = summarize({'status': 'unreviewed_answer_drafts', 'rows': [row(False, 'abstain')]})['query_latency']
    assert empty == {'count': 0, 'missing': 1, 'median_ms': None, 'p95_ms': None}
    for invalid in (-1, float('nan'), True, '3'):
        with pytest.raises(ValueError, match='latency'):
            summarize({'status': 'unreviewed_answer_drafts',
                       'rows': [{**row(False, 'abstain'), 'latency_ms': invalid}]})


def test_question_origin_is_counted_without_claiming_independence():
    generated = {**row(True, 'correct', 'supported'), 'annotation_source': 'model_generated_candidate_unverified'}
    result = summarize({'status': 'unreviewed_answer_drafts', 'rows': [generated]})
    assert result['model_generated_questions'] == 1 and result['question_origin_not_recorded'] == 0


def test_summary_preserves_retrieval_configuration(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import score_answer_review as module

    source, target = tmp_path / 'review.json', tmp_path / 'summary.json'
    source.write_text(json.dumps({'status': 'unreviewed_answer_drafts',
                                  'retrieval_config': {'type': 'semantic_experimental', 'threshold': 0.4},
                                  'rows': [row(True, 'correct', 'supported')]}), encoding='utf-8')
    monkeypatch.setattr(module, 'private_output', lambda path: path)
    monkeypatch.setattr(sys, 'argv', ['score', '--review', str(source), '--output', str(target), '--confirm-human-review'])
    assert module.main() == 0
    assert json.loads(target.read_text(encoding='utf-8'))['retrieval_config']['type'] == 'semantic_experimental'
