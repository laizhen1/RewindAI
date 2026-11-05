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
