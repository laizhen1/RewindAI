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


def test_question_origin_is_counted_without_claiming_independence():
    generated = {**row(True, 'correct', 'supported'), 'annotation_source': 'model_generated_candidate_unverified'}
    result = summarize({'status': 'unreviewed_answer_drafts', 'rows': [generated]})
    assert result['model_generated_questions'] == 1 and result['question_origin_not_recorded'] == 0
