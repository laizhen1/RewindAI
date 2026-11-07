import pytest

from scripts.evaluate_local_judge import evaluate


def row(answer_verdict='correct', evidence_verdict='supported', **changes):
    item = {'expected_answerable': True, 'human_answer_verdict': answer_verdict,
            'human_evidence_verdict': evidence_verdict, 'model_question': 'Synthetic question',
            'answer': 'Synthetic answer', 'evidence': [{'model_excerpt': 'Synthetic excerpt'}],
            'retrieval_abstained': False}
    return {**item, **changes}


def test_judge_agreement_excludes_missing_inputs_and_counts_invalid_separately():
    report = {'status': 'unreviewed_answer_drafts', 'split': 'development', 'rows': [
        row(), row('incorrect', 'unsupported'), row('abstain', 'not_applicable', evidence=[])]}
    predictions = iter(['SUPPORTED', 'not a valid label'])
    result = evaluate(report, lambda question, answer, excerpts: next(predictions))
    assert result['eligible'] == 2 and result['excluded_missing_model_input'] == 1
    assert result['valid'] == 1 and result['invalid'] == 1 and result['agreed'] == 1
    assert result['agreement_valid'] == 1 and result['agreement_all'] == 0.5
    assert result['cases'][1]['judge_suggestion'] == 'INVALID'
    assert 'Synthetic question' not in str(result)


def test_judge_evaluator_rejects_test_data_and_inconsistent_reviews():
    report = {'status': 'unreviewed_answer_drafts', 'split': 'test', 'rows': [row()]}
    with pytest.raises(ValueError, match='development or validation'):
        evaluate(report, lambda *args: 'SUPPORTED')
    report['split'] = 'development'
    report['rows'] = [row('correct', 'unsupported')]
    with pytest.raises(ValueError, match='supported evidence'):
        evaluate(report, lambda *args: 'SUPPORTED')


def test_judge_cli_saves_only_local_diagnostic_without_source_text(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import evaluate_local_judge as module

    review = tmp_path / 'review.json'
    review.write_text(json.dumps({'status': 'unreviewed_answer_drafts', 'split': 'development',
                                  'rows': [row()]}), encoding='utf-8')
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'synthetic model')
    output = tmp_path / 'result.json'
    monkeypatch.setattr(module, 'private_output', lambda path: path.resolve())
    monkeypatch.setattr(module, 'local_judge', lambda path: lambda *args: 'SUPPORTED')
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--review', str(review), '--model', str(model),
                                      '--output', str(output), '--confirm-reviewed'])
    assert module.main() == 0
    content = output.read_text(encoding='utf-8')
    result = json.loads(content)
    assert result['result']['agreed'] == 1
    assert result['status'] == 'experimental_judge_agreement_not_ground_truth'
    assert 'Synthetic question' not in content and 'Synthetic excerpt' not in content
    with pytest.raises(SystemExit):
        module.main()  # A previous report must never be overwritten.
