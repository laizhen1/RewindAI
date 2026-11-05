from types import SimpleNamespace

import pytest

from scripts.evaluate_ytcommentqa import partition
from scripts.evaluate_ytcommentqa_local_answers import score, select_cases


def test_balanced_selection_is_stable_and_filters_invalid():
    video = next(str(i) for i in range(100) if partition(str(i)) == 'sample')
    base = {'videoID': video, 'question': 'what color?', 'caption': [{'id': 1, 'start': 0, 'end': 1, 'text': 'red car'}], 'evidence': [1, 1]}
    data = {'a': {**base, 'answerability': [2]}, 'b': {**base, 'answerability': [0]},
            'c': {**base, 'answerability': [1]}, 'bad': {**base, 'answerability': [False]}}
    selected = select_cases(data, 1)
    assert {key for key, _, _ in selected} == {'a', 'b'}
    assert select_cases(dict(reversed(list(data.items()))), 1) == selected
    with pytest.raises(ValueError, match='Not enough'):
        select_cases(data, 2)
    with pytest.raises(ValueError, match='Not enough'):
        select_cases(data, 1, 'other')


def test_literal_refusal_proxy_is_not_retrieval_abstention(monkeypatch):
    class Pipeline:
        def __init__(self, row):
            self.row = row

        def ask(self, question):
            return SimpleNamespace(evidence=[object()], answer='Insufficient evidence in the provided excerpts.' if question == 'no' else 'red')

    monkeypatch.setattr('scripts.evaluate_ytcommentqa_local_answers.evidence_segments', lambda row: row)
    builder = lambda segments, generator: Pipeline(segments)
    result = score([('a', True, {'question': 'yes'}), ('b', False, {'question': 'no'})], object(), builder)
    assert result['retrieval']['fp'] == 1 and result['retrieval']['tp'] == 1
    assert result['exact_refusal_proxy']['tn'] == 1 and result['exact_refusal_proxy']['tp'] == 1
