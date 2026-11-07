from types import SimpleNamespace

import pytest

from scripts.evaluate_ytcommentqa import partition
from scripts.evaluate_ytcommentqa_local_answers import (
    BalancedExperimentGenerator,
    score,
    select_cases,
)


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


def test_public_answer_cli_loads_qwen_gguf_and_fingerprints_it(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import evaluate_ytcommentqa_local_answers as module
    from video_rag.retrieval import onnx_semantic, semantic_models

    dataset = tmp_path / 'dataset.json'
    dataset.write_text('{}', encoding='utf-8')
    answer_model = tmp_path / 'answer.gguf'
    answer_model.write_bytes(b'answer weights')
    embedding = tmp_path / 'embedding.gguf'
    embedding.write_bytes(b'embedding weights')
    target = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', ['eval', '--dataset', str(dataset), '--model', str(answer_model),
                                      '--semantic-model', str(embedding), '--output', str(target), '--per-class', '1'])
    monkeypatch.setattr(module, 'private_output', lambda path: path)
    monkeypatch.setattr(module, 'select_cases', lambda *a: [('x', True, {'videoID': 'public'})])
    monkeypatch.setattr(module, 'LocalAnswerGenerator', lambda path: SimpleNamespace(metadata=dict))
    sentinel = object()
    monkeypatch.setattr(semantic_models, 'load_semantic_encoder', lambda path: sentinel)
    monkeypatch.setattr(onnx_semantic, 'build_semantic_pipeline',
                        lambda segments, encoder, threshold, generator: (encoder, threshold, generator))

    def fake_score(cases, generator, builder, progress=None):
        encoder, threshold, used_generator = builder([], generator)
        assert encoder is sentinel and threshold == 0.4 and used_generator is generator
        return {'single_model': {}}

    monkeypatch.setattr(module, 'score', fake_score)
    assert module.main() == 0
    report = json.loads(target.read_text(encoding='utf-8'))
    assert len(report['input_sha256']['semantic_model']) == 64
    assert 'semantic_tokenizer' not in report['input_sha256']


def test_answer_experiment_progress_exposes_only_counts(monkeypatch):
    class Pipeline:
        def ask(self, question):
            return SimpleNamespace(evidence=[object()], answer='red')

    monkeypatch.setattr('scripts.evaluate_ytcommentqa_local_answers.evidence_segments', lambda row: row)
    counts = []
    cases = [('a', True, {'question': 'private draft text'}), ('b', False, {'question': 'other'})]
    score(cases, object(), lambda segments, generator: Pipeline(), progress=counts.append)
    assert counts == [1, 2]


def test_balanced_experiment_does_not_score_truncated_answers():
    from threading import Lock

    generator = object.__new__(BalancedExperimentGenerator)
    generator._lock = Lock()
    generator._model = SimpleNamespace(create_chat_completion=lambda **kwargs: {
        'choices': [{'message': {'content': 'unfinished'}, 'finish_reason': 'length'}]})
    with pytest.raises(RuntimeError, match='truncated'):
        generator.generate('question', [SimpleNamespace(transcript='synthetic evidence')])
