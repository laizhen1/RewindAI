import json
import sys
from types import SimpleNamespace

import pytest

from scripts.evaluate_retrieval import load_questions, metrics_with_breakdown, rerank_candidates


def test_reranker_compares_identical_candidates_and_keeps_ties_stable():
    segments = {key: SimpleNamespace(segment_id=key) for key in ('a', 'b', 'c')}

    class Reranker:
        def score(self, question, candidates):
            assert question == 'question'
            assert [item.segment_id for item in candidates] == ['a', 'b', 'c']
            return [0.1, 0.9, 0.9]

    assert rerank_candidates('question', ['a', 'b', 'c'], segments, Reranker()) == ['b', 'c', 'a']
    assert rerank_candidates('question', [], segments, Reranker()) == []
    class BadReranker:
        def score(self, question, candidates):
            return [float('nan')] * len(candidates)
    with pytest.raises(ValueError, match='finite'):
        rerank_candidates('question', ['a'], segments, BadReranker())


def test_question_type_breakdown_uses_requested_cutoff_and_all_relevant_labels():
    questions = [{'question_id': 'q', 'question_type': 'audio', 'relevant_segment_ids': ['a', 'b']}]
    result = metrics_with_breakdown({'q': ['a', 'x']}, questions, cutoffs=(1, 5))
    assert result['overall']['recall@5'] == 0.5
    assert result['overall']['hit@5'] == 1
    assert result['overall']['mrr@5'] == 1
    assert result['question_count'] == result['by_question_type']['audio']['question_count'] == 1
    assert result['by_question_type']['audio']['recall@5'] == result['overall']['recall@5']
    assert 'recall@10' not in result['overall']


def test_question_loader_rejects_duplicate_or_unknown_relevance(tmp_path):
    path = tmp_path / 'questions.jsonl'
    row = {'question_id': 'q', 'question': 'Why?', 'answer': 'A', 'question_type': 'audio',
           'relevant_segment_ids': ['a']}
    path.write_text(json.dumps(row) + '\n', encoding='utf-8')
    assert load_questions(path, {'a'}) == [row]
    for labels in (['a', 'a'], ['missing'], [], [{}]):
        path.write_text(json.dumps({**row, 'relevant_segment_ids': labels}) + '\n', encoding='utf-8')
        with pytest.raises(ValueError):
            load_questions(path, {'a'})


def test_cli_reports_embedding_before_and_after_reranking_on_same_candidates(tmp_path, monkeypatch):
    from scripts import evaluate_retrieval as module
    from video_rag.schemas import SearchHit, VideoSegment

    questions = tmp_path / 'questions.jsonl'
    questions.write_text(json.dumps({'question_id': 'q', 'question': 'question', 'answer': 'answer',
                                    'question_type': 'audio', 'relevant_segment_ids': ['a', 'b']}) + '\n', encoding='utf-8')
    segments = [VideoSegment(key, 'v', 'synthetic.mp4', i, i + 1, transcript='synthetic')
                for i, key in enumerate(('a', 'b', 'x'))]
    class Retriever:
        def __init__(self, *args, **kwargs):
            pass
        def build(self, values):
            assert len(values) == 3
        def search(self, question, top_k):
            return [SearchHit(key, 1.0, 'synthetic', rank) for rank, key in enumerate(('x', 'a', 'b'), 1)]
    class Reranker:
        def __init__(self, model, *, device_map, unload_after_score):
            assert device_map == 'cpu'
        def score(self, question, candidates):
            return [0.1 if item.segment_id == 'x' else 0.9 for item in candidates]
    config = SimpleNamespace(models=SimpleNamespace(text_embedding='text', clip='clip', reranker='reranker'),
                             retrieval=SimpleNamespace(rrf_k=60))
    monkeypatch.setattr(module, 'load_config', lambda path: config)
    monkeypatch.setattr(module, 'load_segments', lambda path: segments)
    monkeypatch.setattr(module, 'validate_manifest', lambda **kwargs: {'schema_version': 2})
    monkeypatch.setattr(module, 'InMemoryLexicalRetriever', Retriever)
    monkeypatch.setattr(module, 'QwenTextRetriever', Retriever)
    monkeypatch.setattr(module, 'ClipVisionRetriever', Retriever)
    monkeypatch.setattr(module, 'Qwen3Reranker', Reranker)
    output, csv_output = tmp_path / 'result.json', tmp_path / 'result.csv'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--questions', str(questions), '--top-k', '5',
                                     '--with-reranker', '--output', str(output), '--csv-output', str(csv_output)])
    module.main()
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['device'] == 'cpu' and report['cutoffs'] == [1, 5]
    assert report['index_manifest'] == {'schema_version': 2} and report['reranker_model'] == 'reranker'
    assert report['routes']['embedding']['overall']['mrr@5'] == 0.5
    assert report['routes']['embedding+reranker']['overall']['mrr@5'] == 1
    assert report['routes']['embedding']['overall']['recall@5'] == 1
    assert report['routes']['embedding+reranker']['overall']['recall@5'] == 1
    assert 'all_rrf+reranker' in report['routes']
    header = csv_output.read_text(encoding='utf-8').splitlines()[0]
    assert 'recall@5' in header and 'hit@5' in header and 'mrr@5' in header and 'recall@10' not in header


def test_reranker_cpu_selection_does_not_choose_gpu_dtype(monkeypatch):
    from video_rag.adapters.qwen import Qwen3Reranker

    seen = {}
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [1]
        def convert_tokens_to_ids(self, value):
            return 1
    class TokenizerFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Tokenizer()
    class Model:
        def eval(self):
            return self
    class ModelFactory:
        @staticmethod
        def from_pretrained(model, **kwargs):
            seen.update(kwargs)
            return Model()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        bfloat16='gpu_dtype', float32='cpu_dtype', cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=ModelFactory, AutoTokenizer=TokenizerFactory))
    Qwen3Reranker('synthetic', device_map='cpu', local_files_only=True)._load()
    assert seen['device_map'] == 'cpu' and seen['torch_dtype'] == 'cpu_dtype' and seen['local_files_only'] is True
