import sys
import types

import numpy as np
import pytest

from video_rag.retrieval.qwen_semantic import QUERY_INSTRUCTION, Qwen3EmbeddingEncoder
from video_rag.retrieval.semantic_models import load_semantic_encoder, semantic_model_files


def test_qwen_queries_use_instruction_documents_do_not(tmp_path, monkeypatch):
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def embed(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return [[1.0] + [0.0] * 1023 for _ in texts]

    monkeypatch.setitem(sys.modules, 'llama_cpp', types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setitem(sys.modules, 'llama_cpp.llama_cpp', types.SimpleNamespace(LLAMA_POOLING_TYPE_LAST=7))
    model = tmp_path / 'model.gguf'
    model.touch()
    encoder = load_semantic_encoder(model)
    docs = encoder.encode_documents(['document'])
    query = encoder.encode_query(' question ')
    assert calls[0]['pooling_type'] == 7
    assert calls[0]['n_gpu_layers'] == 0
    assert calls[0]['n_ctx'] == 512
    assert calls[1][0] == ['document']
    assert calls[2][0] == [f'Instruct: {QUERY_INSTRUCTION}\nQuery:question']
    assert calls[1][1] == {'normalize': True, 'truncate': True}
    assert calls[2][1] == {'normalize': True, 'truncate': False}
    assert docs.shape == (1, 1024) and query.shape == (1024,)
    assert np.linalg.norm(query) == pytest.approx(1)
    assert semantic_model_files(model) == {'semantic_model': model.resolve()}


def test_qwen_long_query_is_rejected_instead_of_truncated(tmp_path, monkeypatch):
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            pass

        def embed(self, texts, **kwargs):
            calls.append(kwargs)
            raise ValueError('Requested tokens (513) exceed batch size of 512')

    monkeypatch.setitem(sys.modules, 'llama_cpp', types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setitem(sys.modules, 'llama_cpp.llama_cpp', types.SimpleNamespace(LLAMA_POOLING_TYPE_LAST=7))
    model = tmp_path / 'model.gguf'; model.touch()
    encoder = Qwen3EmbeddingEncoder(model)
    with pytest.raises(RuntimeError, match='shorten'):
        encoder.encode_query('long question')
    assert calls == [{'normalize': True, 'truncate': False}]


def test_qwen_rejects_invalid_embedding_output():
    encoder = object.__new__(Qwen3EmbeddingEncoder)
    from collections import OrderedDict
    from threading import Lock
    encoder._lock = Lock()
    encoder._cache_lock = Lock()
    encoder._query_lock = Lock()
    encoder._last_query = None
    encoder._document_cache = OrderedDict()
    encoder._model = types.SimpleNamespace(embed=lambda *args, **kwargs: [[float('nan')] * 1024])
    with pytest.raises(ValueError, match='invalid vectors'):
        encoder.encode_documents(['document'])
    encoder._model = types.SimpleNamespace(embed=lambda *args, **kwargs: [[1.0] * 5])
    with pytest.raises(ValueError, match='unexpected shape'):
        encoder.encode_documents(['document'])
    with pytest.raises(ValueError, match='nonempty'):
        encoder.encode_query('  ')


def test_qwen_rerank_reuses_indexed_documents_and_handles_changed_text():
    from video_rag.retrieval.onnx_semantic import SemanticReranker, SemanticRetriever
    from video_rag.schemas import VideoSegment

    class Encoder:
        def __init__(self):
            self.document_calls = 0

        def encode_documents(self, texts):
            self.document_calls += 1
            return np.asarray([[1.0, 0.0] if 'apple' in t else [0.0, 1.0] for t in texts])

        def encode_query(self, query):
            return np.asarray([1.0, 0.0])

    encoder = Encoder()
    retriever = SemanticRetriever(encoder)
    original = VideoSegment('a', 'v', 'v.mp4', 0, 1, transcript='apple')
    retriever.build([original])
    reranker = SemanticReranker(encoder, retriever)
    assert reranker.score('apple', [original]) == [1.0]
    assert encoder.document_calls == 1
    changed = VideoSegment('a', 'v', 'v.mp4', 0, 1, transcript='orange')
    assert reranker.score('apple', [changed]) == [0.0]
    assert encoder.document_calls == 2
    retriever.build([changed])
    assert reranker.score('apple', [changed]) == [0.0]
    assert encoder.document_calls == 3


def test_document_cache_reuses_embeddings_without_losing_evicted_batch_items(tmp_path, monkeypatch):
    import video_rag.retrieval.qwen_semantic as module

    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            pass

        def embed(self, texts, **kwargs):
            calls.append(texts[:])
            return [[1.0] + [0.0] * 1023 for _ in texts]

    monkeypatch.setattr(module, 'DOCUMENT_CACHE_SIZE', 2)
    monkeypatch.setitem(sys.modules, 'llama_cpp', types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setitem(sys.modules, 'llama_cpp.llama_cpp', types.SimpleNamespace(LLAMA_POOLING_TYPE_LAST=7))
    model = tmp_path / 'model.gguf'
    model.touch()
    encoder = Qwen3EmbeddingEncoder(model)
    encoder.encode_documents(['old'])
    vectors = encoder.encode_documents(['new1', 'new2', 'old', 'new1'])
    assert vectors.shape == (4, 1024)
    assert calls == [['old'], ['new1', 'new2']]
    assert len(encoder._document_cache) == 2
    encoder.encode_documents(['old'])
    assert len(calls) == 2
    encoder.encode_documents(['new2'])
    assert calls[-1] == ['new2']


def test_query_cache_is_bounded_and_returned_vectors_do_not_mutate_it(tmp_path, monkeypatch):
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            pass

        def embed(self, texts, **kwargs):
            calls.append(texts[:])
            return [[1.0] + [0.0] * 1023 for _ in texts]

    monkeypatch.setitem(sys.modules, 'llama_cpp', types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setitem(sys.modules, 'llama_cpp.llama_cpp', types.SimpleNamespace(LLAMA_POOLING_TYPE_LAST=7))
    model = tmp_path / 'model.gguf'
    model.touch()
    encoder = Qwen3EmbeddingEncoder(model)
    result = encoder.encode_query(' question ')
    result[0] = 0.0
    assert encoder.encode_query('question')[0] == 1.0
    assert len(calls) == 1
    encoder.encode_query('another question')
    encoder.encode_query('question')
    assert len(calls) == 3


def test_qwen_pipeline_embeds_one_query_across_retrieval_and_reranking(tmp_path, monkeypatch):
    from video_rag.retrieval.onnx_semantic import build_semantic_pipeline
    from video_rag.schemas import VideoSegment

    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            pass

        def embed(self, texts, **kwargs):
            calls.append(texts[:])
            return [[1.0] + [0.0] * 1023 for _ in texts]

    monkeypatch.setitem(sys.modules, 'llama_cpp', types.SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setitem(sys.modules, 'llama_cpp.llama_cpp', types.SimpleNamespace(LLAMA_POOLING_TYPE_LAST=7))
    model = tmp_path / 'model.gguf'
    model.touch()
    pipeline = build_semantic_pipeline([VideoSegment('a', 'v', 'v.mp4', 0, 3, transcript='document')],
                                       Qwen3EmbeddingEncoder(model), 0.4)
    assert not pipeline.ask('question').abstained
    assert len(calls) == 2
    assert calls[0] == ['document'] and calls[1][0].endswith('Query:question')
