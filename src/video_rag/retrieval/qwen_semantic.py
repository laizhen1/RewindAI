"""Local Qwen3-Embedding GGUF inference with instruction-aware queries."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import Lock

import numpy as np

QUERY_INSTRUCTION = 'Given a question about a lecture, retrieve relevant transcript passages that answer the question.'
DOCUMENT_CACHE_SIZE = 4096


class Qwen3EmbeddingEncoder:
    """Use last-token pooled Qwen3 embeddings locally on CPU; no model download."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path).expanduser().absolute()
        if not path.is_file() or path.suffix.lower() != '.gguf':
            raise ValueError('Provide an existing local Qwen3 embedding GGUF file')
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_cpp import LLAMA_POOLING_TYPE_LAST
        except ImportError as exc:
            raise RuntimeError('Qwen3 semantic search requires llama-cpp-python') from exc
        self._model = Llama(model_path=str(path), embedding=True, pooling_type=LLAMA_POOLING_TYPE_LAST,
                            n_ctx=512, n_batch=512, n_ubatch=512, n_gpu_layers=0, verbose=False)
        self._lock = Lock()
        self._cache_lock = Lock()
        self._query_lock = Lock()
        self._last_query: tuple[str, np.ndarray] | None = None
        self._document_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Provide at least one nonempty text')
        with self._lock:
            vectors = np.asarray(self._model.embed(texts, normalize=True, truncate=True), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] != 1024:
            raise ValueError('Qwen3 embedding model returned an unexpected shape')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.all(np.isfinite(vectors)) or np.any(norms == 0):
            raise ValueError('Qwen3 embedding model returned invalid vectors')
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Provide at least one nonempty text')
        # The bounded cache exists only for this encoder's lifetime. It never writes
        # course text or vectors to disk and is shared by index rebuilds.
        with self._cache_lock:
            existing = {text: self._document_cache[text] for text in texts if text in self._document_cache}
            missing = list(dict.fromkeys(text for text in texts if text not in existing))
            encoded = self._embed(missing) if missing else None
            new = {text: vector.copy() for text, vector in zip(missing, encoded, strict=True)} if encoded is not None else {}
            output = []
            for text in texts:
                if text in self._document_cache:
                    vector = self._document_cache.pop(text)
                elif text in existing:
                    vector = existing[text]
                else:
                    vector = new[text]
                self._document_cache[text] = vector
                output.append(vector)
                if len(self._document_cache) > DOCUMENT_CACHE_SIZE:
                    self._document_cache.popitem(last=False)
            return np.ascontiguousarray(np.stack(output), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError('A nonempty query is required')
        query = query.strip()
        # Retrieval and reranking use the same query. Retain only the latest
        # vector in memory, return copies, and never persist question text.
        with self._query_lock:
            if self._last_query is not None and self._last_query[0] == query:
                return self._last_query[1].copy()
            vector = self._embed([f'Instruct: {QUERY_INSTRUCTION}\nQuery:{query}'])[0]
            self._last_query = (query, vector.copy())
            return vector.copy()
