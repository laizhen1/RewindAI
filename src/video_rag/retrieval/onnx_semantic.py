"""Optional local English sentence embeddings using a pre-downloaded ONNX model."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from video_rag.schemas import SearchHit, VideoSegment


class OnnxSentenceEncoder:
    """Mean-pool MiniLM token embeddings; CPU inference, no network access."""

    def __init__(self, model_dir: str | Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.session = ort.InferenceSession(
            str(directory / "onnx" / "model_quantized.onnx"), providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("At least one text is required")
        encoded = self.tokenizer.encode_batch(texts)
        ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        masks = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        types = np.asarray([item.type_ids for item in encoded], dtype=np.int64)
        output = self.session.run(None, {"input_ids": ids, "attention_mask": masks,
                                         "token_type_ids": types})[0]
        pooled = (output * masks[:, :, None]).sum(axis=1) / np.maximum(masks.sum(axis=1, keepdims=True), 1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.all(np.isfinite(pooled)):
            raise ValueError("Model produced invalid embeddings")
        return np.ascontiguousarray(pooled / norms, dtype=np.float32)


class SemanticRetriever:
    name = "local_semantic"

    def __init__(self, encoder: OnnxSentenceEncoder) -> None:
        self.encoder = encoder
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None

    def build(self, segments: list[VideoSegment]) -> None:
        items = [segment for segment in segments if segment.transcript.strip()]
        if not items:
            self.ids = []
            self.vectors = None
            return
        self.ids = [segment.segment_id for segment in items]
        self.vectors = np.concatenate([
            self.encoder.encode([segment.transcript for segment in items[i:i + 32]])
            for i in range(0, len(items), 32)
        ])

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if self.vectors is None:
            raise RuntimeError("Index is not built")
        if top_k <= 0 or not query.strip():
            return []
        scores = self.vectors @ self.encoder.encode([query])[0]
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.ids[i]))[:top_k]
        return [SearchHit(self.ids[i], float(scores[i]), self.name, rank)
                for rank, i in enumerate(order, 1)]


class SemanticReranker:
    def __init__(self, encoder: OnnxSentenceEncoder) -> None:
        self.encoder = encoder

    def score(self, query: str, segments: list[VideoSegment]) -> list[float]:
        vectors = self.encoder.encode([query, *[segment.transcript for segment in segments]])
        return [float(value) for value in vectors[1:] @ vectors[0]]


def build_semantic_pipeline(segments: list[VideoSegment], encoder: OnnxSentenceEncoder, threshold: float, *, generator=None):
    """Create an experimental local transcript index with a shared encoder."""
    from video_rag.adapters import EvidenceGenerator
    from video_rag.pipeline import VideoRAGPipeline

    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Semantic threshold must be finite and between -1 and 1")
    pipeline = VideoRAGPipeline(retrievers=[SemanticRetriever(encoder)], reranker=SemanticReranker(encoder),
                                generator=generator if generator is not None else EvidenceGenerator(), minimum_rerank_score=threshold)
    pipeline.build(segments)
    return pipeline
