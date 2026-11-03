from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

from video_rag.retrieval.base import Generator, Reranker, Retriever
from video_rag.retrieval.fusion import reciprocal_rank_fusion
from video_rag.schemas import Answer, Evidence, VideoSegment


class VideoRAGPipeline:
    def __init__(
        self,
        *,
        retrievers: list[Retriever],
        reranker: Reranker,
        generator: Generator,
        recall_top_k: int = 20,
        fusion_top_k: int = 20,
        rerank_top_k: int = 3,
        rrf_k: int = 60,
        minimum_rerank_score: float = 0.20,
    ) -> None:
        if not retrievers:
            raise ValueError("At least one retriever is required")
        if min(recall_top_k, fusion_top_k, rerank_top_k, rrf_k) <= 0:
            raise ValueError("Top-k and RRF k values must be positive")
        self._retrievers = retrievers
        self._reranker = reranker
        self._generator = generator
        self._recall_top_k = recall_top_k
        self._fusion_top_k = fusion_top_k
        self._rerank_top_k = rerank_top_k
        self._rrf_k = rrf_k
        self._minimum_rerank_score = minimum_rerank_score
        self._segments: dict[str, VideoSegment] = {}

    def build(self, segments: Iterable[VideoSegment]) -> None:
        materialized = list(segments)
        identifiers = [segment.segment_id for segment in materialized]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("segment_id must be unique")
        self._segments = {segment.segment_id: segment for segment in materialized}
        for retriever in self._retrievers:
            retriever.build(materialized)

    def warmup(self, query: str = "video content") -> None:
        """Load retrieval models and execute one minimal query before serving traffic."""
        if not self._segments:
            raise RuntimeError("Pipeline index is empty; call build() first")
        for retriever in self._retrievers:
            retriever.search(query, 1)

    def ask(self, query: str) -> Answer:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not self._segments:
            raise RuntimeError("Pipeline index is empty; call build() first")

        started = perf_counter()
        result_lists = [
            retriever.search(query, self._recall_top_k)
            for retriever in self._retrievers
        ]
        recalled_at = perf_counter()
        fused = reciprocal_rank_fusion(
            result_lists,
            k=self._rrf_k,
            top_k=self._fusion_top_k,
        )
        candidates = [
            self._segments[hit.segment_id]
            for hit in fused
            if hit.segment_id in self._segments
        ]
        fused_score = {hit.segment_id: hit.score for hit in fused}
        fused_at = perf_counter()

        if not candidates:
            return self._abstention(started, recalled_at, fused_at)

        rerank_scores = self._reranker.score(query, candidates)
        if len(rerank_scores) != len(candidates):
            raise ValueError("Reranker must return one score per candidate")
        ranked = sorted(
            zip(candidates, rerank_scores, strict=True),
            key=lambda item: (-item[1], -fused_score[item[0].segment_id]),
        )[: self._rerank_top_k]
        reranked_at = perf_counter()

        if not ranked or ranked[0][1] < self._minimum_rerank_score:
            return self._abstention(started, recalled_at, fused_at, reranked_at)

        # A strong first match does not make weaker matches reliable evidence.
        # Apply the same relevance threshold to every returned segment and to
        # the material supplied to the answer generator.
        ranked = [(segment, score) for segment, score in ranked if score >= self._minimum_rerank_score]
        selected_segments = [item[0] for item in ranked]
        answer_text = self._generator.generate(query, selected_segments)
        finished = perf_counter()
        evidence = tuple(
            Evidence(
                segment=segment,
                fused_score=fused_score[segment.segment_id],
                rerank_score=score,
            )
            for segment, score in ranked
        )
        return Answer(
            answer=answer_text,
            evidence=evidence,
            latency_ms={
                "recall": (recalled_at - started) * 1000,
                "fusion": (fused_at - recalled_at) * 1000,
                "rerank": (reranked_at - fused_at) * 1000,
                "generation": (finished - reranked_at) * 1000,
                "total": (finished - started) * 1000,
            },
        )

    def video_path(self, video_id: str) -> str | None:
        for segment in self._segments.values():
            if segment.video_id == video_id:
                return segment.source_path
        return None

    def videos(self) -> list[dict]:
        """Describe indexed videos without exposing their local file paths."""
        videos: dict[str, dict] = {}
        for segment in self._segments.values():
            video = videos.setdefault(segment.video_id, {
                "id": segment.video_id,
                "title": Path(segment.source_path).name or segment.video_id,
                "duration": 0.0,
                "segment_count": 0,
                "available": Path(segment.source_path).is_file(),
            })
            video["duration"] = max(video["duration"], segment.end_time)
            video["segment_count"] += 1
        return sorted(videos.values(), key=lambda item: item["title"].casefold())

    def segment(self, segment_id: str) -> VideoSegment | None:
        """Return an indexed segment for server-side evidence validation."""
        return self._segments.get(segment_id)

    def _abstention(
        self,
        started: float,
        recalled_at: float,
        fused_at: float,
        reranked_at: float | None = None,
    ) -> Answer:
        finished = perf_counter()
        return Answer(
            answer="根据当前视频内容无法确定。",
            evidence=(),
            abstained=True,
            latency_ms={
                "recall": (recalled_at - started) * 1000,
                "fusion": (fused_at - recalled_at) * 1000,
                "rerank": 0.0 if reranked_at is None else (reranked_at - fused_at) * 1000,
                "generation": 0.0,
                "total": (finished - started) * 1000,
            },
        )

