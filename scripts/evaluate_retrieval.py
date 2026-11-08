from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from video_rag.adapters import Qwen3Reranker
from video_rag.config import load_config
from video_rag.evaluation import evaluate_retrieval
from video_rag.index_manifest import validate_manifest
from video_rag.retrieval import (
    ClipVisionRetriever,
    InMemoryLexicalRetriever,
    QwenTextRetriever,
    reciprocal_rank_fusion,
)
from video_rag.storage import load_segments

REQUIRED_FIELDS = {
    "question_id",
    "question",
    "answer",
    "relevant_segment_ids",
    "question_type",
}
ROUTES = {
    "bm25": ("bm25",),
    "embedding": ("embedding",),
    "clip": ("clip",),
    "bm25+embedding": ("bm25", "embedding"),
    "bm25+clip": ("bm25", "clip"),
    "embedding+clip": ("embedding", "clip"),
    "all_rrf": ("bm25", "embedding", "clip"),
}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def load_questions(path: Path, known_segment_ids: set[str]) -> list[dict]:
    questions: list[dict] = []
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"line {line_number}: each question must be an object")
            missing = REQUIRED_FIELDS - item.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")
            question_id = item["question_id"]
            if (not isinstance(question_id, str) or not question_id.strip()
                    or question_id != question_id.strip() or question_id in identifiers):
                raise ValueError(f"line {line_number}: duplicate or empty question_id")
            relevant = item["relevant_segment_ids"]
            if (not isinstance(relevant, list) or not relevant
                    or any(not isinstance(value, str) or not value for value in relevant)
                    or len(set(relevant)) != len(relevant)):
                raise ValueError(f"line {line_number}: relevant_segment_ids must be non-empty")
            if (not isinstance(item["question"], str) or not item["question"].strip()
                    or not isinstance(item["question_type"], str) or not item["question_type"].strip()):
                raise ValueError(f"line {line_number}: question and question_type must be non-empty strings")
            unknown = set(relevant) - known_segment_ids
            if unknown:
                raise ValueError(f"line {line_number}: unknown segments {sorted(unknown)}")
            identifiers.add(question_id)
            questions.append(item)
    if not questions:
        raise ValueError("question file must not be empty")
    return questions


def metrics_with_breakdown(
    predictions: dict[str, list[str]], questions: list[dict], *, cutoffs: tuple[int, ...] = (1, 5, 10)
) -> dict[str, object]:
    ground_truth = {
        item["question_id"]: item["relevant_segment_ids"] for item in questions
    }
    overall = evaluate_retrieval(predictions, ground_truth, cutoffs=cutoffs)
    by_type: dict[str, dict[str, float]] = {}
    types = sorted({item["question_type"] for item in questions})
    for question_type in types:
        subset = [item for item in questions if item["question_type"] == question_type]
        subset_truth = {
            item["question_id"]: item["relevant_segment_ids"] for item in subset
        }
        by_type[question_type] = {"question_count": len(subset),
                                  **evaluate_retrieval(predictions, subset_truth, cutoffs=cutoffs)}
    return {"question_count": len(questions), "overall": overall, "by_question_type": by_type}


def rerank_candidates(question: str, candidate_ids: list[str], segments: dict, reranker) -> list[str]:
    """Rerank only the same candidates; retain the input order on tied scores."""
    if not candidate_ids:
        return []
    candidates = [segments[value] for value in candidate_ids]
    scores = reranker.score(question, candidates)
    if len(scores) != len(candidates) or any(not math.isfinite(score) for score in scores):
        raise ValueError("Reranker must return one finite score per candidate")
    return [segment.segment_id for segment, _ in sorted(
        zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible retrieval ablations.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--segments", default=Path("artifacts/segments.jsonl"), type=Path)
    parser.add_argument("--index-dir", default=Path("artifacts/indexes"), type=Path)
    parser.add_argument("--config", default=Path("config.toml"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/evaluation/retrieval_ablation.json"), type=Path)
    parser.add_argument("--csv-output", default=Path("artifacts/evaluation/retrieval_ablation.csv"), type=Path)
    parser.add_argument("--device", default="cpu", help="Model device; CPU is the default")
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--with-reranker", action="store_true")
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    cutoffs = tuple(k for k in (1, 5, 10) if k <= args.top_k)
    config = load_config(args.config)
    segments = load_segments(args.segments)
    segment_by_id = {item.segment_id: item for item in segments}
    questions = load_questions(args.questions, set(segment_by_id))
    index_manifest = validate_manifest(segments_path=args.segments, index_dir=args.index_dir,
                                       text_model=config.models.text_embedding, clip_model=config.models.clip)
    retrievers = {
        "bm25": InMemoryLexicalRetriever(),
        "embedding": QwenTextRetriever(
            config.models.text_embedding, device=args.device, index_dir=args.index_dir
        ),
        "clip": ClipVisionRetriever(
            config.models.clip, device=args.device, index_dir=args.index_dir
        ),
    }
    for retriever in retrievers.values():
        retriever.build(segments)
        retriever.search("视频内容", 1)

    predictions = {name: {} for name in ROUTES}
    latency = {name: [] for name in ROUTES}
    for item in questions:
        question_id = item["question_id"]
        route_hits = {}
        route_latency = {}
        for name, retriever in retrievers.items():
            started = perf_counter()
            route_hits[name] = retriever.search(item["question"], args.top_k)
            route_latency[name] = (perf_counter() - started) * 1000
        for route_name, component_names in ROUTES.items():
            if len(component_names) == 1:
                ranked = route_hits[component_names[0]]
            else:
                ranked = reciprocal_rank_fusion(
                    [route_hits[name] for name in component_names],
                    k=config.retrieval.rrf_k,
                    top_k=args.top_k,
                )
            predictions[route_name][question_id] = [hit.segment_id for hit in ranked]
            latency[route_name].append(sum(route_latency[name] for name in component_names))
    if args.with_reranker:
        reranker = Qwen3Reranker(
            config.models.reranker, device_map="cpu" if args.device == "cpu" else "auto",
            unload_after_score=args.low_vram
        )
        first_candidate = next((segment_by_id[ids[0]] for ids in predictions["all_rrf"].values() if ids), None)
        if first_candidate is not None:
            reranker.score("video content", [first_candidate])
        for source in ("embedding", "all_rrf"):
            route = f"{source}+reranker"
            predictions[route] = {}
            latency[route] = []
            for index, item in enumerate(questions):
                question_id = item["question_id"]
                started = perf_counter()
                predictions[route][question_id] = rerank_candidates(
                    item["question"], predictions[source][question_id], segment_by_id, reranker)
                latency[route].append(latency[source][index] + (perf_counter() - started) * 1000)

    results = {}
    for name, route_predictions in predictions.items():
        route_metrics = metrics_with_breakdown(route_predictions, questions, cutoffs=cutoffs)
        timings = latency[name]
        route_metrics["latency_ms"] = {
            "mean": mean(timings),
            "median": median(timings),
            "p95": percentile(timings, 0.95),
        }
        results[name] = route_metrics

    report = {
        "dataset": str(args.questions),
        "question_count": len(questions),
        "top_k": args.top_k,
        "device": args.device,
        "index_manifest": index_manifest,
        "reranker_model": config.models.reranker if args.with_reranker else None,
        "warning": "Exploratory retrieval only: text routes and reranker use searchable_text, "
                   "which may contain visual descriptions. Labels and their completeness require human verification.",
        "latency_scope": "Sequential query timings sum component retrieval and optional reranking. "
                         "Fusion, index building, model loading and warmup are excluded, except that "
                         "low-vram reranking can reload the model per call. These are not user-facing latency.",
        "cutoffs": cutoffs,
        "metric_definitions": {
            "recall_at_k": "macro mean of retrieved relevant segments / labeled relevant segments",
            "hit_at_k": "fraction of questions with at least one labeled relevant segment in top K",
            "mrr_at_k": "mean reciprocal rank of the first labeled relevant segment within top K",
            "mrr": "mean reciprocal rank within the complete returned ranking",
        },
        "routes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        metric_columns = [f"{name}@{k}" for k in cutoffs for name in ("recall", "hit", "mrr", "ndcg")]
        writer.writerow(("route", *metric_columns, "mrr", "mean_ms", "p95_ms"))
        for name, result in results.items():
            metrics = result["overall"]
            timings = result["latency_ms"]
            writer.writerow((name, *(metrics[column] for column in metric_columns), metrics["mrr"], timings["mean"], timings["p95"]))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
