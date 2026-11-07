"""Run a small, explicitly provisional timestamp-retrieval diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_rag.evaluation.provenance import verify_sources
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


def overlap(hit: dict, start: float, end: float) -> bool:
    """Intervals touching at a single boundary do not overlap."""
    return hit["start_time"] < end and hit["end_time"] > start


def coverage(hit: dict, start: float, end: float) -> float:
    """Fraction of the reference interval covered by a single retrieved hit."""
    return max(0.0, min(hit["end_time"], end) - max(hit["start_time"], start)) / (end - start)


def union_coverage(hits: list[dict], start: float, end: float) -> float:
    """Reference interval covered by the union of returned hits; never double count overlap."""
    intervals = sorted((max(start, hit["start_time"]), min(end, hit["end_time"]))
                       for hit in hits if overlap(hit, start, end))
    covered = 0.0
    previous_end = start
    for left, right in intervals:
        covered += max(0.0, right - max(left, previous_end))
        previous_end = max(previous_end, right)
    return covered / (end - start)


def coverage_feasibility(questions: list[dict], segments: list, min_coverage: float = 0.5) -> dict:
    """Check whether a single indexed segment can meet each reference interval.

    Segments must belong to the same video as the questions. This checks only
    interval geometry, not whether a segment contains relevant information.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    ranges = [{"start_time": s.start_time, "end_time": s.end_time} for s in segments]
    answerable = unattainable = 0
    for item in questions:
        expected = item.get("expected_answerable", True)
        if not isinstance(expected, bool):
            raise TypeError("expected_answerable must be boolean")
        if not expected:
            continue
        start, end = float(item["evidence_start_seconds"]), float(item["evidence_end_seconds"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("Invalid question or reference evidence range")
        answerable += 1
        if not any(coverage(segment, start, end) >= min_coverage for segment in ranges):
            unattainable += 1
    return {"answerable_questions": answerable, "structurally_unattainable": unattainable,
            "reference_coverage_threshold": min_coverage}


def diagnose(pipeline, questions: list[dict]) -> list[dict]:
    results = []
    for item in questions:
        question = item["question"]
        answerable = item.get("expected_answerable", True)
        if not isinstance(answerable, bool) or not isinstance(question, str) or not question.strip():
            raise ValueError("Invalid question or reference evidence range")
        if answerable:
            start, end = float(item["evidence_start_seconds"]), float(item["evidence_end_seconds"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("Invalid question or reference evidence range")
        question_type = item.get("question_type")
        if question_type is not None and (not isinstance(question_type, str) or not question_type.strip() or len(question_type) > 80):
            raise ValueError("question_type must be a nonempty string of at most 80 characters")
        started = perf_counter()
        result = pipeline.ask(question).to_dict()
        latency_ms = (perf_counter() - started) * 1000
        hits = result["evidence"]
        row = {
            "question": question,
            "expected_answerable": answerable,
            "abstained": result["abstained"],
            "query_latency_ms": latency_ms,
            "returned_ranges": [[hit["start_time"], hit["end_time"]] for hit in hits],
        }
        if hits and all(isinstance(hit.get("sources"), list) for hit in hits):
            row["returned_sources"] = [hit["sources"] for hit in hits]
        if question_type is not None:
            row["question_type"] = question_type.strip()
        if answerable:
            row.update({
                "first_hit_overlaps": bool(hits and overlap(hits[0], start, end)),
                "any_hit_overlaps": any(overlap(hit, start, end) for hit in hits),
                "first_hit_reference_coverage": coverage(hits[0], start, end) if hits else 0.0,
                "best_hit_reference_coverage": max((coverage(hit, start, end) for hit in hits), default=0.0),
                "returned_union_reference_coverage": union_coverage(hits, start, end),
            })
        else:
            row["correct_abstention"] = bool(result["abstained"])
        results.append(row)
    return results


def latency_summary(results: list[dict]) -> dict:
    """Summarize sequential ask-and-serialization wall time, including first query."""
    values = sorted(row["query_latency_ms"] for row in results)
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    return {"count": len(values), "mean_ms": sum(values) / len(values),
            "median_ms": median(values), "p95_ms": values[math.ceil(.95 * len(values)) - 1]}


def metrics(results: list[dict], min_coverage: float = 0.5) -> dict:
    """Question-level micro metrics: wrong answers count as both FP and FN.

    A correct answerable prediction must return a top-1 segment covering at
    least min_coverage of the reference interval. This measures temporal
    evidence, not semantic correctness or recall across all relevant segments.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    tp = fp = fn = tn = 0
    for row in results:
        answerable = row["expected_answerable"]
        predicted = not row["abstained"]
        correct = predicted and answerable and row["first_hit_reference_coverage"] >= min_coverage
        if correct:
            tp += 1
        else:
            fp += int(predicted)
            fn += int(answerable)
            tn += int(not predicted and not answerable)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "reference_coverage_threshold": min_coverage,
            "question_count": len(results)}


def operating_metrics(results: list[dict], min_coverage: float = 0.5) -> dict:
    """Separate answerable temporal hits from rejection of unanswerable questions."""
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    positive = [row for row in results if row["expected_answerable"]]
    negative = [row for row in results if not row["expected_answerable"]]
    first = sum(not row["abstained"] and row["first_hit_reference_coverage"] >= min_coverage for row in positive)
    any_returned = sum(not row["abstained"] and row["best_hit_reference_coverage"] >= min_coverage for row in positive)
    union_available = all("returned_union_reference_coverage" in row for row in positive)
    union_returned = (sum(not row["abstained"] and row["returned_union_reference_coverage"] >= min_coverage
                          for row in positive) if union_available else None)
    rejected_negative = sum(row["abstained"] for row in negative)
    rejected_positive = sum(row["abstained"] for row in positive)
    return {
        "reference_coverage_threshold": min_coverage,
        "answerable_questions": len(positive),
        "top1_temporal_hits": first,
        "top1_temporal_hit_rate": first / len(positive) if positive else None,
        "any_returned_temporal_hits": any_returned,
        "any_returned_temporal_hit_rate": any_returned / len(positive) if positive else None,
        "union_returned_temporal_hits": union_returned,
        "union_returned_temporal_hit_rate": union_returned / len(positive) if positive and union_available else None,
        "answerable_abstentions": rejected_positive,
        "answerable_abstention_rate": rejected_positive / len(positive) if positive else None,
        "unanswerable_questions": len(negative),
        "unanswerable_abstentions": rejected_negative,
        "unanswerable_abstention_rate": rejected_negative / len(negative) if negative else None,
        "unanswerable_returned_candidates": len(negative) - rejected_negative,
        "unanswerable_candidate_rate": (len(negative) - rejected_negative) / len(negative) if negative else None,
    }


def metrics_by_question_type(results: list[dict], min_coverage: float = 0.5) -> dict:
    """Only report groups actually labeled; unlabeled rows remain explicit."""
    groups: dict[str, list[dict]] = {}
    for row in results:
        groups.setdefault(row.get("question_type", "unlabeled"), []).append(row)
    return {name: operating_metrics(group, min_coverage) for name, group in sorted(groups.items())}


def comparison_by_question_type(results: list[dict], min_coverage: float = 0.5) -> dict:
    """Return evidence metrics and operating counts for each observed type."""
    groups: dict[str, list[dict]] = {}
    for row in results:
        groups.setdefault(row.get("question_type", "unlabeled"), []).append(row)
    return {name: {"metrics": metrics(group, min_coverage),
                   "operating": operating_metrics(group, min_coverage)}
            for name, group in sorted(groups.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args()
    try:
        data = json.loads(args.questions.read_text(encoding="utf-8-sig"))
        source_binding = verify_sources(data, args.video, args.subtitles)
        segments = segments_from_subtitles(args.video, args.subtitles)
        feasibility = coverage_feasibility(data["questions"], segments, args.min_coverage)
        results = diagnose(build_transcript_segments(segments), data["questions"])
        summary = metrics(results, args.min_coverage)
        verify_sources(data, args.video, args.subtitles)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Diagnostic failed: {exc}\n")
    print(json.dumps({"warning": "Depends on label quality; temporal overlap is not answer correctness", "metrics": summary, "source_binding": source_binding,
                      "coverage_feasibility": feasibility,
                      "operating_metrics": operating_metrics(results, args.min_coverage), "query_latency": latency_summary(results),
                      "by_question_type": metrics_by_question_type(results, args.min_coverage), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
