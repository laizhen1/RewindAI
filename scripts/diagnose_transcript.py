"""Run a small, explicitly provisional timestamp-retrieval diagnostic."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_rag.transcript_baseline import build_transcript_pipeline


def overlap(hit: dict, start: float, end: float) -> bool:
    """Intervals touching at a single boundary do not overlap."""
    return hit["start_time"] < end and hit["end_time"] > start


def coverage(hit: dict, start: float, end: float) -> float:
    """Fraction of the reference interval covered by a single retrieved hit."""
    return max(0.0, min(hit["end_time"], end) - max(hit["start_time"], start)) / (end - start)


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
        result = pipeline.ask(question).to_dict()
        hits = result["evidence"]
        row = {
            "question": question,
            "expected_answerable": answerable,
            "abstained": result["abstained"],
            "returned_ranges": [[hit["start_time"], hit["end_time"]] for hit in hits],
        }
        if answerable:
            row.update({
                "first_hit_overlaps": bool(hits and overlap(hits[0], start, end)),
                "any_hit_overlaps": any(overlap(hit, start, end) for hit in hits),
                "first_hit_reference_coverage": coverage(hits[0], start, end) if hits else 0.0,
                "best_hit_reference_coverage": max((coverage(hit, start, end) for hit in hits), default=0.0),
            })
        else:
            row["correct_abstention"] = bool(result["abstained"])
        results.append(row)
    return results


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
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else
          0.0 if precision is not None and recall is not None else None)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "reference_coverage_threshold": min_coverage,
            "question_count": len(results)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    args = parser.parse_args()
    try:
        data = json.loads(args.questions.read_text(encoding="utf-8-sig"))
        results = diagnose(build_transcript_pipeline(args.video, args.subtitles), data["questions"])
        summary = metrics(results, args.min_coverage)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Diagnostic failed: {exc}\n")
    print(json.dumps({"warning": "Depends on label quality; not semantic accuracy", "metrics": summary,
                      "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

