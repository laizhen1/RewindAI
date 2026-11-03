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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    args = parser.parse_args()
    try:
        data = json.loads(args.questions.read_text(encoding="utf-8-sig"))
        results = diagnose(build_transcript_pipeline(args.video, args.subtitles), data["questions"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Diagnostic failed: {exc}\n")
    print(json.dumps({"warning": "Provisional diagnostic; not validated accuracy", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

