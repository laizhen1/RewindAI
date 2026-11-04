from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from video_rag.evaluation.dataset_validation import read_jsonl, validate_questions
from video_rag.evaluation.private_output import check_immutable, private_output, write_immutable
from video_rag.evaluation.review_events import load_review_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Export accepted append-only review events as verified JSONL.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=Path("artifacts/annotations/review_events.jsonl"))
    parser.add_argument("--segments", type=Path, default=Path("artifacts/segments.supplement_a.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/annotations/questions.zh.verified.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation/verified_export.json"))
    args = parser.parse_args()
    args.output, args.report = private_output(args.output), private_output(args.report)
    if args.output == args.report:
        raise ValueError("Output and report must be different files")
    candidate_sha = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
    candidates = {item["question_id"]: item for item in read_jsonl(args.candidates)}
    _events, latest = load_review_events(
        args.events, candidate_sha256=candidate_sha, known_question_ids=set(candidates)
    )
    accepted = [latest[question_id]["annotation_after"] for question_id in candidates if latest.get(question_id, {}).get("action") == "accept"]
    segments = read_jsonl(args.segments)
    report = validate_questions(accepted, segments)
    if not report.valid:
        raise ValueError("verified export failed validation: " + "; ".join(report.errors))
    content = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in accepted)
    summary = {
        **report.to_dict(),
        "candidate_file": str(args.candidates),
        "candidate_file_sha256": candidate_sha,
        "review_event_file": str(args.events),
        "review_event_file_sha256": hashlib.sha256(args.events.read_bytes()).hexdigest() if args.events.exists() else None,
        "accepted": len(accepted),
        "rejected": sum(item.get("action") == "reject" for item in latest.values()),
        "pending": len(candidates) - len(latest),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "formal_metric_eligible": False,
        "eligibility_note": "Export alone does not establish independent review or a frozen video-disjoint test set.",
    }
    report_content = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    check_immutable(args.output, content)
    check_immutable(args.report, report_content)
    write_immutable(args.output, content)
    write_immutable(args.report, report_content)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
