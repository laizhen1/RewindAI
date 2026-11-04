"""Convert accepted review events from one frozen video split to local time diagnostics.

The event chain is checked, but local records are not tamper-proof and do not
prove reviewer independence or label correctness. Output stays in private roots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.prepare_temporal_questions import write_output
from video_rag.evaluation.dataset_validation import read_jsonl, validate_questions
from video_rag.evaluation.provenance import bind_sources, fingerprint
from video_rag.evaluation.review_events import load_review_events


def clean(row):
    return {key: value for key, value in row.items() if key not in {"_line_number", "split"}}


def validate_manifest(manifest, questions_path):
    if not isinstance(manifest, dict) or manifest.get("output_sha256") != fingerprint(questions_path):
        raise ValueError("Frozen split manifest does not match the question file")


def convert(questions, segments, candidates, events_path, candidate_path, video_id, split, video, subtitles):
    if split not in {"development", "validation", "test"} or not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("A video ID and valid frozen split are required")
    if not questions or not candidates:
        raise ValueError("Questions and candidate records cannot be empty")
    candidate_ids = [item.get("question_id") for item in candidates]
    if any(not isinstance(q, str) or not q.strip() for q in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate question IDs must be unique nonempty strings")
    if any(row.get("verification_status") != "verified" for row in questions):
        raise ValueError("All frozen questions must have accepted review status")
    validation = validate_questions(questions, segments, require_verified_splits=True)
    if not validation.valid:
        raise ValueError("Frozen annotations failed validation: " + "; ".join(validation.errors))
    _, latest = load_review_events(events_path, candidate_sha256=fingerprint(candidate_path),
                                   known_question_ids=set(candidate_ids))
    for row in questions:
        event = latest.get(row["question_id"])
        if not event or event.get("action") != "accept" or not isinstance(event.get("annotation_after"), dict):
            raise ValueError("Each question must match its latest accepted review event")
        behavior = event.get("review_behavior")
        if (not isinstance(behavior, dict) or behavior.get("video_played") is not True
                or not behavior.get("playback_ranges")):
            raise ValueError("An accepted review must declare video playback")
        if behavior.get("smoke_simulated") is True:
            raise ValueError("Simulated smoke reviews cannot be used as human annotations")
        if (event.get("event_id") != row.get("review_event_id") or event.get("reviewer_id") != row.get("reviewer_id")
                or event.get("reviewed_at") != row.get("reviewed_at")):
            raise ValueError("Question review provenance differs from the event")
        if clean(row) != clean(event["annotation_after"]):
            raise ValueError("Frozen question differs from the accepted review record")
    selected = [q for q in questions if q["video_id"] == video_id and q.get("split") == split]
    if not selected:
        raise ValueError("No accepted questions match that video and split")
    video_path = Path(video).resolve()
    related_segments = [s for s in segments if s.get("video_id") == video_id]
    if not related_segments or any(Path(str(s.get("source_path", ""))).resolve() != video_path for s in related_segments):
        raise ValueError("The supplied video does not match the annotated segment source")
    output = []
    for q in selected:
        item = {"question_id": q["question_id"], "question": q["question"],
                "question_type": q["question_type"], "expected_answerable": q["answerable"]}
        if q["answerable"]:
            item.update({"evidence_start_seconds": q["evidence_start"], "evidence_end_seconds": q["evidence_end"],
                         "expected_answer": q["answer"]})
        else:
            item["checked_scope"] = "Review record declares full-video coverage"
        output.append(item)
    data = {"status": "accepted_review_events_independence_not_established", "video_id": video_id, "split": split,
            "notes": "Local review events and the split manifest are not tamper-proof or proof of independence. "
                     "Video content at review time is not authenticated by this conversion.",
            "formal_metric_eligible": False, "questions": output}
    bind_sources(data, Path(video), Path(subtitles))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "segments", "candidates", "events", "manifest", "video", "subtitles", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--split", choices=("development", "validation", "test"), required=True)
    args = parser.parse_args()
    try:
        names = ("questions", "segments", "candidates", "events", "manifest")
        hashes = {name: fingerprint(getattr(args, name)) for name in names}
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        validate_manifest(manifest, args.questions)
        questions, segments, candidates = (read_jsonl(p) for p in (args.questions, args.segments, args.candidates))
        data = convert(questions, segments, candidates, args.events, args.candidates,
                       args.video_id, args.split, args.video, args.subtitles)
        if hashes != {name: fingerprint(getattr(args, name)) for name in names}:
            raise ValueError("Review inputs changed during conversion")
        data["review_inputs_sha256"] = hashes
        write_output(args.output, data, (ROOT / "artifacts", ROOT / "local_data"))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Conversion failed: {exc}\n")
    print(f"Wrote {len(data['questions'])} review-linked questions locally; independence and correctness remain to be checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
