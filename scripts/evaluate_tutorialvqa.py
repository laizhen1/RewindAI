"""Evaluate transcript sentence retrieval against locally obtained TutorialVQA labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_rag.schemas import VideoSegment
from video_rag.transcript_baseline import build_transcript_segments


def partition_for_video(video_id: str) -> str:
    """Assign videos deterministically, keeping every question for a video together."""
    digest = hashlib.sha256(str(video_id).encode("utf-8")).digest()
    return "heldout" if int.from_bytes(digest[:8], "big") % 5 == 0 else "development"


def evaluate(videos: list[dict], questions: list[dict], context_sentences: int = 0) -> dict:
    """Compute micro sentence overlap with inclusive annotated answer spans."""
    if type(context_sentences) is not int or not 0 <= context_sentences <= 5:
        raise ValueError("context_sentences must be an integer between 0 and 5")
    lookup = {v["video_id"]: v for v in videos}
    if len(lookup) != len(videos) or not questions:
        raise ValueError("Duplicate video IDs or empty question set")
    pipelines = {}
    tp = fp = fn = abstained = top1_overlap = 0
    for row in questions:
        vid = row["video_id"]
        transcript = lookup[vid]["transcript"]
        start, end = row["answer_start"], row["answer_end"]
        if (not isinstance(row["question"], str) or not row["question"].strip() or
                type(start) is not int or type(end) is not int or not 0 <= start <= end < len(transcript)):
            raise ValueError("Invalid question or answer sentence range")
        if vid not in pipelines:
            segments = [VideoSegment(segment_id=str(index), video_id=str(vid), source_path="local-only",
                                     start_time=float(index), end_time=float(index + 1), transcript=text)
                        for index, text in enumerate(transcript)]
            pipelines[vid] = build_transcript_segments(segments)
        answer = pipelines[vid].ask(row["question"])
        predicted = [int(e.segment.segment_id) for e in answer.evidence]
        reference = set(range(start, end + 1))
        found = {index for hit in predicted
                 for index in range(max(0, hit - context_sentences),
                                    min(len(transcript), hit + context_sentences + 1))}
        tp += len(found & reference)
        fp += len(found - reference)
        fn += len(reference - found)
        abstained += int(answer.abstained)
        top1_overlap += int(bool(predicted) and predicted[0] in reference)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"question_count": len(questions), "video_count": len(pipelines), "context_sentences": context_sentences,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0, "top1_overlap_count": top1_overlap,
            "top1_overlap_rate": top1_overlap / len(questions), "abstained": abstained}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Locally obtained TutorialVQA videos.json")
    parser.add_argument("--questions", type=Path, required=True, help="Locally obtained TutorialVQA split JSON")
    parser.add_argument("--context-sentences", type=int, default=0,
                        help="Adjacent transcript sentences on each side of a hit (0–5, default 0)")
    parser.add_argument("--video-partition", choices=("all", "development", "heldout"), default="all",
                        help="Deterministic video-ID partition; questions must come from the desired source split")
    args = parser.parse_args()
    try:
        videos = json.loads(args.videos.read_text(encoding="utf-8-sig"))
        questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
        if args.video_partition != "all":
            questions = [row for row in questions if partition_for_video(row["video_id"]) == args.video_partition]
        result = evaluate(videos, questions, context_sentences=args.context_sentences)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({"warning": "Sentence-index experiment, not video time or answer accuracy; "
                      "labels assumed inclusive.", "video_partition": args.video_partition,
                      "metrics": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
