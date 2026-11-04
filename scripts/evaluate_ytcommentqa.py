"""Evaluate a transcript-search proxy on independently annotated YTCommentQA snippets.

The source labels describe answerability within an annotated snippet. This
script does not download videos, generate answers, or evaluate whole-video
answerability. The method is a heuristic: any retrieved segment means positive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_rag.schemas import VideoSegment
from video_rag.transcript_baseline import build_transcript_segments


def partition(video_id: str) -> str:
    return "sample" if int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big") % 5 == 0 else "other"


def evidence_segments(row: dict) -> list[VideoSegment]:
    """Use only captions within the inclusive annotated evidence ID range."""
    video = row.get("videoID")
    captions, evidence = row.get("caption"), row.get("evidence")
    if not isinstance(video, str) or not video or not isinstance(captions, list):
        raise ValueError("invalid video ID or captions")
    if (not isinstance(evidence, list) or len(evidence) != 2 or
            any(type(i) is not int for i in evidence) or evidence[0] > evidence[1]):
        raise ValueError("invalid evidence IDs")
    ids = [item.get("id") for item in captions if isinstance(item, dict)]
    if len(ids) != len(captions) or any(type(i) is not int for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("invalid caption IDs")
    if evidence[0] not in ids or evidence[1] not in ids:
        raise ValueError("evidence endpoints absent")
    chosen = sorted((c for c in captions if evidence[0] <= c["id"] <= evidence[1]), key=lambda c: c["id"])
    segments = []
    for cue in chosen:
        start, end, text = cue.get("start"), cue.get("end"), cue.get("text")
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end
                or not isinstance(text, str) or not text.strip()):
            raise ValueError("invalid evidence caption")
        segments.append(VideoSegment(segment_id=str(cue["id"]), video_id=video, source_path="not-downloaded",
                                     start_time=float(start), end_time=float(end), transcript=text))
    return segments


def evaluate(rows: dict, *, sample: str = "sample") -> dict:
    if sample not in {"sample", "other", "all"} or not isinstance(rows, dict):
        raise ValueError("Invalid data or sample selection")
    counts = Counter()
    videos = set()
    for row in rows.values():
        if not isinstance(row, dict):
            counts["invalid_row"] += 1
            continue
        video, labels, question = row.get("videoID"), row.get("answerability"), row.get("question")
        if not isinstance(video, str) or not video:
            counts["invalid_row"] += 1
            continue
        if sample != "all" and partition(video) != sample:
            continue
        counts["selected_rows"] += 1
        if labels not in ([0], [1], [2], [1, 2], [2, 1], [3]) or not isinstance(question, str) or not question.strip():
            counts["invalid_row"] += 1
            continue
        # Visual-only and combined-required examples are neither positives nor
        # negatives for this script-vs-unanswerable snippet-level comparison.
        if labels in ([1], [3]):
            counts["excluded_visual_or_combined_only"] += 1
            continue
        try:
            segments = evidence_segments(row)
        except (KeyError, TypeError, ValueError):
            counts["invalid_evidence_captions"] += 1
            continue
        predicted = bool(build_transcript_segments(segments).ask(question).evidence)
        positive = 2 in labels
        key = "tp" if positive and predicted else "fn" if positive else "fp" if predicted else "tn"
        counts[key] += 1
        videos.add(video)
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"selected_rows": counts["selected_rows"], "evaluated_rows": tp + fp + fn + tn,
            "evaluated_videos": len(videos), "excluded_visual_or_combined_only": counts["excluded_visual_or_combined_only"],
            "invalid_evidence_captions": counts["invalid_evidence_captions"], "invalid_rows": counts["invalid_row"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1,
            "unanswerable_returned_candidates": fp, "unanswerable_total": fp + tn}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Locally obtained YTCommentQA dataset.json")
    parser.add_argument("--sample", choices=("sample", "other", "all"), default="sample")
    args = parser.parse_args()
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
        result = evaluate(dataset, sample=args.sample)
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({"warning": "Annotated evidence snippets only; predicted answerability is proxied by "
                                 "retrieval returning a candidate, not evidence sufficiency or answer correctness. "
                                 "Video files were not used.", "sample": args.sample, "metrics": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
