"""Evaluate visual ranking against locally reviewed labels for sampled frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_rag.retrieval.onnx_clip import ClipFrameRetriever, OnnxClipEncoder
from video_rag.transcript_baseline import segments_from_subtitles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(labels: dict, video: Path, subtitles: Path) -> None:
    if labels.get("video_sha256") != sha256_file(video) or labels.get("subtitles_sha256") != sha256_file(subtitles):
        raise ValueError("Label source hashes do not match this video and subtitle file")


def evaluate(retriever, segments, labels: dict, *, allow_unreviewed: bool = False) -> dict:
    if not isinstance(labels, dict) or not isinstance(labels.get("queries"), list) or not labels["queries"]:
        raise ValueError("Expected a nonempty queries list")
    status = labels.get("review_status")
    if status != "reviewed" and not allow_unreviewed:
        raise ValueError("Labels are not reviewed; use --allow-unreviewed only for provisional diagnostics")
    expected = labels.get("segment_midpoints_seconds")
    actual = [(segment.start_time + segment.end_time) / 2 for segment in segments]
    if (not isinstance(expected, list) or len(expected) != len(actual) or
            any(type(value) not in (int, float) or not math.isfinite(value) or abs(value - target) > 0.001
                for value, target in zip(expected, actual, strict=True))):
        raise ValueError("Label frame times do not match the current segmentation")
    lookup = {segment.segment_id: index for index, segment in enumerate(segments)}
    tp = fp = fn = tn = absent = false_positive_absent = 0
    rows = []
    for item in labels["queries"]:
        query = item.get("query")
        relevant = item.get("relevant_segment_indices")
        if (not isinstance(query, str) or not query.strip() or not isinstance(relevant, list) or
                any(type(i) is not int or not 0 <= i < len(segments) for i in relevant) or
                len(relevant) != len(set(relevant))):
            raise ValueError("Invalid visual query or relevant segment indices")
        hits = retriever.search(query, 3)
        predicted = {lookup[hit.segment_id] for hit in hits}
        truth = set(relevant)
        tp += len(predicted & truth)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
        tn += len(segments) - len(predicted | truth)
        absent += int(not truth)
        false_positive_absent += int(not truth and bool(predicted))
        rows.append({"query": query, "predicted_indices": [lookup[hit.segment_id] for hit in hits],
                     "scores": [hit.score for hit in hits], "relevant_indices": relevant})
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall) if precision is not None and recall is not None
          and precision + recall else 0.0 if precision is not None and recall is not None else None)
    return {"review_status": status, "queries": len(rows), "sampled_frames": len(segments),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1,
            "absent_queries": absent, "absent_queries_returning_candidates": false_positive_absent, "results": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    try:
        segments = segments_from_subtitles(args.video, args.subtitles)
        labels = json.loads(args.labels.read_text(encoding="utf-8-sig"))
        validate_sources(labels, args.video, args.subtitles)
        retriever = ClipFrameRetriever(OnnxClipEncoder(args.model_dir))
        retriever.build(segments)
        results = evaluate(retriever, segments, labels, allow_unreviewed=args.allow_unreviewed)
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({"warning": "Frame-level ranking metrics depend on label quality; scores are not probabilities. "
                                 "Only sampled midpoint frames are evaluated, not entire videos.",
                      "metrics": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
