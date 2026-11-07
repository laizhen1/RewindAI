"""Evaluate transcript sentence retrieval against locally obtained TutorialVQA labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_rag.pipeline import VideoRAGPipeline
from video_rag.schemas import VideoSegment
from video_rag.transcript_baseline import build_transcript_segments


def partition_for_video(video_id: str) -> str:
    """Assign videos deterministically, keeping every question for a video together."""
    digest = hashlib.sha256(str(video_id).encode("utf-8")).digest()
    return "heldout" if int.from_bytes(digest[:8], "big") % 5 == 0 else "development"


def evaluate(videos: list[dict], questions: list[dict], context_sentences: int = 0,
             builder: Callable[[list[VideoSegment]], VideoRAGPipeline] = build_transcript_segments,
             bootstrap_samples: int = 0, seed: int = 42) -> dict:
    """Compute micro sentence overlap with inclusive annotated answer spans."""
    if type(context_sentences) is not int or not 0 <= context_sentences <= 5:
        raise ValueError("context_sentences must be an integer between 0 and 5")
    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 10000:
        raise ValueError("bootstrap_samples must be an integer between 0 and 10000")
    lookup = {v["video_id"]: v for v in videos}
    if len(lookup) != len(videos) or not questions:
        raise ValueError("Duplicate video IDs or empty question set")
    pipelines = {}
    per_video: dict[str, list[int]] = {}
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
            pipelines[vid] = builder(segments)
        answer = pipelines[vid].ask(row["question"])
        predicted = [int(e.segment.segment_id) for e in answer.evidence]
        reference = set(range(start, end + 1))
        found = {index for hit in predicted
                 for index in range(max(0, hit - context_sentences),
                                    min(len(transcript), hit + context_sentences + 1))}
        counts = [len(found & reference), len(found - reference), len(reference - found)]
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
        total = per_video.setdefault(vid, [0, 0, 0])
        for index, count in enumerate(counts):
            total[index] += count
        abstained += int(answer.abstained)
        top1_overlap += int(bool(predicted) and predicted[0] in reference)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    result = {"question_count": len(questions), "video_count": len(pipelines), "context_sentences": context_sentences,
              "tp": tp, "fp": fp, "fn": fn,
              "precision": precision, "recall": recall, "f1": f1, "top1_overlap_count": top1_overlap,
              "top1_overlap_rate": top1_overlap / len(questions), "abstained": abstained}
    if bootstrap_samples:
        if len(per_video) < 2:
            raise ValueError("Video bootstrap requires at least two videos")
        rng = random.Random(seed)
        units = list(per_video.values())
        distributions = {"precision": [], "recall": [], "f1": []}
        for _ in range(bootstrap_samples):
            sampled = [rng.choice(units) for _ in units]
            a, b, c = (sum(unit[i] for unit in sampled) for i in range(3))
            values = {"precision": a / (a + b) if a + b else None,
                      "recall": a / (a + c) if a + c else None,
                      "f1": 2 * a / (2 * a + b + c) if 2 * a + b + c else None}
            for key, value in values.items():
                if value is not None:
                    distributions[key].append(value)
        intervals = {}
        for key, values in distributions.items():
            values.sort()
            intervals[key] = ([values[int((len(values) - 1) * 0.025)], values[int((len(values) - 1) * 0.975)]]
                              if values else None)
        result["video_bootstrap"] = {"samples": bootstrap_samples, "seed": seed, "percentile_95": intervals,
                                     "defined_samples": {key: len(values) for key, values in distributions.items()},
                                     "warning": "Intervals exclude samples where the metric is undefined."}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Locally obtained TutorialVQA videos.json")
    parser.add_argument("--questions", type=Path, required=True, help="Locally obtained TutorialVQA split JSON")
    parser.add_argument("--context-sentences", type=int, default=0,
                        help="Adjacent transcript sentences on each side of a hit (0–5, default 0)")
    parser.add_argument("--video-partition", choices=("all", "development", "heldout"), default="all",
                        help="Deterministic video-ID partition; questions must come from the desired source split")
    parser.add_argument("--semantic-model", type=Path, help="Optional local English ONNX model directory")
    parser.add_argument("--semantic-threshold", type=float, default=0.4,
                        help="Uncalibrated cosine cutoff for semantic mode (default 0.4)")
    parser.add_argument("--bootstrap-samples", type=int, default=0,
                        help="Video-level bootstrap resamples for exploratory 95%% intervals (0–10000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        videos = json.loads(args.videos.read_text(encoding="utf-8-sig"))
        questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
        if args.video_partition != "all":
            questions = [row for row in questions if partition_for_video(row["video_id"]) == args.video_partition]
        builder = build_transcript_segments
        if args.semantic_model:
            from video_rag.retrieval.onnx_semantic import (
                OnnxSentenceEncoder,
                build_semantic_pipeline,
            )

            encoder = OnnxSentenceEncoder(args.semantic_model)
            builder = lambda segments: build_semantic_pipeline(segments, encoder, args.semantic_threshold)
        result = evaluate(videos, questions, context_sentences=args.context_sentences, builder=builder,
                          bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({"warning": "Sentence-index experiment, not video time or answer accuracy; "
                      "labels assumed inclusive.", "video_partition": args.video_partition,
                      "retrieval": "semantic_experimental" if args.semantic_model else "keyword",
                      "semantic_threshold": args.semantic_threshold if args.semantic_model else None,
                      "metrics": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
