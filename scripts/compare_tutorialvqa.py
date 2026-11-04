"""Compare keyword and semantic sentence overlap with paired video resampling.

Reads locally acquired TutorialVQA files and a locally installed ONNX model only.
The resulting ranges are exploratory and are not answer-accuracy estimates.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.evaluate_tutorialvqa import evaluate, partition_for_video
from video_rag.retrieval.onnx_semantic import OnnxSentenceEncoder, build_semantic_pipeline


def metrics(counts):
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def paired_bootstrap(units, samples=2000, seed=42):
    """Resample video-paired (keyword, semantic) TP/FP/FN and compute semantic minus keyword."""
    if len(units) < 2 or type(samples) is not int or not 1 <= samples <= 10000:
        raise ValueError("At least two paired videos and 1–10000 resamples are required")
    if any(len(unit) != 2 or any(len(counts) != 3 or any(type(x) is not int or x < 0 for x in counts)
                                     for counts in unit) for unit in units):
        raise ValueError("Expected paired nonnegative integer TP/FP/FN counts")
    def summed(chosen, method):
        return [sum(unit[method][i] for unit in chosen) for i in range(3)]
    def difference(chosen):
        a, b = metrics(summed(chosen, 0)), metrics(summed(chosen, 1))
        return {key: b[key] - a[key] for key in a}
    point = difference(units)
    rng = random.Random(seed)
    values = {key: [] for key in point}
    for _ in range(samples):
        delta = difference([rng.choice(units) for _ in units])
        for key in values:
            values[key].append(delta[key])
    intervals = {}
    for key, dist in values.items():
        dist.sort()
        intervals[key] = [dist[int((samples - 1) * .025)], dist[int((samples - 1) * .975)]]
    return {"samples": samples, "seed": seed, "semantic_minus_keyword": point,
            "percentile_95": intervals}


def compare(videos, questions, semantic_builder, samples=2000, seed=42):
    groups = defaultdict(list)
    for question in questions:
        groups[question["video_id"]].append(question)
    if len(groups) < 2:
        raise ValueError("At least two videos with questions are required")
    totals = [[0, 0, 0], [0, 0, 0]]
    units = []
    for vid in sorted(groups):
        a = evaluate(videos, groups[vid])
        b = evaluate(videos, groups[vid], builder=semantic_builder)
        pair = [[row[k] for k in ("tp", "fp", "fn")] for row in (a, b)]
        units.append(pair)
        for method in range(2):
            totals[method] = [x + y for x, y in zip(totals[method], pair[method], strict=True)]
    return {"questions": len(questions), "videos": len(groups),
            "keyword": dict(zip(("tp", "fp", "fn"), totals[0], strict=True)) | metrics(totals[0]),
            "semantic": dict(zip(("tp", "fp", "fn"), totals[1], strict=True)) | metrics(totals[1]),
            "paired_video_bootstrap": paired_bootstrap(units, samples, seed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--semantic-model", type=Path, required=True)
    parser.add_argument("--semantic-threshold", type=float, default=0.4)
    parser.add_argument("--video-partition", choices=("all", "development", "heldout"), default="heldout")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        videos = json.loads(args.videos.read_text(encoding="utf-8-sig"))
        questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
        if args.video_partition != "all":
            questions = [q for q in questions if partition_for_video(q["video_id"]) == args.video_partition]
        encoder = OnnxSentenceEncoder(args.semantic_model)
        builder = lambda segments: build_semantic_pipeline(segments, encoder, args.semantic_threshold)
        result = compare(videos, questions, builder, args.samples, args.seed)
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({"warning": "Exploratory paired video resampling, inclusive sentence-label overlap; "
                                 "not a measure of answer accuracy or a pristine held-out benchmark.",
                      "video_partition": args.video_partition, "semantic_threshold": args.semantic_threshold,
                      "metrics": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
