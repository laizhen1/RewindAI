"""Locate annotated YTCommentQA windows within provided full transcripts.

This is transcript-only temporal overlap, not video understanding or answer
correctness. It does not download videos and evaluates only script-answerable
questions; snippet-level unanswerable labels are not whole-video negatives.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_ytcommentqa import evidence_segments, partition
from video_rag.schemas import TimedText, VideoSegment
from video_rag.transcript_baseline import build_transcript_segments, group_subtitle_cues


def full_transcript(row, *, allow_incomplete=False, segmentation="sentence"):
    """Represent each provided valid caption as one segment, preserving seconds."""
    if segmentation not in {"sentence", "workspace"}:
        raise ValueError("Invalid segmentation")
    reference = evidence_segments(row)
    video = row["videoID"]
    captions = row["caption"]
    segments = []
    skipped = 0
    for cue in captions:
        start, end, text = cue.get("start"), cue.get("end"), cue.get("text")
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end
                or not isinstance(text, str) or not text.strip()):
            if allow_incomplete and not (row["evidence"][0] <= cue["id"] <= row["evidence"][1]):
                skipped += 1
                continue
            raise ValueError("invalid full-transcript caption")
        segments.append(VideoSegment(segment_id=str(cue["id"]), video_id=video, source_path="not-downloaded",
                                     start_time=float(start), end_time=float(end), transcript=text))
    segments.sort(key=lambda segment: (segment.start_time, segment.end_time, segment.segment_id))
    if segmentation == "workspace":
        cues = [TimedText(s.start_time, s.end_time, s.transcript) for s in segments]
        groups = group_subtitle_cues(cues)
        segments = [VideoSegment(segment_id=f"{i:05d}", video_id=video, source_path="not-downloaded",
                                 start_time=g[0].start_time, end_time=max(c.end_time for c in g),
                                 transcript=" ".join(c.text for c in g)) for i, g in enumerate(groups)]
    return segments, (min(s.start_time for s in reference), max(s.end_time for s in reference)), skipped


def overlaps(segment, reference):
    return segment.start_time < reference[1] and segment.end_time > reference[0]


def reference_coverage(segments, reference):
    """Fraction of the reference window covered by the union of returned segments."""
    start, end = reference
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("Invalid reference window")
    spans = sorted((max(segment.start_time, start), min(segment.end_time, end)) for segment in segments
                   if overlaps(segment, reference))
    total = 0.0
    left = right = None
    for begin, finish in spans:
        if left is None:
            left, right = begin, finish
        elif begin <= right:
            right = max(right, finish)
        else:
            total += right - left
            left, right = begin, finish
    if left is not None:
        total += right - left
    return total / (end - start)


def evaluate(dataset, *, sample="sample", builder=build_transcript_segments, include_rows=False, allow_incomplete=False, segmentation="sentence"):
    if sample not in {"sample", "other", "all"} or not isinstance(dataset, dict):
        raise ValueError("Invalid dataset or sample")
    counts = Counter()
    videos, outcomes = set(), {}
    for row_id, row in dataset.items():
        if not isinstance(row, dict) or not isinstance(row.get("videoID"), str) or not row["videoID"]:
            counts["invalid_rows"] += 1
            continue
        if sample != "all" and partition(row["videoID"]) != sample:
            continue
        counts["selected_rows"] += 1
        labels, question = row.get("answerability"), row.get("question")
        if (not isinstance(labels, list) or any(type(value) is not int for value in labels)
                or labels not in ([0], [1], [2], [1, 2], [2, 1], [3])
                or not isinstance(question, str) or not question.strip()):
            counts["invalid_rows"] += 1
            continue
        if 2 not in labels:
            counts["excluded_other_labels"] += 1
            continue
        try:
            segments, ref, skipped = full_transcript(row, allow_incomplete=allow_incomplete, segmentation=segmentation)
        except (KeyError, TypeError, ValueError):
            counts["invalid_transcripts"] += 1
            continue
        counts["incomplete_transcripts"] += int(skipped > 0)
        counts["skipped_invalid_cues"] += skipped
        response = builder(segments).ask(question)
        hits = [evidence.segment for evidence in response.evidence][:3]
        top1 = bool(hits and overlaps(hits[0], ref))
        top3 = any(overlaps(hit, ref) for hit in hits)
        first3 = any(overlaps(segment, ref) for segment in segments[:3])
        top3_coverage = reference_coverage(hits, ref)
        first3_coverage = reference_coverage(segments[:3], ref)
        counts["top3_cover_half"] += int(top3_coverage >= 0.5)
        counts["first3_cover_half"] += int(first3_coverage >= 0.5)
        counts["sum_top3_coverage"] += top3_coverage
        counts["sum_first3_coverage"] += first3_coverage
        counts["overlapping_returned_segments"] += sum(overlaps(hit, ref) for hit in hits)
        counts["top1"] += int(top1)
        counts["top3"] += int(top3)
        counts["first3"] += int(first3)
        counts["abstained"] += int(not hits)
        counts["returned_segments"] += len(hits)
        videos.add(row["videoID"])
        outcomes[row_id] = (top1, top3)
    n = len(outcomes)
    result = {"selected_rows": counts["selected_rows"], "evaluated_questions": n,
              "evaluated_videos": len(videos), "excluded_other_labels": counts["excluded_other_labels"],
              "invalid_rows": counts["invalid_rows"], "invalid_transcripts": counts["invalid_transcripts"],
              "incomplete_transcripts": counts["incomplete_transcripts"], "skipped_invalid_cues": counts["skipped_invalid_cues"],
              "top1_overlap_count": counts["top1"], "top1_overlap_rate": counts["top1"] / n if n else None,
              "top3_overlap_count": counts["top3"], "top3_overlap_rate": counts["top3"] / n if n else None,
              "first_three_caption_overlap_count": counts["first3"],
              "first_three_caption_overlap_rate": counts["first3"] / n if n else None,
              "top3_reference_half_covered_count": counts["top3_cover_half"],
              "top3_reference_half_covered_rate": counts["top3_cover_half"] / n if n else None,
              "mean_top3_reference_coverage": counts["sum_top3_coverage"] / n if n else None,
              "first_three_reference_half_covered_count": counts["first3_cover_half"],
              "mean_first_three_reference_coverage": counts["sum_first3_coverage"] / n if n else None,
              "overlapping_returned_segments": counts["overlapping_returned_segments"],
              "returned_segments": counts["returned_segments"],
              "returned_segment_window_overlap_fraction": (counts["overlapping_returned_segments"] / counts["returned_segments"]
                                                           if counts["returned_segments"] else None),
              "abstained": counts["abstained"]}
    if include_rows:
        result["_outcomes"] = outcomes
    return result


def compare(dataset, semantic_builder, *, sample="sample", allow_incomplete=False, segmentation="sentence"):
    keyword = evaluate(dataset, sample=sample, include_rows=True, allow_incomplete=allow_incomplete, segmentation=segmentation)
    semantic = evaluate(dataset, sample=sample, builder=semantic_builder, include_rows=True, allow_incomplete=allow_incomplete, segmentation=segmentation)
    a, b = keyword.pop("_outcomes"), semantic.pop("_outcomes")
    if a.keys() != b.keys():
        raise ValueError("Methods were evaluated on different questions")
    pairs = Counter()
    for key in a:
        left, right = a[key][1], b[key][1]
        label = "both_overlap" if left and right else "keyword_only" if left else "semantic_only" if right else "neither_overlap"
        pairs[label] += 1
    return {"keyword": keyword, "semantic": semantic,
            "paired_top3": {key: pairs[key] for key in ("both_overlap", "keyword_only", "semantic_only", "neither_overlap")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample", choices=("sample", "other", "all"), default="sample")
    parser.add_argument("--compare-model", type=Path, help="Optional local English ONNX model directory")
    parser.add_argument("--semantic-threshold", type=float, default=0.4)
    parser.add_argument("--segmentation", choices=("sentence", "workspace"), default="sentence",
                        help="Use supplied sentences or the workspace's subtitle grouping")
    parser.add_argument("--allow-incomplete-captions", action="store_true",
                        help="Skip invalid non-reference cues and report incomplete transcripts")
    args = parser.parse_args()
    try:
        data = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
        if args.compare_model:
            from video_rag.retrieval.onnx_semantic import (
                OnnxSentenceEncoder,
                build_semantic_pipeline,
            )
            if not math.isfinite(args.semantic_threshold) or not -1 <= args.semantic_threshold <= 1:
                raise ValueError("Invalid semantic threshold")
            encoder = OnnxSentenceEncoder(args.compare_model)
            result = compare(data, lambda segments: build_semantic_pipeline(segments, encoder, args.semantic_threshold),
                             sample=args.sample, allow_incomplete=args.allow_incomplete_captions, segmentation=args.segmentation)
        else:
            result = evaluate(data, sample=args.sample, allow_incomplete=args.allow_incomplete_captions, segmentation=args.segmentation)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({"warning": "Temporal overlap and reference coverage do not establish answer correctness. "
                                 "Full provided transcripts are searched; actual videos and audio are not used.",
                      "sample": args.sample, "segmentation": args.segmentation, "allow_incomplete_captions": args.allow_incomplete_captions,
                      "semantic_threshold": args.semantic_threshold if args.compare_model else None,
                      "metrics": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
