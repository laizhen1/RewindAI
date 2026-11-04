"""Compare subtitle and visual retrieval against the same sampled-frame labels.

The labels describe only sampled midpoint frames, not full-video relevance.
The comparison runs locally and cannot establish answer or multimodal accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_clip_frames import evaluate, validate_label_header, validate_sources
from video_rag.retrieval.onnx_clip import ClipFrameRetriever, OnnxClipEncoder
from video_rag.schemas import SearchHit
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


class TranscriptAdapter:
    """Use the actual workspace pipeline rather than a separate lexical baseline."""
    def __init__(self, segments):
        self.pipeline = build_transcript_segments(segments)

    def search(self, query, count):
        response = self.pipeline.ask(query)
        return [SearchHit(item.segment.segment_id, item.rerank_score, "transcript", rank)
                for rank, item in enumerate(response.evidence[:count], 1)]


def compare(segments, labels, visual, *, allow_unreviewed=False):
    if len(segments) > 100:
        raise ValueError("At most 100 segments are supported; do not compare different sampled sets")
    validate_label_header(segments, labels, allow_unreviewed=allow_unreviewed)
    transcript = evaluate(TranscriptAdapter(segments), segments, labels, allow_unreviewed=allow_unreviewed)
    visual_metrics = evaluate(visual, segments, labels, allow_unreviewed=allow_unreviewed)
    return {"transcript": transcript, "visual": visual_metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    try:
        labels = json.loads(args.labels.read_text(encoding="utf-8-sig"))
        segments = segments_from_subtitles(args.video, args.subtitles)
        if len(segments) > 100:
            raise ValueError("At most 100 segments are supported; do not compare different sampled sets")
        validate_label_header(segments, labels, allow_unreviewed=args.allow_unreviewed)
        validate_sources(labels, args.video, args.subtitles)
        visual = ClipFrameRetriever(OnnxClipEncoder(args.model_dir))
        visual.build(segments)
        result = compare(segments, labels, visual, allow_unreviewed=args.allow_unreviewed)
        validate_sources(labels, args.video, args.subtitles)
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Comparison failed: {exc}\n")
    print(json.dumps({"warning": "Only sampled midpoint frames are labeled. Text and image searches are "
                                 "separate; this is not fused multimodal retrieval or answer accuracy. "
                                 "Output contains queries and must remain local for private media.",
                      "metrics": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
