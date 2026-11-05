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
from video_rag.retrieval.fusion import reciprocal_rank_fusion
from video_rag.retrieval.onnx_clip import ClipFrameRetriever, OnnxClipEncoder
from video_rag.schemas import SearchHit
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


class TranscriptAdapter:
    """Use the actual workspace pipeline rather than a separate lexical baseline."""
    def __init__(self, segments, pipeline=None):
        self.pipeline = pipeline if pipeline is not None else build_transcript_segments(segments)

    def search(self, query, count):
        response = self.pipeline.ask(query)
        return [SearchHit(item.segment.segment_id, item.rerank_score, "transcript", rank)
                for rank, item in enumerate(response.evidence[:count], 1)]


class FusedAdapter:
    """Combine ranks, not uncalibrated lexical and CLIP score magnitudes."""
    def __init__(self, transcript, visual):
        self.transcript, self.visual = transcript, visual

    def search(self, query, count):
        return reciprocal_rank_fusion([self.transcript.search(query, count),
                                       self.visual.search(query, count)], k=60, top_k=count)


def compare(segments, labels, visual, *, allow_unreviewed=False, transcript_pipeline=None):
    if len(segments) > 100:
        raise ValueError("At most 100 segments are supported; do not compare different sampled sets")
    validate_label_header(segments, labels, allow_unreviewed=allow_unreviewed)
    text = TranscriptAdapter(segments, pipeline=transcript_pipeline)
    transcript = evaluate(text, segments, labels, allow_unreviewed=allow_unreviewed)
    visual_metrics = evaluate(visual, segments, labels, allow_unreviewed=allow_unreviewed)
    fused = evaluate(FusedAdapter(text, visual), segments, labels, allow_unreviewed=allow_unreviewed)
    return {"transcript": transcript, "visual": visual_metrics, "fused_experimental": fused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--semantic-model", type=Path, help="Optional local MiniLM model directory; default uses keyword search")
    parser.add_argument("--semantic-threshold", type=float, default=0.4,
                        help="Uncalibrated semantic threshold (default 0.4)")
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
        transcript_pipeline = None
        if args.semantic_model:
            from video_rag.retrieval.onnx_semantic import (
                OnnxSentenceEncoder,
                build_semantic_pipeline,
            )

            transcript_pipeline = build_semantic_pipeline(
                segments, OnnxSentenceEncoder(args.semantic_model), args.semantic_threshold)
        result = compare(segments, labels, visual, allow_unreviewed=args.allow_unreviewed,
                         transcript_pipeline=transcript_pipeline)
        validate_sources(labels, args.video, args.subtitles)
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Comparison failed: {exc}\n")
    print(json.dumps({"warning": "Only sampled midpoint frames are labeled. Text and image searches are "
                                 "separate and fused by experimental reciprocal-rank ranking; "
                                 "results do not measure answer accuracy. "
                                 "Output contains queries and must remain local for private media.",
                      "transcript_retrieval": "semantic_experimental" if args.semantic_model else "keyword",
                      "semantic_threshold": args.semantic_threshold if args.semantic_model else None,
                      "metrics": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
