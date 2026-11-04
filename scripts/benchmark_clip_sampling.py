"""Measure local CLIP sampling and indexing on an authorized video."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_rag.api.visual_sampling import sample_visual_segments
from video_rag.retrieval.onnx_clip import ClipFrameRetriever, OnnxClipEncoder
from video_rag.schemas import VideoSegment


def synthetic_segments(video: Path, count: int, step: float) -> tuple[VideoSegment, ...]:
    if not video.is_file() or not 1 <= count <= 5000 or not math.isfinite(step) or step <= 0:
        raise ValueError("Provide a local video, 1–5000 segments and a positive finite step")
    return tuple(VideoSegment(str(i), "local", str(video.resolve()), i * step, (i + 1) * step)
                 for i in range(count))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=105)
    parser.add_argument("--step", type=float, default=0.8,
                        help="Synthetic segment length in seconds; entire range must fit the video")
    parser.add_argument("--query", default="A hospital building")
    args = parser.parse_args()
    try:
        segments = synthetic_segments(args.video, args.count, args.step)
        start = perf_counter()
        selected = sample_visual_segments(segments)
        retriever = ClipFrameRetriever(OnnxClipEncoder(args.model_dir))
        retriever.build(selected)
        indexing = perf_counter() - start
        start = perf_counter()
        hits = retriever.search(args.query, 3)
        query_seconds = perf_counter() - start
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"Benchmark failed: {exc}\n")
    print(json.dumps({"source_segments": len(segments), "sampled_segments": len(selected),
                      "index_seconds_including_model_load": indexing, "query_seconds": query_seconds,
                      "returned_candidates": len(hits),
                      "warning": "Synthetic segment timing; latency depends on machine and source video. "
                                 "Does not measure accuracy or peak memory."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
