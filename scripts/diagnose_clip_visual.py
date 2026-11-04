"""Run experimental local CLIP frame retrieval on an authorized video."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_rag.retrieval.onnx_clip import ClipFrameRetriever, OnnxClipEncoder
from video_rag.transcript_baseline import segments_from_subtitles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--question", action="append", required=True, help="English visual query; may be repeated")
    args = parser.parse_args()
    try:
        segments = segments_from_subtitles(args.video, args.subtitles)
        lookup = {segment.segment_id: segment for segment in segments}
        start = perf_counter()
        retriever = ClipFrameRetriever(OnnxClipEncoder(args.model_dir))
        retriever.build(segments)
        build_seconds = perf_counter() - start
        results = []
        for question in args.question:
            started = perf_counter()
            hits = retriever.search(question, 3)
            results.append({"question": question, "search_seconds": perf_counter() - started,
                            "hits": [{"start_seconds": lookup[hit.segment_id].start_time,
                                      "end_seconds": lookup[hit.segment_id].end_time,
                                      "score": hit.score} for hit in hits]})
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Visual diagnostic failed: {exc}\n")
    print(json.dumps({"warning": "Experimental visual similarity only; one midpoint frame per subtitle segment. "
                                  "Scores are not confidence and results require visual review.",
                      "segments": len(segments), "build_seconds": build_seconds, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
