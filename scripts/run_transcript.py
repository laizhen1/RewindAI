"""Run a local, model-free learning workspace from a video and sidecar subtitles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running this script directly from a source checkout without an
# editable install. Dependencies still need to be installed in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_rag.api import create_app
from video_rag.transcript_baseline import (
    build_transcript_library,
    build_transcript_segments,
    find_subtitles,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline subtitle search; optional local semantic model, no ASR or visual understanding.",
        epilog=(
            'Windows: .venv\\Scripts\\python.exe scripts\\run_transcript.py '
            '--video "C:\\Videos\\lecture.mp4" --subtitles "C:\\Videos\\lecture.srt"'
        ),
    )
    parser.add_argument("--video", type=Path, action="append", default=[], help="Optional local video; repeat for multiple videos")
    parser.add_argument("--subtitles", type=Path, action="append", help="SRT/VTT for each video in the same order; omit all to find same-name sidecars")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--semantic-model", type=Path, help="Optional local English ONNX model directory")
    parser.add_argument("--semantic-threshold", type=float, default=0.4,
                        help="Experimental uncalibrated cosine threshold (default 0.4)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        if args.subtitles and len(args.subtitles) != len(args.video):
            parser.error("Provide one --subtitles for each --video, or omit all for sidecars")
        subtitles = args.subtitles or [find_subtitles(video) for video in args.video]
        sources = list(zip(args.video, subtitles, strict=True))
        pipeline = build_transcript_library(sources) if sources else build_transcript_segments([])
        builder = None
        retrieval_type = "keyword"
        if args.semantic_model:
            from video_rag.retrieval.onnx_semantic import (
                OnnxSentenceEncoder,
                build_semantic_pipeline,
            )

            encoder = OnnxSentenceEncoder(args.semantic_model)
            builder = lambda segments: build_semantic_pipeline(segments, encoder, args.semantic_threshold)
            pipeline = builder(list(pipeline.segments()))
            retrieval_type = "semantic_experimental"
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print("Transcript search mode:", retrieval_type, "— direct subtitle excerpts; no generated answer.")
    print(f"Open http://127.0.0.1:{args.port}/ (local access only)")
    create_app(pipeline, mode="transcript", transcript_builder=builder,
               retrieval_type=retrieval_type).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
