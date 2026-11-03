"""Run a local, model-free learning workspace from a video and sidecar subtitles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support running this script directly from a source checkout without an
# editable install. Dependencies still need to be installed in the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_rag.api import create_app
from video_rag.transcript_baseline import build_transcript_library, find_subtitles


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline subtitle keyword search; no ASR, video understanding or model calls.",
        epilog=(
            'Windows: .venv\\Scripts\\python.exe scripts\\run_transcript.py '
            '--video "C:\\Videos\\lecture.mp4" --subtitles "C:\\Videos\\lecture.srt"'
        ),
    )
    parser.add_argument("--video", type=Path, action="append", required=True, help="Local video file; repeat for multiple videos")
    parser.add_argument("--subtitles", type=Path, action="append", help="SRT/VTT for each video in the same order; omit all to find same-name sidecars")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        if args.subtitles and len(args.subtitles) != len(args.video):
            parser.error("Provide one --subtitles for each --video, or omit all for sidecars")
        subtitles = args.subtitles or [find_subtitles(video) for video in args.video]
        pipeline = build_transcript_library(list(zip(args.video, subtitles, strict=True)))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("Transcript baseline: local keyword search and direct subtitle excerpts; no AI model.")
    print(f"Open http://127.0.0.1:{args.port}/ (local access only)")
    create_app(pipeline, mode="transcript").run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()

