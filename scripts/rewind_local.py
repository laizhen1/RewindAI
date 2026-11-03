"""Open an authorized local video with provided or locally generated subtitles."""
from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_TYPES = {".mp4", ".webm", ".ogv", ".mov", ".m4v"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Authorized local video")
    parser.add_argument("--subtitles", type=Path, help="Existing UTF-8 SRT/VTT file")
    parser.add_argument("--language", choices=("en", "zh"), help="Spoken language for transcription")
    parser.add_argument("--model", default="base", help="Local faster-whisper model or size")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args(argv)
    video = args.video.expanduser().resolve()
    if not video.is_file() or video.suffix.lower() not in VIDEO_TYPES:
        parser.error("video must be an existing MP4, WebM, OGV, MOV or M4V file")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.subtitles:
        subtitles = args.subtitles.expanduser().resolve()
        if not subtitles.is_file() or subtitles.suffix.lower() not in {".srt", ".vtt"}:
            parser.error("subtitles must be an existing SRT or VTT file")
    else:
        subtitles = ROOT / "artifacts" / "transcripts" / f"rewind-{uuid.uuid4().hex}.srt"
        command = [sys.executable, str(ROOT / "scripts" / "transcribe_local.py"), str(video),
                   "--output", str(subtitles), "--model", args.model]
        if args.language:
            command += ["--language", args.language]
        if args.allow_model_download:
            command.append("--allow-model-download")
        status = subprocess.call(command, cwd=ROOT)
        if status:
            return status
    print("Starting local subtitle search; verify results against the video.", flush=True)
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "run_transcript.py"),
                            "--video", str(video), "--subtitles", str(subtitles),
                            "--save-startup-video", "--port", str(args.port)], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
