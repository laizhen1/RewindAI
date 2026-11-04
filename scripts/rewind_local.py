"""Open an authorized local video with provided or locally generated subtitles."""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DATABASE = ROOT / "artifacts" / "rewindai" / "library.sqlite3"
VIDEO_TYPES = {".mp4", ".webm", ".ogv", ".mov", ".m4v"}


def saved_video_status(video: Path, database: Path) -> str:
    """Check a saved source without creating or changing the workspace database."""
    if not database.is_file():
        return "missing"
    video_id = "video_" + hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:16]
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        has_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'local_transcript_videos'"
        ).fetchone()
        if not has_table:
            return "missing"
        row = connection.execute(
            "SELECT video_path, video_size, video_mtime_ns FROM local_transcript_videos WHERE video_id = ?",
            (video_id,),
        ).fetchone()
    if row is None:
        return "missing"
    stat = video.stat()
    if row[0] != str(video) or (row[1], row[2]) != (stat.st_size, stat.st_mtime_ns):
        return "changed"
    return "saved"


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
    try:
        saved_status = saved_video_status(video, WORKSPACE_DATABASE)
    except (OSError, sqlite3.Error) as exc:
        print(f"Error: cannot check the local library: {exc}", file=sys.stderr)
        return 1
    if saved_status == "changed":
        print("This video has changed since it was saved. Open the workspace to restore the original "
              "or remove the unavailable index before re-importing. Original files and notes are retained.", file=sys.stderr)
        return 1
    if saved_status == "saved":
        if args.subtitles:
            print("This video is already saved. To use different subtitles, first remove its index "
                  "in the workspace; original files and notes are retained.", file=sys.stderr)
            return 1
        print("Video is already saved; opening the local workspace without retranscribing.", flush=True)
        return subprocess.call([sys.executable, str(ROOT / "scripts" / "run_transcript.py"),
                                "--port", str(args.port)], cwd=ROOT)
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
