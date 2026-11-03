"""Transcribe an authorized local recording with a local faster-whisper model."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts" / "transcripts"
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".ogv", ".avi", ".m4v", ".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def timestamp(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Subtitle timestamps must be finite and nonnegative")
    millis = round(seconds * 1000)
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_part, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{millis:03d}"


def render_srt(segments: Any) -> str:
    entries: list[str] = []
    for segment in segments:
        text = " ".join(segment.text.split())
        if not text:
            continue
        start, end = float(segment.start), float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("The model returned an invalid subtitle time interval")
        begin, finish = timestamp(start), timestamp(end)
        if begin == finish:
            raise ValueError("A subtitle interval is shorter than SRT millisecond precision")
        entries.append(f"{len(entries) + 1}\n{begin} --> {finish}\n{text}\n")
    if not entries:
        raise ValueError("The model returned no nonempty speech segments")
    return "\n".join(entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create SRT subtitles locally from an authorized video or audio file."
    )
    parser.add_argument("media", type=Path, help="Local video/audio you are authorized to process")
    parser.add_argument("--language", choices=("en", "zh"), help="Spoken language; omit to detect")
    parser.add_argument("--model", default="base", help="faster-whisper model size or local directory")
    parser.add_argument("--output", type=Path, help="SRT path inside artifacts/transcripts")
    parser.add_argument("--allow-model-download", action="store_true", help="Permit model download")
    args = parser.parse_args(argv)

    media = args.media.expanduser().resolve()
    if not media.is_file() or media.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print("Error: provide an existing supported local video or audio file.", file=sys.stderr)
        return 2
    root = OUTPUT_ROOT.resolve()
    output = args.output.expanduser().resolve() if args.output else root / f"{media.stem}.srt"
    if not output.is_relative_to(root) or output == root or output.suffix.lower() != ".srt":
        print("Error: output must be an .srt file inside artifacts/transcripts.", file=sys.stderr)
        return 2
    if output.exists():
        print(f"Error: output already exists; choose a new path: {output}", file=sys.stderr)
        return 2
    model_path = Path(args.model).expanduser()
    if model_path.is_dir() and not (model_path / "tokenizer.json").is_file():
        print("Error: local model directory must include tokenizer.json.", file=sys.stderr)
        return 2

    # The model and media stay local; this option only controls model-weight retrieval.
    if not args.allow_model_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
    except ImportError:
        print("Error: install faster-whisper to use local transcription.", file=sys.stderr)
        return 1
    try:
        if model_path.is_dir():
            resolved_model = model_path.resolve()
        else:
            resolved_model = Path(download_model(
                args.model, local_files_only=not args.allow_model_download
            ))
        # Some versions fall back to fetching a tokenizer even when local_files_only
        # is set. Fail instead if the chosen model snapshot does not include it.
        if not (resolved_model / "tokenizer.json").is_file():
            raise ValueError("The resolved model snapshot does not include tokenizer.json")
        model = WhisperModel(
            str(resolved_model), device="cpu", compute_type="int8", local_files_only=True,
        )
        segments, _ = model.transcribe(str(media), language=args.language)
        srt = render_srt(segments)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation closes the race between the earlier check and writing.
        with output.open("x", encoding="utf-8", newline="\n") as file:
            file.write(srt)
    except FileExistsError:
        print(f"Error: output already exists; choose a new path: {output}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Error: transcription failed: {exc}", file=sys.stderr)
        return 1
    print(f"Local subtitles saved: {output}")
    print("Review the subtitles against the source before treating them as ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
