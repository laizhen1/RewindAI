"""Prepare and index one local video without a hosted inference service."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = ROOT / "artifacts" / "ingestions"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".ogv", ".avi", ".m4v"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Process authorized local videos with local models. Hugging Face offline mode "
            "is enabled by default; missing models require a prior download or explicit opt-in."
        )
    )
    result.add_argument(
        "video", type=Path,
        help="Local video file or folder containing only videos you may process",
    )
    result.add_argument("--language", help="Optional Whisper language code, e.g. en or zh")
    result.add_argument("--config", type=Path, default=ROOT / "config.toml")
    result.add_argument(
        "--device", default="cuda", help="Indexing device (default: cuda; e.g. cpu)"
    )
    result.add_argument(
        "--skip-captions", action="store_true", help="Skip generating visual captions"
    )
    result.add_argument(
        "--allow-model-download", action="store_true",
        help="Allow missing model files to be downloaded from model providers",
    )
    return result


def format_server_command(command: list[str], *, windows: bool) -> str:
    if windows:
        # Single-quoted PowerShell literals do not expand $ or command substitutions.
        return "& " + " ".join("'" + value.replace("'", "''") + "'" for value in command)
    return shlex.join(command)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    video = args.video.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if video.is_dir():
        candidates = [path for path in video.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS]
        valid_video = bool(candidates) and all(path.is_file() for path in candidates)
    else:
        valid_video = video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS
    if not valid_video:
        print("Error: provide a supported local video file or a nonempty video folder.", file=sys.stderr)
        return 2
    if not config.is_file():
        print("Error: the selected configuration file does not exist.", file=sys.stderr)
        return 2

    # A unique directory avoids overwriting an existing project's indexes or annotations.
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = ARTIFACTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    segments = run_dir / "segments.jsonl"
    frames = run_dir / "frames"
    indexes = run_dir / "indexes"
    prepare = [
        sys.executable, str(ROOT / "scripts" / "prepare_videos.py"),
        "--input", str(video), "--output", str(segments), "--frames-dir", str(frames),
        "--config", str(config),
    ]
    if args.language:
        prepare.extend(["--language", args.language])
    if args.skip_captions:
        prepare.append("--skip-captions")
    build = [
        sys.executable, str(ROOT / "scripts" / "build_indexes.py"),
        "--segments", str(segments), "--index-dir", str(indexes),
        "--config", str(config), "--device", args.device,
    ]

    env = os.environ.copy()
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if args.allow_model_download:
        # Explicitly override inherited offline settings only for these child processes.
        env["HF_HUB_OFFLINE"] = "0"
        env["TRANSFORMERS_OFFLINE"] = "0"
    else:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    print(f"Output: {run_dir}", flush=True)
    print(
        "Local processing; model downloads "
        + ("allowed." if args.allow_model_download else "disabled (cached models required)."),
        flush=True,
    )
    print("[1/2] Preparing video...", flush=True)
    try:
        subprocess.run(prepare, cwd=ROOT, env=env, check=True)
        print("[2/2] Building indexes...", flush=True)
        subprocess.run(build, cwd=ROOT, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"Processing failed with exit code {exc.returncode}. "
            f"Partial outputs remain in {run_dir}. Check model cache and dependencies.",
            file=sys.stderr,
        )
        return exc.returncode if exc.returncode > 0 else 1
    print(f"Completed. Segments: {segments}\nIndexes: {indexes}", flush=True)
    server = [
        sys.executable, str(ROOT / "scripts" / "run_server.py"),
        "--segments", str(segments), "--index-dir", str(indexes),
        "--config", str(config), "--device", args.device,
    ]
    # PowerShell needs its invocation operator when the executable is a quoted path.
    command = format_server_command(server, windows=os.name == "nt")
    print(f"Start the real server (PowerShell on Windows) with:\n  {command}", flush=True)
    print("The server runs separately; set its offline environment if needed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
