"""Prepare local frame previews and a draft visual-label manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_clip_frames import sha256_file

from video_rag.api.frames import extract_frame_png
from video_rag.transcript_baseline import segments_from_subtitles


def prepare(video: Path, subtitles: Path, destination: Path, allowed_roots: tuple[Path, ...]) -> int:
    destination = destination.resolve()
    if not any(destination != root.resolve() and destination.is_relative_to(root.resolve()) for root in allowed_roots):
        raise ValueError("Output must be a new subdirectory of artifacts/ or local_data/")
    if destination.exists():
        raise ValueError("Output directory already exists; choose a new one")
    segments = segments_from_subtitles(video, subtitles)
    if len(segments) > 100:
        raise ValueError("This prototype supports at most 100 segments")
    # Hash before and after extraction to avoid binding labels to changed input.
    video_hash, subtitle_hash = sha256_file(video), sha256_file(subtitles)
    midpoints = [(segment.start_time + segment.end_time) / 2 for segment in segments]
    frames = [extract_frame_png(video, midpoint) for midpoint in midpoints]
    if video_hash != sha256_file(video) or subtitle_hash != sha256_file(subtitles):
        raise ValueError("Input changed during frame extraction")
    destination.mkdir(parents=True, exist_ok=False)
    for index, (image, _) in enumerate(frames):
        with (destination / f"frame-{index:03}.png").open("xb") as output:
            output.write(image)
    manifest = {"review_status": "draft", "video_sha256": video_hash, "subtitles_sha256": subtitle_hash,
                "segment_midpoints_seconds": midpoints, "decoded_frame_seconds": [time for _, time in frames],
                "queries": []}
    with (destination / "labels.json").open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return len(segments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = prepare(args.video, args.subtitles, args.output_dir, (ROOT / "artifacts", ROOT / "local_data"))
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Could not prepare local review: {exc}\n")
    print(f"Prepared {count} local frames and draft labels in {args.output_dir}.")
    print("These may contain private material; keep them local and review before publishing any result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
