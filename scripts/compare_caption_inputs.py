"""Compare two local subtitle inputs using identical fixed temporal windows."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_transcript import diagnose, metrics

from video_rag.schemas import TimedText, VideoSegment
from video_rag.transcript_baseline import build_transcript_segments, parse_subtitles


def fixed_windows(cues: list[TimedText], count: int, seconds: float, video: Path) -> list[VideoSegment]:
    """Place each whole cue into the window containing its midpoint."""
    if not math.isfinite(seconds) or seconds <= 0 or count < 1 or count > 100_000:
        raise ValueError("Invalid window size or count")
    windows: list[list[str]] = [[] for _ in range(count)]
    for cue in cues:
        window = int(((cue.start_time + cue.end_time) / 2) // seconds)
        if window >= count:
            raise ValueError("Cue midpoint is outside the supplied windows")
        windows[window].append(cue.text)
    return [VideoSegment(segment_id=f"window_{i}", video_id="comparison_video", source_path=str(video),
                         start_time=i * seconds, end_time=(i + 1) * seconds, transcript=" ".join(text))
            for i, text in enumerate(windows)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference-captions", type=Path, required=True)
    parser.add_argument("--generated-captions", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    args = parser.parse_args()
    try:
        if not args.video.is_file():
            raise ValueError("Video must exist locally")
        if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
            raise ValueError("Window length must be finite and positive")
        cue_sets = [parse_subtitles(p.read_text(encoding="utf-8-sig")) for p in
                    (args.reference_captions, args.generated_captions)]
        count = int(max((cue.start_time + cue.end_time) / 2 for cues in cue_sets for cue in cues)
                    // args.window_seconds) + 1
        questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))["questions"]
        output = {}
        for name, cues in zip(("supplied", "generated"), cue_sets):
            pipeline = build_transcript_segments(fixed_windows(cues, count, args.window_seconds, args.video))
            results = diagnose(pipeline, questions)
            output[name] = {"metrics": metrics(results), "results": results}
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        parser.exit(2, f"Comparison failed: {exc}\n")
    print(json.dumps({"warning": "Provisional labels; midpoint assignment retains timing differences. "
                                 "This does not measure answer accuracy.",
                      "window_seconds": args.window_seconds, "window_count": count, "inputs": output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
