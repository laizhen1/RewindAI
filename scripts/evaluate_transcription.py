"""Compare timestamped transcript text against a reference using word alignment."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_rag.transcript_baseline import parse_subtitles

# CJK characters are counted individually; English words and numbers are
# compared case-insensitively. Punctuation and timestamps are ignored.
TOKEN = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
MAX_CELLS = 9_000_000


def evaluate(reference: str, hypothesis: str) -> dict:
    ref, hyp = TOKEN.findall(reference.lower()), TOKEN.findall(hypothesis.lower())
    if not ref:
        raise ValueError("Reference transcript must contain words")
    if (len(ref) + 1) * (len(hyp) + 1) > MAX_CELLS:
        raise ValueError("Transcripts too long for this evaluator; split into smaller recordings")
    # Each cell stores edit distance and counts for matches, substitutions,
    # deletions, insertions. Prefer a match and then substitution on tied cost.
    previous = [(i, 0, 0, 0, i) for i in range(len(hyp) + 1)]
    for i, expected in enumerate(ref, start=1):
        current = [(i, 0, 0, i, 0)]
        for j, actual in enumerate(hyp, start=1):
            diagonal = previous[j - 1]
            same = expected == actual
            choices = [
                (diagonal[0] + (not same), diagonal[1] + same, diagonal[2] + (not same), diagonal[3], diagonal[4]),
                (previous[j][0] + 1, previous[j][1], previous[j][2], previous[j][3] + 1, previous[j][4]),
                (current[j - 1][0] + 1, current[j - 1][1], current[j - 1][2], current[j - 1][3], current[j - 1][4] + 1),
            ]
            current.append(min(choices, key=lambda value: value[0]))
        previous = current
    _, matches, substitutions, deletions, insertions = previous[-1]
    precision = matches / len(hyp) if hyp else None
    recall = matches / len(ref)
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and precision + recall else 0.0 if precision is not None else None)
    return {"reference_tokens": len(ref), "hypothesis_tokens": len(hyp), "matches": matches,
            "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
            "wer": (substitutions + deletions + insertions) / len(ref),
            "precision": precision, "recall": recall, "f1": f1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--hypothesis", required=True, type=Path)
    args = parser.parse_args()
    try:
        ref = " ".join(c.text for c in parse_subtitles(args.reference.read_text(encoding="utf-8-sig")))
        hyp = " ".join(c.text for c in parse_subtitles(args.hypothesis.read_text(encoding="utf-8-sig")))
        result = evaluate(ref, hyp)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({"warning": "Quality depends on reference; timestamps are not evaluated", "metrics": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

