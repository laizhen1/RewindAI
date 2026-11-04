"""Check local temporal question files for obvious cross-split leakage.

Outputs aggregate counts only. Identical video IDs, bound video hashes and
normalized exact question strings are checked; near-duplicates are not detected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SPLITS = {"development", "validation", "test"}


def audit(datasets: list[dict]) -> dict:
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("At least one dataset is required")
    video_ids, video_hashes, questions = {}, {}, {}
    represented_splits = set()
    unbound = 0
    for data in datasets:
        if not isinstance(data, dict) or data.get("split") not in SPLITS:
            raise ValueError("Every dataset must have a development, validation, or test split")
        video_id, split = data.get("video_id"), data["split"]
        represented_splits.add(split)
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError("Each dataset needs a video_id")
        video_ids.setdefault(video_id.strip(), set()).add(split)
        binding = data.get("source_sha256")
        if binding is None:
            unbound += 1
        else:
            if (not isinstance(binding, dict) or set(binding) != {"video", "subtitles"}
                    or any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in binding.values())):
                raise ValueError("Invalid source fingerprint metadata")
            video_hashes.setdefault(binding["video"], set()).add(split)
        entries = data.get("questions")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Every dataset requires questions")
        for row in entries:
            question = row.get("question") if isinstance(row, dict) else None
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Each question must be a nonempty string")
            normalized = " ".join(question.casefold().split())
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            questions.setdefault(digest, set()).add(split)
    conflicts = {"video_ids_across_splits": sum(len(v) > 1 for v in video_ids.values()),
                 "video_hashes_across_splits": sum(len(v) > 1 for v in video_hashes.values()),
                 "exact_questions_across_splits": sum(len(v) > 1 for v in questions.values())}
    return {"datasets": len(datasets), "splits_present": sorted(represented_splits), "unbound_datasets": unbound, **conflicts,
            "no_detected_leakage": len(represented_splits) >= 2 and not any(conflicts.values()) and unbound == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, action="append", required=True,
                        help="Repeat for every local split JSON file")
    args = parser.parse_args()
    try:
        data = [json.loads(p.read_text(encoding="utf-8-sig")) for p in args.questions]
        report = audit(data)
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")
    print(json.dumps({"warning": "Detects exact matches only; hashes are recorded metadata and near-duplicate "
                                 "videos or paraphrased questions require separate review.", "audit": report}, indent=2))
    return 0 if report["no_detected_leakage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
