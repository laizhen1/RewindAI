"""Convert one video's local CSV annotations into draft temporal diagnostic JSON.

This does not verify labels or create audit events. Output is restricted to ignored
local directories and includes private question and answer text.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_rag.evaluation.provenance import bind_sources

TYPES = {"audio", "visual", "ocr", "multimodal", "unknown_route"}
SPLITS = {"development", "validation", "test"}


def field(row: dict, name: str) -> str:
    value = row.get(name)
    return value.strip() if isinstance(value, str) else ""


def prepare(rows: list[dict], video_id: str, split: str | None = None) -> dict:
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id is required")
    if split is not None and split not in SPLITS:
        raise ValueError("Invalid split")
    video_rows = [r for r in rows if field(r, "video_id") == video_id.strip()]
    if not video_rows:
        raise ValueError("No matching annotations")
    assigned = {field(row, "split") for row in video_rows}
    if assigned - (SPLITS | {""}):
        raise ValueError("Invalid split in selected video's rows")
    if len(assigned) > 1:
        raise ValueError("A video must not span multiple splits or mix assigned and unassigned rows")
    existing_split = next(iter(assigned)) or None
    if split is not None and existing_split != split:
        raise ValueError("The requested split does not match every row for this video")
    selected = video_rows
    effective_split = split or existing_split
    questions = []
    seen = set()
    for row in selected:
        qid = field(row, "question_id")
        question = field(row, "question")
        kind = field(row, "question_type")
        answerable_raw = field(row, "answerable").lower()
        if not qid or qid in seen or not question or kind not in TYPES or answerable_raw not in {"true", "false"}:
            raise ValueError("Each row needs a unique question_id, question, valid type, and true/false answerable")
        seen.add(qid)
        answerable = answerable_raw == "true"
        item = {"question_id": qid, "question": question, "question_type": kind,
                "expected_answerable": answerable}
        if answerable:
            try:
                start = float(field(row, "evidence_start_seconds"))
                end = float(field(row, "evidence_end_seconds"))
            except ValueError as exc:
                raise ValueError("Answerable questions require numeric evidence times") from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("Evidence times must be finite and satisfy 0 <= start < end")
            item.update({"evidence_start_seconds": start, "evidence_end_seconds": end,
                         "expected_answer": field(row, "reference_answer")})
        else:
            scope = field(row, "checked_scope")
            if not scope:
                raise ValueError("Unanswerable questions require a documented checked_scope")
            if field(row, "evidence_start_seconds") or field(row, "evidence_end_seconds"):
                raise ValueError("Unanswerable questions must not include an evidence range")
            item["checked_scope"] = scope
        questions.append(item)
    return {"status": "csv_import_unverified_draft", "video_id": video_id.strip(), "split": effective_split,
            "notes": "CSV conversion does not establish human review, complete scope, or video-time correctness.",
            "questions": questions}


def write_output(output: Path, data: dict, allowed_roots: tuple[Path, ...]) -> None:
    target = output.expanduser().resolve()
    if target.suffix.lower() != ".json" or not any(target != root.resolve() and target.is_relative_to(root.resolve())
                                                   for root in allowed_roots):
        raise ValueError("Output must be a JSON file under artifacts/ or local_data/")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--split", choices=sorted(SPLITS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, help="Optional local video to bind by SHA-256")
    parser.add_argument("--subtitles", type=Path, help="Optional local subtitle file to bind by SHA-256")
    args = parser.parse_args()
    if (args.video is None) != (args.subtitles is None):
        parser.error("Provide both --video and --subtitles to bind sources")
    try:
        with args.csv.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        data = prepare(rows, args.video_id, args.split)
        if args.video is not None:
            bind_sources(data, args.video, args.subtitles)
        write_output(args.output, data, (ROOT / "artifacts", ROOT / "local_data"))
    except (OSError, TypeError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Conversion failed: {exc}\n")
    print(f"Wrote {len(data['questions'])} unverified draft questions locally. Do not publish private data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
