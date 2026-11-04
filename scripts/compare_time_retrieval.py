"""Compare local keyword and semantic search on the same time-labeled questions."""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.diagnose_transcript import (
    comparison_by_question_type,
    diagnose,
    metrics,
    operating_metrics,
)
from scripts.prepare_temporal_questions import write_output
from video_rag.evaluation.provenance import fingerprint, verify_sources
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


def compare(questions, keyword, semantic, min_coverage=0.5):
    if not isinstance(questions, list) or not questions:
        raise ValueError("A nonempty question list is required")
    if not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    a = diagnose(keyword, questions)
    b = diagnose(semantic, questions)
    pairs = {"both_hit": 0, "keyword_only_hit": 0, "semantic_only_hit": 0, "neither_hit": 0}
    negative_pairs = {"both_abstain": 0, "keyword_only_abstains": 0,
                      "semantic_only_abstains": 0, "neither_abstains": 0}
    rows = []
    for left, right in zip(a, b, strict=True):
        if left["expected_answerable"]:
            ah = not left["abstained"] and left["first_hit_reference_coverage"] >= min_coverage
            bh = not right["abstained"] and right["first_hit_reference_coverage"] >= min_coverage
            label = "both_hit" if ah and bh else "keyword_only_hit" if ah else "semantic_only_hit" if bh else "neither_hit"
            pairs[label] += 1
        else:
            aa, ba = left["abstained"], right["abstained"]
            label = "both_abstain" if aa and ba else "keyword_only_abstains" if aa else "semantic_only_abstains" if ba else "neither_abstains"
            negative_pairs[label] += 1
        rows.append({"question": left["question"], "expected_answerable": left["expected_answerable"],
                     "comparison": label, "keyword": left, "semantic": right})
    return {"keyword": {"metrics": metrics(a, min_coverage), "operating": operating_metrics(a, min_coverage),
                        "by_question_type": comparison_by_question_type(a, min_coverage)},
            "semantic": {"metrics": metrics(b, min_coverage), "operating": operating_metrics(b, min_coverage),
                         "by_question_type": comparison_by_question_type(b, min_coverage)},
            "paired_answerable": pairs, "paired_unanswerable": negative_pairs, "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--semantic-threshold", type=float, default=0.4)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Write a new report under artifacts/ or local_data/")
    parser.add_argument("--aggregate-only", action="store_true", help="Omit individual questions and results")
    parser.add_argument("--record-hashes", action="store_true", help="Fingerprint inputs and model before and after the run")
    args = parser.parse_args()
    try:
        from video_rag.retrieval.onnx_semantic import OnnxSentenceEncoder, build_semantic_pipeline
        paths = {"video": args.video, "subtitles": args.subtitles, "questions": args.questions,
                 "model": args.model_dir / "onnx" / "model_quantized.onnx", "tokenizer": args.model_dir / "tokenizer.json"}
        input_hashes = {name: fingerprint(path) for name, path in paths.items()} if args.record_hashes else None
        data = json.loads(args.questions.read_text(encoding="utf-8-sig"))
        source_binding = verify_sources(data, args.video, args.subtitles)
        segments = segments_from_subtitles(args.video, args.subtitles)
        encoder = OnnxSentenceEncoder(args.model_dir)
        keyword = build_transcript_segments(segments)
        semantic = build_semantic_pipeline(segments, encoder, args.semantic_threshold)
        result = compare(data["questions"], keyword, semantic, args.min_coverage)
        verify_sources(data, args.video, args.subtitles)
        if input_hashes is not None and input_hashes != {name: fingerprint(path) for name, path in paths.items()}:
            raise ValueError("An input or model file changed during evaluation")
        if args.aggregate_only:
            result.pop("results")
        report = {"warning": "Temporal coverage depends on label quality and does not establish answer correctness. "
                             "Reports and hashes may contain sensitive course information; keep them local.",
                  "created_at_utc": datetime.now(UTC).isoformat(),
                  "label_status": data.get("status", "not provided"), "source_binding": source_binding,
                  "semantic_threshold": args.semantic_threshold, "min_coverage": args.min_coverage,
                  "aggregate_only": args.aggregate_only, "input_sha256": input_hashes, "comparison": result}
        if args.output:
            write_output(args.output, report, (ROOT / "artifacts", ROOT / "local_data"))
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Comparison failed: {exc}\n")
    if args.output:
        print("Saved a local evaluation report. Keep private data and hashes on this computer.")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
