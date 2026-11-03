"""Compare experimental local English semantic retrieval on a labelled sample."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_transcript import diagnose, metrics

from video_rag.adapters import EvidenceGenerator
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval.onnx_semantic import (
    OnnxSentenceEncoder,
    SemanticReranker,
    SemanticRetriever,
)
from video_rag.transcript_baseline import segments_from_subtitles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Uncalibrated cosine cutoff; select on a separate validation set")
    args = parser.parse_args()
    if not -1 <= args.threshold <= 1:
        parser.error("threshold must be between -1 and 1")
    encoder = OnnxSentenceEncoder(args.model_dir)
    pipeline = VideoRAGPipeline(retrievers=[SemanticRetriever(encoder)],
                                reranker=SemanticReranker(encoder), generator=EvidenceGenerator(),
                                minimum_rerank_score=args.threshold)
    pipeline.build(segments_from_subtitles(args.video, args.subtitles))
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))["questions"]
    results = diagnose(pipeline, questions)
    print(json.dumps({"warning": "Experimental English-only retrieval; threshold is not calibrated; "
                                  "metrics depend on label quality and do not evaluate answer correctness.",
                      "threshold": args.threshold, "metrics": metrics(results), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

