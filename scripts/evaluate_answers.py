from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_rag.evaluation import evaluate_answers


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def index_answers(rows: list[dict]) -> dict[str, str]:
    indexed = {}
    for item in rows:
        if (not isinstance(item, dict) or not isinstance(item.get("question_id"), str)
                or not item["question_id"].strip() or not isinstance(item.get("answer"), str)):
            raise ValueError("Each row requires a nonempty question_id and string answer")
        if item["question_id"] in indexed:
            raise ValueError("Duplicate question_id")
        indexed[item["question_id"]] = item["answer"]
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated answers.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    predictions = read_jsonl(args.predictions)
    references = index_answers(questions)
    predicted_answers = index_answers(predictions)
    result = evaluate_answers(predicted_answers, references)
    print(json.dumps({**result, "method": "lexical token overlap: CJK characters and other word sequences",
                      "warning": "Not semantic correctness, groundedness, or question-level answerability F1. "
                                 "Reference answers must be nonempty; evaluate unanswerable cases separately."}, indent=2))


if __name__ == "__main__":
    main()

