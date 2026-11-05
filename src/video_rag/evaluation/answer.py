from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

# Match CJK ideographs individually and other alphabetic/numeric sequences as words.
# This is lexical overlap, not semantic equivalence or grounded correctness.
TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", re.UNICODE)


def answer_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def exact_match(prediction: str, reference: str) -> float:
    return float(answer_tokens(prediction) == answer_tokens(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(answer_tokens(prediction))
    expected = Counter(answer_tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted or not expected:
        return float(predicted == expected)
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def evaluate_answers(
    predictions: Mapping[str, str],
    references: Mapping[str, str],
) -> dict[str, float]:
    if not references:
        raise ValueError("references must not be empty")
    if any(not isinstance(reference, str) or not answer_tokens(reference) for reference in references.values()):
        raise ValueError("Each reference must contain answer tokens; evaluate unanswerable questions separately")
    if any(not isinstance(value, str) for value in predictions.values()):
        raise ValueError("Predictions must be strings")
    identifiers = sorted(references)
    return {
        "exact_match": sum(
            exact_match(predictions.get(identifier, ""), references[identifier])
            for identifier in identifiers
        )
        / len(identifiers),
        "token_f1": sum(
            token_f1(predictions.get(identifier, ""), references[identifier])
            for identifier in identifiers
        )
        / len(identifiers),
    }

