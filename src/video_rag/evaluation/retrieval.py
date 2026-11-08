from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of all labeled relevant segments retrieved in the first k."""
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Whether at least one labeled relevant segment is in the first k."""
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    return float(bool(set(ranked[:k]) & relevant))


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    seen: set[str] = set()
    dcg = 0.0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant and item not in seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen.add(item)
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(
    predictions: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, Iterable[str]],
    *,
    cutoffs: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    question_ids = sorted(ground_truth)
    if not question_ids:
        raise ValueError("ground_truth must not be empty")
    if not cutoffs or any(type(k) is not int or k <= 0 for k in cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("cutoffs must contain distinct positive integers")

    totals = {f"recall@{k}": 0.0 for k in cutoffs}
    totals.update({f"hit@{k}": 0.0 for k in cutoffs})
    totals.update({f"mrr@{k}": 0.0 for k in cutoffs})
    totals.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    totals["mrr"] = 0.0

    for question_id in question_ids:
        ranked = predictions.get(question_id, ())
        relevant = set(ground_truth[question_id])
        if not relevant:
            raise ValueError("Each evaluated question must have at least one relevant segment")
        if len(set(ranked)) != len(ranked):
            raise ValueError("Predicted rankings must not contain duplicate segments")
        totals["mrr"] += reciprocal_rank(ranked, relevant)
        for k in cutoffs:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
            totals[f"hit@{k}"] += hit_at_k(ranked, relevant, k)
            totals[f"mrr@{k}"] += reciprocal_rank(ranked[:k], relevant)
            totals[f"ndcg@{k}"] += ndcg_at_k(ranked, relevant, k)

    return {name: value / len(question_ids) for name, value in totals.items()}

