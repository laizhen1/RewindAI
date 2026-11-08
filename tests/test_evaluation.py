import unittest

import pytest

from video_rag.evaluation.retrieval import evaluate_retrieval, hit_at_k, ndcg_at_k, recall_at_k


class EvaluationTests(unittest.TestCase):
    def test_retrieval_metrics(self):
        metrics = evaluate_retrieval(
            predictions={"q1": ["x", "a"], "q2": ["b"]},
            ground_truth={"q1": ["a"], "q2": ["b"]},
            cutoffs=(1, 2),
        )
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.75)


def test_recall_fraction_and_hit_rate_are_distinct_for_multiple_relevant_segments():
    ranked, relevant = ['a', 'x', 'y', 'z', 'n', 'b'], {'a', 'b', 'c'}
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(1 / 3)
    assert hit_at_k(ranked, relevant, 5) == 1
    result = evaluate_retrieval({'q': ranked}, {'q': relevant}, cutoffs=(1, 5))
    assert result['recall@5'] == pytest.approx(1 / 3)
    assert result['hit@5'] == 1 and result['mrr@5'] == 1
    result = evaluate_retrieval({'q': ['x', 'y', 'z', 'u', 'v', 'a']}, {'q': relevant}, cutoffs=(5,))
    assert result['mrr@5'] == 0 and result['mrr'] == pytest.approx(1 / 6)


def test_retrieval_metrics_reject_empty_relevance_and_duplicate_rankings():
    with pytest.raises(ValueError, match='relevant'):
        evaluate_retrieval({'q': []}, {'q': []})
    with pytest.raises(ValueError, match='duplicate'):
        evaluate_retrieval({'q': ['a', 'a']}, {'q': ['a']})
    with pytest.raises(ValueError, match='cutoffs'):
        evaluate_retrieval({'q': ['a']}, {'q': ['a']}, cutoffs=(0,))
    assert ndcg_at_k(['a', 'a'], {'a'}, 2) == 1
