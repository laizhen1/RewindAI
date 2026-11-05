import unittest

from scripts.evaluate_answers import index_answers
from video_rag.evaluation.answer import evaluate_answers, exact_match, token_f1


class AnswerEvaluationTests(unittest.TestCase):
    def test_exact_match_ignores_spacing_and_case(self):
        self.assertEqual(exact_match("CLIP 模型", "clip模型"), 1.0)

    def test_token_f1_partial_match(self):
        self.assertGreater(token_f1("飞机在机场降落", "飞机在机场"), 0.0)

    def test_english_f1_uses_words_not_character_anagrams(self):
        self.assertEqual(token_f1("cat", "act"), 0.0)
        self.assertAlmostEqual(token_f1("red car", "blue car"), 0.5)
        self.assertEqual(exact_match("the rapist", "therapist"), 0.0)

    def test_empty_references_are_not_scored_as_correct(self):
        with self.assertRaisesRegex(ValueError, "unanswerable"):
            evaluate_answers({"q1": ""}, {"q1": ""})
        self.assertEqual(evaluate_answers({}, {"q1": "answer"})["token_f1"], 0.0)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_answers([{"question_id": "q1", "answer": "one"},
                           {"question_id": "q1", "answer": "two"}])

    def test_evaluate_answers(self):
        result = evaluate_answers({"q1": "CLIP"}, {"q1": "clip"})
        self.assertEqual(result["exact_match"], 1.0)

