import pytest

from scripts.compare_time_retrieval import compare


class Response:
    def __init__(self, evidence):
        self.evidence = evidence
    def to_dict(self):
        return {"abstained": not self.evidence, "evidence": self.evidence}


class Pipeline:
    def __init__(self, answers):
        self.answers = answers
    def ask(self, question):
        return Response(self.answers[question])


def test_paired_counts_keep_wrong_evidence_and_absences_separate():
    hit = [{"start_time": 0, "end_time": 5}]
    wrong = [{"start_time": 10, "end_time": 15}]
    questions = [{"question": q, "evidence_start_seconds": 0, "evidence_end_seconds": 5} for q in ("a", "b", "c", "d")]
    questions += [{"question": q, "expected_answerable": False} for q in ("e", "f", "g", "h")]
    keyword = Pipeline({"a": hit, "b": hit, "c": wrong, "d": [], "e": [], "f": [], "g": hit, "h": hit})
    semantic = Pipeline({"a": hit, "b": wrong, "c": hit, "d": [], "e": [], "f": hit, "g": [], "h": hit})
    result = compare(questions, keyword, semantic)
    assert result["paired_answerable"] == {"both_hit": 1, "keyword_only_hit": 1, "semantic_only_hit": 1, "neither_hit": 1}
    assert result["paired_unanswerable"] == {"both_abstain": 1, "keyword_only_abstains": 1, "semantic_only_abstains": 1, "neither_abstains": 1}
    assert result["keyword"]["metrics"]["fp"] == 3
    with pytest.raises(ValueError):
        compare([], keyword, semantic)
