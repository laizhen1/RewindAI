from scripts.diagnose_transcript import diagnose, overlap


def test_overlap_requires_positive_duration():
    assert overlap({"start_time": 10, "end_time": 20}, 19, 30)
    assert not overlap({"start_time": 10, "end_time": 20}, 20, 30)


def test_diagnostic_reports_first_and_any_overlap():
    class Result:
        def to_dict(self):
            return {"abstained": False, "evidence": [
                {"start_time": 0, "end_time": 5},
                {"start_time": 10, "end_time": 20},
            ]}

    class Pipeline:
        def ask(self, question):
            return Result()

    result = diagnose(Pipeline(), [{"question": "Example?", "evidence_start_seconds": 12,
                                   "evidence_end_seconds": 14}])[0]
    assert result["first_hit_overlaps"] is False
    assert result["any_hit_overlaps"] is True

