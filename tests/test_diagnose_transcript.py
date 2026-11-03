from scripts.diagnose_transcript import coverage, diagnose, metrics, overlap


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
    assert result["best_hit_reference_coverage"] == 1.0


def test_coverage_and_unanswerable_result():
    assert coverage({"start_time": 19, "end_time": 30}, 10, 20) == 0.1

    class Pipeline:
        def ask(self, question):
            class Result:
                def to_dict(self):
                    return {"abstained": True, "evidence": []}
            return Result()

    row = diagnose(Pipeline(), [{"question": "Not in video?", "expected_answerable": False}])[0]
    assert row["correct_abstention"] is True
    assert "any_hit_overlaps" not in row


def test_metrics_count_wrong_evidence_as_false_positive_and_false_negative():
    rows = [
        {"expected_answerable": True, "abstained": False, "first_hit_reference_coverage": 0.9},
        {"expected_answerable": True, "abstained": False, "first_hit_reference_coverage": 0.1},
        {"expected_answerable": False, "abstained": False},
        {"expected_answerable": False, "abstained": True},
        {"expected_answerable": True, "abstained": True},
    ]
    result = metrics(rows)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 2, 2, 1)
    assert result["precision"] == result["recall"] == result["f1"] == 1 / 3

