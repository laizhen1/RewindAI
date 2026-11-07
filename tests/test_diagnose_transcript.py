from scripts.diagnose_transcript import (
    coverage,
    coverage_feasibility,
    diagnose,
    latency_summary,
    metrics,
    metrics_by_question_type,
    operating_metrics,
    overlap,
)


def test_overlap_requires_positive_duration():
    assert overlap({"start_time": 10, "end_time": 20}, 19, 30)
    assert not overlap({"start_time": 10, "end_time": 20}, 20, 30)


def test_coverage_feasibility_checks_actual_overlap_and_excludes_negative_questions():
    from types import SimpleNamespace

    questions = [
        {"evidence_start_seconds": 0, "evidence_end_seconds": 10},
        {"evidence_start_seconds": 0, "evidence_end_seconds": 100},
        {"evidence_start_seconds": 200, "evidence_end_seconds": 210},
        {"expected_answerable": False},
    ]
    segments = [SimpleNamespace(start_time=0, end_time=5),
                SimpleNamespace(start_time=100, end_time=150)]
    assert coverage_feasibility(questions, segments) == {
        "answerable_questions": 3, "structurally_unattainable": 2,
        "reference_coverage_threshold": 0.5,
    }
    assert coverage_feasibility(questions, [])['structurally_unattainable'] == 3


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
    assert result["query_latency_ms"] >= 0


def test_latency_summary_reports_nearest_rank_p95_and_empty():
    assert latency_summary([]) == {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    summary = latency_summary([{"query_latency_ms": value} for value in range(1, 21)])
    assert summary == {"count": 20, "mean_ms": 10.5, "median_ms": 10.5, "p95_ms": 19}


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


def test_f1_is_zero_for_missed_positive_and_undefined_for_no_positives():
    missed = metrics([{"expected_answerable": True, "abstained": True}])
    assert missed["precision"] is None and missed["recall"] == 0 and missed["f1"] == 0
    empty = metrics([{"expected_answerable": False, "abstained": True}])
    assert empty["precision"] is None and empty["recall"] is None and empty["f1"] is None


def test_separate_temporal_hits_and_unanswerable_rejection_rates():
    rows = [
        {"expected_answerable": True, "abstained": False, "first_hit_reference_coverage": .2, "best_hit_reference_coverage": .8, "question_type": "audio"},
        {"expected_answerable": True, "abstained": True, "first_hit_reference_coverage": 0, "best_hit_reference_coverage": 0, "question_type": "visual"},
        {"expected_answerable": False, "abstained": False, "question_type": "visual"},
        {"expected_answerable": False, "abstained": True},
    ]
    result = operating_metrics(rows)
    assert result["top1_temporal_hits"] == 0
    assert result["any_returned_temporal_hits"] == 1
    assert result["any_returned_temporal_hit_rate"] == .5
    assert result["answerable_abstention_rate"] == .5
    assert result["unanswerable_abstention_rate"] == .5
    assert result["unanswerable_candidate_rate"] == .5
    groups = metrics_by_question_type(rows)
    assert groups["audio"]["unanswerable_abstention_rate"] is None
    assert groups["unlabeled"]["unanswerable_abstention_rate"] == 1
    assert groups["visual"]["answerable_abstention_rate"] == 1
