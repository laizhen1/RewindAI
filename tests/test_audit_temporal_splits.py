import pytest

from scripts.audit_temporal_splits import audit


def data(video, split, question, digest="a"):
    return {"video_id": video, "split": split, "questions": [{"question": question}],
            "source_sha256": {"video": digest * 64, "subtitles": "f" * 64}}


def test_detects_cross_split_id_hash_and_exact_questions_without_disclosing_content():
    rows = [data("v1", "development", "Private question", "a"),
            data("v1", "test", "  PRIVATE   QUESTION ", "a")]
    result = audit(rows)
    assert result["video_ids_across_splits"] == 1
    assert result["video_hashes_across_splits"] == 1
    assert result["exact_questions_across_splits"] == 1
    assert not result["no_detected_leakage"]
    assert "Private question" not in str(result)


def test_disjoint_bound_sets_pass_and_missing_binding_is_inconclusive():
    rows = [data("v1", "development", "question A", "a"), data("v2", "test", "question B", "b")]
    assert audit(rows)["no_detected_leakage"]
    assert not audit(rows[:1])["no_detected_leakage"]
    rows[0].pop("source_sha256")
    result = audit(rows)
    assert result["unbound_datasets"] == 1
    assert not result["no_detected_leakage"]
    rows[0]["split"] = None
    with pytest.raises(ValueError):
        audit(rows)
