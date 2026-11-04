import pytest

from scripts.prepare_temporal_questions import prepare, write_output


def row(**updates):
    data = {"question_id": "q1", "video_id": "v1", "question_type": "audio", "answerable": "true",
            "question": "What?", "reference_answer": "Answer", "evidence_start_seconds": "2", "evidence_end_seconds": "5",
            "checked_scope": "", "annotation_status": "verified", "split": "test"}
    return data | updates


def test_conversion_stays_draft_and_filters_video_and_split():
    data = prepare([row(), row(question_id="q2", video_id="v2"), row(question_id="q3", split="validation"),
                    row(question_id="q4", answerable="false", evidence_start_seconds="", evidence_end_seconds="", checked_scope="full video")],
                   "v1", "test")
    assert data["status"] == "csv_import_unverified_draft"
    assert [q["question_id"] for q in data["questions"]] == ["q1", "q4"]
    assert data["questions"][1]["checked_scope"] == "full video"
    assert "verification_status" not in data["questions"][0]


def test_conversion_rejects_incomplete_or_inconsistent_rows():
    for bad in [row(answerable="false"), row(answerable="maybe"), row(evidence_start_seconds="nan"),
                row(evidence_end_seconds="1"), row(question_type="unsupported")]:
        with pytest.raises(ValueError):
            prepare([bad], "v1")
    with pytest.raises(ValueError):
        prepare([row(), row()], "v1")


def test_local_output_only_and_never_overwrites(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    target = root / "private" / "questions.json"
    write_output(target, prepare([row()], "v1"), (root,))
    assert target.is_file()
    with pytest.raises(FileExistsError):
        write_output(target, {}, (root,))
    with pytest.raises(ValueError):
        write_output(tmp_path / "outside.json", {}, (root,))
