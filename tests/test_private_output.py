import pytest

from video_rag.evaluation.private_output import check_immutable, private_output, write_immutable


def test_private_output_stays_in_approved_roots_and_does_not_overwrite(tmp_path):
    root = tmp_path / "artifacts"
    target = root / "annotations" / "questions.jsonl"
    assert private_output(target, (root,)) == target.resolve()
    with pytest.raises(ValueError):
        private_output(root / ".." / "public.jsonl", (root,))
    with pytest.raises(ValueError):
        private_output(root, (root,))
    write_immutable(target, "private\n")
    write_immutable(target, "private\n")
    with pytest.raises(ValueError):
        check_immutable(target, "different\n")
    with pytest.raises(ValueError):
        write_immutable(target, "different\n")
    assert target.read_text(encoding="utf-8") == "private\n"
