import pytest

from scripts.evaluate_transcription import evaluate


def test_alignment_counts_substitution_insertion_and_deletion():
    result = evaluate("alpha beta gamma delta", "alpha beet gamma extra")
    assert result["matches"] == 2
    assert result["substitutions"] == 2
    assert result["wer"] == 0.5
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    insertion = evaluate("one two", "one extra two")
    assert insertion["insertions"] == 1
    assert insertion["wer"] == 0.5
    deletion = evaluate("one two three", "one three")
    assert deletion["deletions"] == 1


def test_cjk_and_empty_input():
    assert evaluate("你好世界", "你好世界")["f1"] == 1
    assert evaluate("hello", "")["precision"] is None
    with pytest.raises(ValueError):
        evaluate("", "text")

