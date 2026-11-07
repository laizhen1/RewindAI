"""Regression checks for edits made while review requests are pending."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_answer_review_preserves_new_choices_during_requests():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser-script regression test")
    subprocess.run(
        [node, str(Path(__file__).with_name("answer_review_race.js"))],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
