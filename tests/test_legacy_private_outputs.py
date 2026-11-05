import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('script', ['generate_question_candidates.py', 'prepare_review_queue.py'])
def test_legacy_question_outputs_reject_tracked_paths_before_loading_inputs(script, tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(root / 'src')}
    result = subprocess.run([sys.executable, str(root / 'scripts' / script), '--output', str(root / 'data' / 'evaluation' / 'unsafe.jsonl')],
                            cwd=root, env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert 'Private evaluation outputs must be under' in result.stderr
