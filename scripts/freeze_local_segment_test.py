"""Freeze declared, locally reviewed segment-retrieval test inputs before evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.evaluate_local_segment_retrieval import (
    comparison_gold_metadata,
    frozen_snapshot,
    prepare_inputs,
    verify_inputs,
)
from video_rag.evaluation.private_output import private_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'subtitles', 'labels', 'gold', 'semantic-model', 'visual-model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--other-gold', type=Path, action='append', required=True,
                        help='Development/validation relevance file; repeat for all videos used in tuning')
    parser.add_argument('--reranker-model', type=Path)
    parser.add_argument('--cover-gaps', action='store_true', help='Use the same visual-only gap windows as the review')
    parser.add_argument('--confirm-reviewed', action='store_true')
    parser.add_argument('--confirm-test-isolation', action='store_true',
                        help='Declare no test video or questions were used to select settings')
    args = parser.parse_args()
    args.split = 'test'
    try:
        if not args.confirm_reviewed or not args.confirm_test_isolation:
            raise ValueError('Declare reviewed labels and test isolation before freezing')
        target = private_output(args.output)
        if target.suffix.lower() != '.json' or target.exists():
            raise ValueError('Choose a new local JSON manifest')
        prepared = prepare_inputs(args)
        required = ['llama-cpp-python', 'numpy', 'onnxruntime', 'tokenizers', 'av']
        if args.reranker_model is not None:
            required += ['torch', 'transformers', 'accelerate']
        if any(prepared['dependency_versions'].get(name) is None for name in required):
            raise ValueError('Install the required local runtime dependencies before freezing')
        other = comparison_gold_metadata(args.other_gold, prepared['hashes']['video'])
        verify_inputs(prepared)
        if other != comparison_gold_metadata(args.other_gold, prepared['hashes']['video']):
            raise ValueError('Comparison relevance files changed while freezing')
        report = {'status': 'frozen_segment_test_declared_not_independently_verified',
                  **frozen_snapshot(prepared, other),
                  'warning': 'Declarations and split coverage are not independently verified. '
                             'Video-hash checks cannot detect near-duplicate videos or overlapping content. '
                             'Keep this manifest and fingerprints local.'}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as out:
            json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write('\n')
        print('Private test manifest saved. Declarations and split independence are not independently verified.')
        return 0
    except (ImportError, OSError, UnicodeError, KeyError, TypeError, ValueError, RuntimeError, csv.Error) as exc:
        parser.exit(2, f'Could not freeze local segment test: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
