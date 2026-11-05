"""Declare a local answer test frozen after human review; declaration is not independent proof."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.prepare_temporal_questions import write_output
from scripts.validate_frozen_labels import validate_test_labels
from video_rag.evaluation.frozen_answers import config, input_paths, snapshot
from video_rag.evaluation.private_output import private_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'subtitles', 'labels', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--semantic-model', type=Path)
    parser.add_argument('--semantic-threshold', type=float, default=0.4)
    parser.add_argument('--confirm-labels-reviewed', action='store_true',
                        help='Declare that references and checked scopes were reviewed against the video')
    args = parser.parse_args()
    try:
        if not args.confirm_labels_reviewed:
            raise ValueError('A human must review the test labels before explicitly declaring them frozen')
        output = private_output(args.output)
        if output.suffix.lower() != '.json' or output.exists():
            raise ValueError('Choose a new local JSON manifest')
        labels = private_output(args.labels)
        settings = config(args.video_id, args.semantic_model, args.semantic_threshold)
        paths = input_paths(args.video, args.subtitles, labels, args.model, args.semantic_model)
        before = snapshot(paths, settings)
        validate_test_labels(args.video, labels, args.video_id, before['input_sha256']['video'])
        if snapshot(paths, settings) != before:
            raise ValueError('An input, code file or runtime changed while freezing')
        manifest = {'status': 'frozen_answer_test_declared_not_independently_verified', **before,
                    'warning': 'A human declaration is not independent verification of labels, '
                               'video-disjoint splits, or correctness. Keep this manifest private.'}
        write_output(output, manifest, (ROOT / 'artifacts', ROOT / 'local_data'))
    except (ImportError, OSError, UnicodeError, KeyError, TypeError, ValueError, RuntimeError, csv.Error) as exc:
        parser.exit(2, f'Could not freeze answer test: {exc}\n')
    print('Answer-test manifest saved locally. The review declaration was not independently verified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
