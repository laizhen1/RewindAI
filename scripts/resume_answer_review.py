"""Resume a local development or validation answer review for the same video."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.review_answers import create_review_app
from video_rag.evaluation.private_output import PRIVATE_ROOTS, private_output
from video_rag.evaluation.provenance import fingerprint


def find_review(video: Path, directory: Path, *, allowed_roots=PRIVATE_ROOTS) -> Path:
    directory = private_output(Path(directory) / '_directory_check', roots=allowed_roots).parent
    if not directory.is_dir() or not Path(video).is_file():
        raise ValueError('Local video or review directory not found')
    video_hash = fingerprint(video)
    for pattern in ('progress-*.json', 'answer-draft-*.json'):
        candidates = []
        for path in directory.glob(pattern):
            try:
                path = private_output(path, roots=allowed_roots)
                if path.is_file() and path.stat().st_size <= 20 * 1024 * 1024:
                    candidates.append((path.stat().st_mtime_ns, path))
            except (OSError, ValueError):
                continue
        for _, path in sorted(candidates, reverse=True):
            try:
                report = json.loads(path.read_text(encoding='utf-8-sig'))
                if (isinstance(report, dict) and report.get('status') == 'unreviewed_answer_drafts'
                        and report.get('split') in {'development', 'validation'}
                        and isinstance(report.get('input_sha256'), dict)
                        and report['input_sha256'].get('video') == video_hash
                        and isinstance(report.get('rows'), list) and report['rows']):
                    return path
            except (OSError, UnicodeError, ValueError, TypeError):
                continue
    raise ValueError('No matching local development or validation review found')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--review-dir', type=Path, default=ROOT / 'artifacts/private_lecture_review/answer_runs')
    parser.add_argument('--port', type=int, default=5097)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Port must be between 1024 and 65535')
    try:
        review = find_review(args.video, args.review_dir)
        app = create_review_app(args.video, review)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        parser.exit(2, f'Could not resume local review: {exc}\n')
    print(f'Open http://127.0.0.1:{args.port}/ to continue the local answer review.', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
