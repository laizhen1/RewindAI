"""Create an answer-review draft from locally bound question labels and open its local review page."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.evaluate_local_answers import prepare_review
from scripts.prepare_temporal_questions import prepare
from scripts.review_answers import create_review_app
from video_rag.adapters.local_answer import LocalAnswerGenerator
from video_rag.evaluation.private_output import PRIVATE_ROOTS, private_output
from video_rag.evaluation.provenance import fingerprint
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


def create_draft(video: Path, subtitles: Path, labels: Path, video_id: str, split: str,
                 model: Path, output_dir: Path, *, allowed_roots=PRIVATE_ROOTS) -> Path:
    if split not in {'development', 'validation'}:
        raise ValueError('Use development or validation; final test is reserved')
    labels = private_output(labels, roots=allowed_roots)
    output_dir = private_output(output_dir, roots=allowed_roots)
    binding_path = labels.with_suffix(labels.suffix + '.source.json')
    sources = {'video': video, 'subtitles': subtitles, 'annotations': labels,
               'annotation_binding': binding_path, 'model': model}
    before = {key: fingerprint(path) for key, path in sources.items()}
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    if binding != {'video_sha256': before['video'], 'video_id': video_id, 'split': split}:
        raise ValueError('Labels do not match this video and split')
    rows = list(csv.DictReader(io.StringIO(labels.read_text(encoding='utf-8-sig'), newline='')))
    if not rows:
        raise ValueError('Create development or validation questions first')
    data = prepare(rows, video_id, split)
    generator = LocalAnswerGenerator(model)
    pipeline = build_transcript_segments(segments_from_subtitles(video, subtitles), generator=generator)
    review_rows = prepare_review(pipeline, data)
    if any(fingerprint(path) != before[key] for key, path in sources.items()):
        raise ValueError('An input changed during the run; no review was saved')
    report = {'status': 'unreviewed_answer_drafts', 'binding': 'video_matched',
              'input_sha256': before, 'split': split, 'rows': review_rows,
              'warning': 'Labels and answers are unverified drafts. Review against the original video.'}
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f'answer-draft-{uuid.uuid4().hex}.json'
    with target.open('x', encoding='utf-8', newline='\n') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('video', 'subtitles', 'labels', 'model'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--split', choices=['development', 'validation'], required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/private_lecture_review/answer_runs')
    parser.add_argument('--port', type=int, default=5097)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Port must be between 1024 and 65535')
    try:
        print('Preparing local answer drafts; no lecture material is uploaded. This may take time.', flush=True)
        target = create_draft(args.video, args.subtitles, args.labels, args.video_id, args.split,
                              args.model, args.output_dir)
        app = create_review_app(args.video, target)
    except (OSError, UnicodeError, ValueError, TypeError, RuntimeError, csv.Error) as exc:
        parser.exit(2, f'Could not prepare local review: {exc}\n')
    print(f'Open http://127.0.0.1:{args.port}/ to review the unverified answers.', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
