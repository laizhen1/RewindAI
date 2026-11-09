"""Review relevant transcript segments locally without showing retrieval rankings."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from flask import Flask, jsonify, render_template, request, send_file

from scripts.prepare_temporal_questions import prepare
from video_rag.api.visual_timeline import cover_subtitle_gaps, video_duration_seconds
from video_rag.evaluation.private_output import PRIVATE_ROOTS, private_output
from video_rag.evaluation.provenance import fingerprint
from video_rag.transcript_baseline import segments_from_subtitles


def segment_fingerprint(segments) -> str:
    """Bind relevance labels to exact segment IDs, times and transcript content."""
    values = [{'id': item.segment_id, 'start': item.start_time, 'end': item.end_time,
               'text': item.transcript} for item in segments]
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def create_relevance_app(video: Path, subtitles: Path, labels: Path, output_dir: Path,
                         video_id: str, split: str, duration: float, *, cover_gaps=False, allowed_roots=PRIVATE_ROOTS) -> Flask:
    """Bind a private, single-video review to its exact source files."""
    if split not in {'development', 'validation', 'test'} or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Provide a valid split and video duration')
    video, subtitles = Path(video).expanduser().resolve(), Path(subtitles).expanduser().resolve()
    labels = private_output(labels, roots=allowed_roots)
    output_dir = private_output(output_dir, roots=allowed_roots)
    binding_path = labels.with_suffix(labels.suffix + '.source.json')
    paths = {'video': video, 'subtitles': subtitles, 'annotations': labels, 'annotation_binding': binding_path}
    hashes = {name: fingerprint(path) for name, path in paths.items()}
    binding = json.loads(binding_path.read_text(encoding='utf-8-sig'))
    if binding != {'video_sha256': hashes['video'], 'video_id': video_id, 'split': split}:
        raise ValueError('Labels do not match the selected video and split')
    with labels.open(encoding='utf-8-sig', newline='') as source:
        data = prepare(list(csv.DictReader(source)), video_id, split)
    questions = [item for item in data['questions'] if item['expected_answerable']]
    if not questions:
        raise ValueError('At least one answerable question is required')
    if any(item['evidence_end_seconds'] > duration for item in questions):
        raise ValueError('A reference range exceeds the video duration')
    segments = segments_from_subtitles(video, subtitles)
    if max(item.end_time for item in segments) > duration + 1:
        raise ValueError('Subtitle timestamps exceed the video duration')
    if cover_gaps:
        segments = cover_subtitle_gaps(segments, duration)
    if hashes != {name: fingerprint(path) for name, path in paths.items()}:
        raise ValueError('A source changed while preparing review')
    segment_ids = {item.segment_id for item in segments}
    segment_sha256 = segment_fingerprint(segments)
    question_ids = {item['question_id'] for item in questions}
    def path_versions():
        result = {}
        for name, path in paths.items():
            stat = path.stat()
            result[name] = (stat.st_size, stat.st_mtime_ns)
        return result

    versions = path_versions()
    video_stat = video.stat()
    video_version = (video_stat.st_size, video_stat.st_mtime_ns)
    app = Flask(__name__, template_folder=str(ROOT / 'src/video_rag/api/templates'))
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

    def video_unchanged() -> bool:
        try:
            stat = video.stat()
            return video.is_file() and (stat.st_size, stat.st_mtime_ns) == video_version
        except OSError:
            return False

    def sources_unchanged() -> bool:
        try:
            return (video_unchanged() and path_versions() == versions
                    and hashes == {name: fingerprint(path) for name, path in paths.items()}
                    and path_versions() == versions and video_unchanged())
        except (OSError, ValueError):
            return False

    def normalize_rows(rows):
        if not isinstance(rows, list) or len(rows) != len(questions):
            raise ValueError('Provide one entry for each answerable question')
        normalized = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'question_id', 'relevant_segment_ids', 'verified', 'exhaustive'}:
                raise ValueError('Invalid relevance entry')
            qid, ids = row['question_id'], row['relevant_segment_ids']
            if (not isinstance(qid, str) or qid not in question_ids or qid in normalized
                    or not isinstance(ids, list) or any(not isinstance(value, str) for value in ids)
                    or len(set(ids)) != len(ids) or not set(ids) <= segment_ids
                    or type(row['verified']) is not bool or type(row['exhaustive']) is not bool
                    or (row['verified'] and not ids) or (row['exhaustive'] and not row['verified'])):
                raise ValueError('Invalid relevance labels or review declaration')
            normalized[qid] = {'question_id': qid, 'relevant_segment_ids': ids,
                               'verified': row['verified'], 'exhaustive': row['exhaustive']}
        return [normalized[item['question_id']] for item in questions]

    @app.before_request
    def local_only():
        try:
            if urlsplit(f'http://{request.host}').hostname not in {'127.0.0.1', 'localhost', '::1'}:
                return jsonify(error='Local access only'), 403
            origin = request.headers.get('Origin')
            if origin:
                parsed = urlsplit(origin)
                if (parsed.scheme, parsed.netloc.lower()) != (request.scheme, request.host.lower()):
                    return jsonify(error='Cross-origin access is not allowed'), 403
        except ValueError:
            return jsonify(error='Invalid request origin'), 403

    @app.after_request
    def no_cache(response):
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def index():
        return render_template('relevance_review.html')

    @app.get('/video')
    def media():
        if not video_unchanged():
            return jsonify(error='Video changed; restart review'), 409
        return send_file(video, conditional=True)

    @app.get('/api/data')
    def review_data():
        if not sources_unchanged():
            return jsonify(error='A source changed; restart review'), 409
        return jsonify(questions=questions, segments=[{'id': item.segment_id, 'start': item.start_time,
                                                        'end': item.end_time, 'text': item.transcript}
                                                       for item in segments], split=split, cover_gaps=cover_gaps,
                       unanswerable_excluded=len(data['questions']) - len(questions))

    @app.post('/api/save')
    def save():
        payload = request.get_json(silent=True)
        rows = payload.get('judgments') if isinstance(payload, dict) else None
        try:
            normalized = normalize_rows(rows)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if not sources_unchanged():
            return jsonify(error='A source changed; no progress was saved'), 409
        output_dir.mkdir(parents=True, exist_ok=True)
        target = private_output(output_dir / f'relevance-{uuid.uuid4().hex}.json', roots=allowed_roots)
        report = {'schema_version': 1, 'status': 'relevance_draft_not_independently_verified',
                  'source_sha256': hashes, 'video_id': video_id, 'split': split,
                  'segment_count': len(segments), 'segment_sha256': segment_sha256, 'judgments': normalized}
        with target.open('x', encoding='utf-8', newline='\n') as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
        return jsonify(filename=target.name), 201

    @app.get('/api/latest')
    def latest():
        if not sources_unchanged():
            return jsonify(error='A source changed; restart review'), 409
        candidates = []
        if output_dir.is_dir():
            for path in output_dir.glob('relevance-*.json'):
                try:
                    path = private_output(path, roots=allowed_roots)
                    if path.is_file() and path.stat().st_size <= 2 * 1024 * 1024:
                        candidates.append((path.stat().st_mtime_ns, path))
                except (OSError, ValueError):
                    continue
        for _, path in sorted(candidates, reverse=True)[:100]:
            try:
                report = json.loads(path.read_text(encoding='utf-8-sig'))
                if not isinstance(report, dict):
                    continue
                if (report.get('schema_version') == 1 and report.get('status') == 'relevance_draft_not_independently_verified'
                        and report.get('source_sha256') == hashes
                        and report.get('video_id') == video_id and report.get('split') == split
                        and report.get('segment_count') == len(segments)
                        and report.get('segment_sha256') == segment_sha256):
                    restored = normalize_rows(report['judgments'])
                    if not sources_unchanged():
                        return jsonify(error='A source changed; restart review'), 409
                    return jsonify(judgments=restored)
            except (OSError, UnicodeError, ValueError, TypeError, KeyError):
                continue
        return jsonify(error='No matching saved progress'), 404

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'subtitles', 'labels', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--split', choices=('development', 'validation', 'test'), required=True)
    parser.add_argument('--port', type=int, default=5098)
    parser.add_argument('--cover-gaps', action='store_true', help='Add visual-only windows between subtitle segments')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Port must be between 1024 and 65535')
    try:
        duration = video_duration_seconds(args.video)
        app = create_relevance_app(args.video, args.subtitles, args.labels, args.output_dir,
                                   args.video_id, args.split, duration, cover_gaps=args.cover_gaps)
    except (ImportError, OSError, ValueError, TypeError, csv.Error) as exc:
        parser.exit(2, f'Cannot start local relevance review: {exc}\n')
    print(f'Open http://127.0.0.1:{args.port}/; keep private data on this computer.', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
