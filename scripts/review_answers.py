"""Review local generated answers alongside the original video, without uploading."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from flask import Flask, jsonify, render_template, request, send_file

from scripts.score_answer_review import summarize
from video_rag.evaluation.private_output import PRIVATE_ROOTS, private_output
from video_rag.evaluation.provenance import fingerprint


def create_review_app(video: Path, review: Path, *, allowed_roots=PRIVATE_ROOTS,
                      frozen_test_manifest: Path | None = None) -> Flask:
    video = Path(video).expanduser().resolve()
    review = private_output(review, roots=allowed_roots)
    if not video.is_file() or video.suffix.lower() not in {'.mp4', '.webm', '.ogv', '.mov', '.m4v'}:
        raise ValueError('Provide an existing local video')
    if not review.is_file() or review.suffix.lower() != '.json' or review.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('Provide a local answer-review JSON under an ignored directory')
    raw = review.read_bytes()
    report = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(report, dict) or report.get('status') != 'unreviewed_answer_drafts':
        raise ValueError('Expected an answer review draft')
    if report.get('split') == 'test' and report.get('test_declaration') != 'frozen_answer_test_declared_not_independently_verified':
        raise ValueError('Test review requires a frozen-answer-test declaration')
    manifest = None
    if report.get('split') == 'test':
        from video_rag.evaluation.frozen_answers import verify_review_manifest

        if frozen_test_manifest is None:
            raise ValueError('Test review requires its frozen-answer-test manifest')
        manifest = private_output(frozen_test_manifest, roots=allowed_roots)
        verify_review_manifest(report, manifest)
    elif frozen_test_manifest is not None:
        raise ValueError('A frozen-answer-test manifest is only used for test reviews')
    rows = report.get('rows')
    if not isinstance(rows, list) or not rows or len(rows) > 1000 or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Invalid review rows')
    hashes = report.get('input_sha256')
    if not isinstance(hashes, dict) or hashes.get('video') != fingerprint(video):
        raise ValueError('Review is not bound to this video')
    stat = video.stat()
    video_version = (stat.st_size, stat.st_mtime_ns)
    digest = hashlib.sha256(raw).hexdigest()
    app = Flask(__name__, template_folder=str(ROOT / 'src/video_rag/api/templates'))
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024

    def unchanged():
        try:
            st = video.stat()
            return ((st.st_size, st.st_mtime_ns) == video_version and
                    hashlib.sha256(review.read_bytes()).hexdigest() == digest and
                    (manifest is None or fingerprint(manifest) == hashes['frozen_manifest']))
        except (OSError, ValueError):
            return False

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
        return render_template('answer_review.html')

    @app.get('/video')
    def media():
        if not unchanged():
            return jsonify(error='Video or review changed; restart'), 409
        return send_file(video, conditional=True)

    @app.get('/api/review')
    def get_review():
        if not unchanged():
            return jsonify(error='Video or review changed; restart'), 409
        return jsonify(rows=rows, split=report.get('split'), binding=report.get('binding'),
                       retrieval_config=report.get('retrieval_config'))

    @app.post('/api/progress')
    def save_progress():
        if not unchanged():
            return jsonify(error='Video or review changed; restart'), 409
        data = request.get_json(silent=True)
        judgments = data.get('judgments') if isinstance(data, dict) else None
        if not isinstance(judgments, list) or len(judgments) != len(rows):
            return jsonify(error='Invalid review progress'), 400
        reviewed = json.loads(raw.decode('utf-8-sig'))
        for row, decision in zip(reviewed['rows'], judgments, strict=True):
            if not isinstance(decision, dict) or set(decision) != {'answer', 'evidence'}:
                return jsonify(error='Invalid verdict'), 400
            answer, evidence = decision['answer'], decision['evidence']
            if (not isinstance(answer, str) or answer not in {'', 'correct', 'incorrect', 'abstain'} or
                    not isinstance(evidence, str) or evidence not in {'', 'supported', 'unsupported', 'not_applicable'}):
                return jsonify(error='Invalid verdict'), 400
            row['human_answer_verdict'] = answer or None
            row['human_evidence_verdict'] = evidence or None
        if not unchanged():
            return jsonify(error='Video or review changed; restart'), 409
        reviewed.pop('human_review_declaration', None)
        reviewed['human_review_state'] = 'in_progress_not_verified'
        target = review.parent / f'progress-{uuid.uuid4().hex}.json'
        with target.open('x', encoding='utf-8', newline='\n') as out:
            json.dump(reviewed, out, ensure_ascii=False, indent=2)
            out.write('\n')
        return jsonify(filename=target.name), 201

    @app.post('/api/save')
    def save():
        if not unchanged():
            return jsonify(error='Video or review changed; restart'), 409
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or data.get('confirm_human_review') is not True:
            return jsonify(error='Human review must be declared'), 400
        judgments = data.get('judgments')
        if not isinstance(judgments, list) or len(judgments) != len(rows):
            return jsonify(error='A verdict is required for every row'), 400
        reviewed = json.loads(raw.decode('utf-8-sig'))
        for row, decision in zip(reviewed['rows'], judgments, strict=True):
            if not isinstance(decision, dict) or set(decision) != {'answer', 'evidence'}:
                return jsonify(error='Invalid verdict'), 400
            row['human_answer_verdict'] = decision['answer']
            row['human_evidence_verdict'] = decision['evidence']
        try:
            if manifest is not None:
                verify_review_manifest(reviewed, manifest)
            scores = summarize(reviewed)
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
        if not unchanged():
            return jsonify(error='Video or review changed; restart'), 409
        reviewed['human_review_state'] = 'complete_not_independently_verified'
        reviewed['human_review_declaration'] = 'declared_by_local_user_not_independently_verified'
        target = review.parent / f'reviewed-{uuid.uuid4().hex}.json'
        with target.open('x', encoding='utf-8', newline='\n') as out:
            json.dump(reviewed, out, ensure_ascii=False, indent=2)
            out.write('\n')
        return jsonify(filename=target.name, metrics=scores), 201

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--frozen-test-manifest', type=Path, help='Required for a frozen answer-test report')
    parser.add_argument('--port', type=int, default=5097)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Port must be between 1024 and 65535')
    try:
        app = create_review_app(args.video, args.review, frozen_test_manifest=args.frozen_test_manifest)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.error(str(exc))
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
