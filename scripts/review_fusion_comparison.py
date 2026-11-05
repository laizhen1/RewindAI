"""Review a private timestamp-comparison report beside its bound local video."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from flask import Flask, jsonify, render_template, request, send_file

from video_rag.evaluation.private_output import PRIVATE_ROOTS, private_output
from video_rag.evaluation.provenance import fingerprint


def create_app(video: Path, report_path: Path, *, allowed_roots=PRIVATE_ROOTS) -> Flask:
    video = Path(video).expanduser().resolve()
    report_path = private_output(report_path, roots=allowed_roots)
    if not video.is_file() or video.suffix.lower() not in {'.mp4', '.webm', '.ogv', '.mov', '.m4v'}:
        raise ValueError('Provide an existing local video')
    if not report_path.is_file() or report_path.suffix.lower() != '.json' or report_path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError('Provide a local detailed comparison JSON under an ignored directory')
    raw = report_path.read_bytes()
    report = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(report, dict) or report.get('split') not in (None, 'development', 'validation', 'test'):
        raise ValueError('Invalid report split')
    if report.get('split') == 'test' and report.get('test_declaration') != 'frozen_test_declared_not_independently_verified':
        raise ValueError('Test reports require a frozen-test declaration')
    comparison = report.get('comparison')
    rows = comparison.get('results') if isinstance(comparison, dict) else None
    if not isinstance(rows, list) or not rows or len(rows) > 1000 or any(not isinstance(row, dict) for row in rows):
        raise ValueError('A detailed report with question results is required; do not use aggregate-only output')
    hashes = report.get('input_sha256')
    initial = video.stat()
    if not isinstance(hashes, dict) or hashes.get('video') != fingerprint(video):
        raise ValueError('Report does not match the selected video')
    stat = video.stat()
    if (initial.st_size, initial.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError('Video changed while loading the report')
    version = (stat.st_size, stat.st_mtime_ns)
    digest = hashlib.sha256(raw).hexdigest()
    app = Flask(__name__, template_folder=str(ROOT / 'src/video_rag/api/templates'))

    def unchanged():
        try:
            st = video.stat()
            return ((st.st_size, st.st_mtime_ns) == version and
                    hashlib.sha256(report_path.read_bytes()).hexdigest() == digest)
        except OSError:
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
        return render_template('fusion_review.html')

    @app.get('/video')
    def media():
        if not unchanged():
            return jsonify(error='Video or report changed; restart'), 409
        return send_file(video, conditional=True)

    @app.get('/api/comparison')
    def comparison_rows():
        if not unchanged():
            return jsonify(error='Video or report changed; restart'), 409
        return jsonify(rows=rows, split=report.get('split'), min_coverage=report.get('min_coverage'),
                       visual_min_score=report.get('visual_min_score_experimental'),
                       transcript_retrieval=report.get('transcript_retrieval'))

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5098)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Port must be between 1024 and 65535')
    try:
        app = create_app(args.video, args.report)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        parser.exit(2, f'Could not open local comparison: {exc}\n')
    print(f'Open http://127.0.0.1:{args.port}/ on this computer.', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
