"""Review and correct timed subtitles beside a local video without uploading them."""
from __future__ import annotations

import argparse
import hashlib
import html
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, jsonify, render_template, request, send_file

from video_rag.evaluation.private_output import PRIVATE_ROOTS, private_output
from video_rag.transcript_baseline import MAX_SUBTITLE_BYTES, parse_subtitles


def timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def create_review_app(video: Path, subtitles: Path, output_dir: Path, *, allowed_roots=PRIVATE_ROOTS) -> Flask:
    video, subtitles = Path(video).expanduser().resolve(), Path(subtitles).expanduser().resolve()
    output_dir = private_output(output_dir, roots=allowed_roots)
    if not video.is_file() or video.suffix.lower() not in {".mp4", ".webm", ".ogv", ".mov", ".m4v"}:
        raise ValueError("Provide an existing supported video")
    if not subtitles.is_file() or subtitles.suffix.lower() not in {".srt", ".vtt"}:
        raise ValueError("Provide existing timed subtitles")
    if subtitles.stat().st_size > MAX_SUBTITLE_BYTES:
        raise ValueError("Subtitle file too large")
    source = subtitles.read_bytes()
    cues = parse_subtitles(source.decode("utf-8-sig"))
    version = (video.stat().st_size, video.stat().st_mtime_ns)
    source_digest = hashlib.sha256(source).hexdigest()
    app = Flask(__name__, template_folder=str(ROOT / "src" / "video_rag" / "api" / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = MAX_SUBTITLE_BYTES

    def unchanged() -> bool:
        try:
            current = video.stat()
            return (current.st_size, current.st_mtime_ns) == version and hashlib.sha256(subtitles.read_bytes()).hexdigest() == source_digest
        except OSError:
            return False

    @app.before_request
    def local_only():
        try:
            if urlsplit(f"http://{request.host}").hostname not in {"localhost", "127.0.0.1", "::1"}:
                return jsonify(error="Local access only"), 403
            origin = request.headers.get("Origin")
            if origin:
                parsed = urlsplit(origin)
                if (parsed.scheme, parsed.netloc.lower()) != (request.scheme, request.host.lower()):
                    return jsonify(error="Cross-origin access is not allowed"), 403
        except ValueError:
            return jsonify(error="Invalid request origin"), 403

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("subtitle_review.html")

    @app.get("/video")
    def media():
        if not unchanged():
            return jsonify(error="Source changed; restart review"), 409
        return send_file(video, conditional=True)

    @app.get("/api/cues")
    def list_cues():
        if not unchanged():
            return jsonify(error="Source changed; restart review"), 409
        return jsonify(cues=[{"start": c.start_time, "end": c.end_time, "text": c.text} for c in cues])

    @app.post("/api/export")
    def export():
        if not unchanged():
            return jsonify(error="Source changed; restart review"), 409
        data = request.get_json(silent=True)
        edits = data.get("edits") if isinstance(data, dict) else None
        if not isinstance(edits, dict) or len(edits) > len(cues):
            return jsonify(error="Invalid corrections"), 400
        checked = {}
        for key, value in edits.items():
            if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                return jsonify(error="Invalid cue index"), 400
            i = int(key)
            if i >= len(cues) or not isinstance(value, str) or not value.strip() or len(value) > 5000:
                return jsonify(error="Invalid cue text"), 400
            # A single line avoids blank-line and timing-line SRT injection.
            value = " ".join(value.split())
            if "-->" in value:
                return jsonify(error="Cue text cannot contain a timing arrow"), 400
            checked[i] = value
        if not checked:
            return jsonify(error="No corrections to export"), 400
        srt = "\n\n".join(f"{i+1}\n{timestamp(c.start_time)} --> {timestamp(c.end_time)}\n{html.escape(checked.get(i, c.text))}" for i, c in enumerate(cues)) + "\n"
        if not unchanged():
            return jsonify(error="Source changed; restart review"), 409
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"corrected-{uuid.uuid4().hex}.srt"
        with target.open("x", encoding="utf-8", newline="\n") as f:
            f.write(srt)
        return jsonify(filename=target.name, corrections=len(checked)), 201

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--subtitles", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "corrected_subtitles")
    parser.add_argument("--port", type=int, default=5096)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    try:
        app = create_review_app(args.video, args.subtitles, args.output_dir)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.error(str(exc))
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
