"""Review locally prepared visual labels in a loopback-only browser interface."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_clip_frames import validate_sources


def create_review_app(directory: Path, video: Path, subtitles: Path, *, allowed_roots: tuple[Path, ...]) -> Flask:
    directory = directory.resolve()
    if not any(directory != root.resolve() and directory.is_relative_to(root.resolve()) for root in allowed_roots):
        raise ValueError("Review directory must be inside artifacts/ or local_data/")
    labels_path = directory / "labels.json"
    initial = json.loads(labels_path.read_text(encoding="utf-8-sig"))
    validate_sources(initial, video, subtitles)
    times = initial.get("segment_midpoints_seconds")
    if (not isinstance(times, list) or not 1 <= len(times) <= 100 or
            not isinstance(initial.get("decoded_frame_seconds"), list) or
            len(initial["decoded_frame_seconds"]) != len(times)):
        raise ValueError("Invalid frame manifest")
    if any(not (directory / f"frame-{index:03}.png").is_file() for index in range(len(times))):
        raise ValueError("A preview frame is missing")
    template_dir = ROOT / "src" / "video_rag" / "api" / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
    lock = Lock()

    def read_labels():
        raw = labels_path.read_bytes()
        return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()

    @app.before_request
    def local_only():
        try:
            host = urlsplit(f"http://{request.host}").hostname
            origin = request.headers.get("Origin")
            if host not in {"localhost", "127.0.0.1", "::1"}:
                return jsonify({"error": "local access only"}), 403
            if origin:
                parsed = urlsplit(origin)
                if (parsed.scheme, parsed.netloc.lower()) != (request.scheme, request.host.lower()):
                    return jsonify({"error": "cross-origin access is not allowed"}), 403
        except ValueError:
            return jsonify({"error": "invalid request origin"}), 403

    @app.get("/")
    def index():
        return render_template("visual_review.html")

    @app.get("/api/review")
    def review():
        manifest, revision = read_labels()
        return jsonify({"frame_times": manifest["decoded_frame_seconds"], "queries": manifest["queries"],
                        "review_status": manifest["review_status"], "revision": revision})

    @app.get("/api/frames/<int:index>")
    def frame(index: int):
        if index >= len(times):
            return jsonify({"error": "frame not found"}), 404
        response = send_file(directory / f"frame-{index:03}.png", mimetype="image/png")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.put("/api/review")
    def save():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
            return jsonify({"error": "Invalid review"}), 400
        queries = payload["queries"]
        if len(queries) > 200:
            return jsonify({"error": "At most 200 queries are supported"}), 400
        cleaned = []
        for item in queries:
            if not isinstance(item, dict):
                return jsonify({"error": "Invalid query"}), 400
            query, indices = item.get("query"), item.get("relevant_segment_indices")
            if (not isinstance(query, str) or not query.strip() or len(query) > 1000 or
                    not isinstance(indices, list) or
                    any(type(index) is not int or not 0 <= index < len(times) for index in indices) or
                    len(indices) != len(set(indices))):
                return jsonify({"error": "Invalid query or frame selection"}), 400
            cleaned.append({"query": query.strip(), "relevant_segment_indices": sorted(indices)})
        reviewed = payload.get("reviewed", False)
        if type(reviewed) is not bool or (reviewed and not cleaned):
            return jsonify({"error": "At least one query is required before review"}), 400
        with lock:
            manifest, revision = read_labels()
            if payload.get("revision") != revision:
                return jsonify({"error": "Labels changed in another session; reload before saving"}), 409
            manifest["queries"] = cleaned
            manifest["review_status"] = "reviewed" if reviewed else "draft"
            data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=".review-", suffix=".tmp", delete=False) as f:
                    temp_name = f.name
                    f.write(data)
                os.replace(temp_name, labels_path)
            finally:
                if temp_name and Path(temp_name).exists():
                    Path(temp_name).unlink()
        return jsonify({"revision": hashlib.sha256(data).hexdigest(), "review_status": manifest["review_status"]})

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5094)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        app = create_review_app(args.review_dir, args.video, args.subtitles,
                                allowed_roots=(ROOT / "artifacts", ROOT / "local_data"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Could not open review: {exc}\n")
    print(f"Local-only review: http://127.0.0.1:{args.port}/", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
