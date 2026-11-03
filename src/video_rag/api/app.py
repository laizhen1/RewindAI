from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request, send_file

from video_rag.api.library import LibraryStore
from video_rag.pipeline import VideoRAGPipeline


def create_app(
    pipeline: VideoRAGPipeline,
    *,
    database_path: str | Path = "artifacts/rewindai/library.sqlite3",
    mode: str = "unknown",
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    store = LibraryStore(database_path)
    if mode not in {"demo", "full", "transcript", "unknown"}:
        raise ValueError("mode must be demo, full, transcript or unknown")

    @app.before_request
    def protect_local_workspace():
        # This prototype stores private notes and offers local video access. Only
        # serve loopback hostnames to mitigate DNS rebinding, and reject browser
        # requests from another origin. Remote deployment requires authentication.
        try:
            host = urlsplit(f"http://{request.host}").hostname
        except ValueError:
            host = None
        if host not in {"localhost", "127.0.0.1", "::1"}:
            return jsonify({"error": "local access only"}), 403
        origin = request.headers.get("Origin")
        if origin:
            try:
                parsed = urlsplit(origin)
                if (parsed.scheme, parsed.netloc.lower()) != (request.scheme, request.host.lower()):
                    return jsonify({"error": "cross-origin access is not allowed"}), 403
            except ValueError:
                return jsonify({"error": "invalid request origin"}), 403

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "mode": mode})

    @app.get("/api/library")
    def library():
        assignments = store.assignments()
        videos = pipeline.videos()
        for item in videos:
            item["course_id"] = assignments.get(item["id"])
            item["video_url"] = f"/api/videos/{item['id']}"
        return jsonify({"courses": store.courses(), "videos": videos})

    @app.post("/api/courses")
    def create_course():
        payload = request.get_json(silent=True)
        name = payload.get("name") if isinstance(payload, dict) else None
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            return jsonify({"error": "name must be between 1 and 120 characters"}), 400
        return jsonify(store.create_course(name.strip())), 201

    @app.put("/api/videos/<video_id>/course")
    def assign_course(video_id: str):
        if pipeline.video_path(video_id) is None:
            return jsonify({"error": "video not found in the loaded index"}), 404
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "course_id" not in payload:
            return jsonify({"error": "course_id is required"}), 400
        course_id = payload["course_id"]
        if course_id is not None and (type(course_id) is not int or course_id <= 0):
            return jsonify({"error": "course_id must be a positive integer or null"}), 400
        if not store.assign(video_id, course_id):
            return jsonify({"error": "course not found"}), 404
        return jsonify({"video_id": video_id, "course_id": course_id})

    @app.get("/api/bookmarks")
    def bookmarks():
        return jsonify({"bookmarks": store.bookmarks()})

    @app.post("/api/bookmarks")
    def add_bookmark():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400
        video_id, segment_id = payload.get("video_id"), payload.get("segment_id")
        note = payload.get("note", "")
        if not isinstance(video_id, str) or not isinstance(segment_id, str):
            return jsonify({"error": "video_id and segment_id are required"}), 400
        if not isinstance(note, str) or len(note) > 4000:
            return jsonify({"error": "note must be at most 4000 characters"}), 400
        segment = pipeline.segment(segment_id)
        if segment is None or segment.video_id != video_id:
            return jsonify({"error": "segment not found in the loaded video"}), 404
        # Persist canonical evidence times from the index, not untrusted client values.
        return jsonify(store.create_bookmark(
            video_id, segment_id, segment.start_time, segment.end_time, note
        )), 201

    @app.patch("/api/bookmarks/<int:bookmark_id>")
    def edit_bookmark(bookmark_id: int):
        payload = request.get_json(silent=True)
        note = payload.get("note") if isinstance(payload, dict) else None
        if not isinstance(note, str) or len(note) > 4000:
            return jsonify({"error": "note must be at most 4000 characters"}), 400
        item = store.update_note(bookmark_id, note)
        if item is None:
            return jsonify({"error": "bookmark not found"}), 404
        return jsonify(item)

    @app.delete("/api/bookmarks/<int:bookmark_id>")
    def delete_bookmark(bookmark_id: int):
        if not store.delete_bookmark(bookmark_id):
            return jsonify({"error": "bookmark not found"}), 404
        return "", 204

    @app.post("/api/feedback")
    def feedback():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400
        question, segment_id = payload.get("question"), payload.get("segment_id")
        rating, comment = payload.get("rating"), payload.get("comment", "")
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            return jsonify({"error": "question must contain 1 to 4000 characters"}), 400
        if rating not in ("helpful", "not_helpful"):
            return jsonify({"error": "invalid rating"}), 400
        if not isinstance(comment, str) or len(comment) > 4000:
            return jsonify({"error": "comment must be at most 4000 characters"}), 400
        if segment_id is not None and (not isinstance(segment_id, str) or pipeline.segment(segment_id) is None):
            return jsonify({"error": "segment not found"}), 404
        feedback_id = store.add_feedback(question.strip(), segment_id, rating, comment)
        return jsonify({"id": feedback_id}), 201

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            return jsonify({"error": "question must contain 1 to 4000 characters"}), 400
        video_id = payload.get("video_id")
        if video_id is not None and (not isinstance(video_id, str) or not video_id):
            return jsonify({"error": "video_id must be a non-empty string"}), 400
        if video_id is not None and pipeline.video_path(video_id) is None:
            return jsonify({"error": "video not found in the loaded index"}), 404
        try:
            result = pipeline.ask(question, video_id=video_id) if video_id is not None else pipeline.ask(question)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        response = result.to_dict()
        for evidence in response["evidence"]:
            # Local source paths may contain personal directory names. Video access
            # goes through the indexed ID rather than exposing filesystem paths.
            evidence.pop("source_path", None)
            for keyframe in evidence.get("keyframes", []):
                keyframe.pop("path", None)
            evidence["video_url"] = f"/api/videos/{evidence['video_id']}"
        return jsonify(response)

    @app.get("/api/videos/<video_id>")
    def video(video_id: str):
        source = pipeline.video_path(video_id)
        if source is None:
            return jsonify({"error": "video not found"}), 404
        resolved = Path(source).resolve()
        if not resolved.is_file():
            return jsonify({"error": "video file is unavailable"}), 404
        return send_file(resolved, conditional=True)

    return app

