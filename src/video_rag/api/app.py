from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request, send_file

from video_rag.api.library import LibraryStore
from video_rag.pipeline import VideoRAGPipeline
from video_rag.schemas import VideoSegment


def create_app(
    pipeline: VideoRAGPipeline,
    *,
    database_path: str | Path = "artifacts/rewindai/library.sqlite3",
    mode: str = "unknown",
    transcript_builder: Callable[[list[VideoSegment]], VideoRAGPipeline] | None = None,
    retrieval_type: str = "keyword",
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    store = LibraryStore(database_path)
    import_lock = Lock()
    if mode not in {"demo", "full", "transcript", "unknown"}:
        raise ValueError("mode must be demo, full, transcript or unknown")
    unavailable_videos: dict[str, str] = {}
    if mode == "transcript":
        from video_rag.transcript_baseline import build_transcript_segments

        transcript_builder = transcript_builder or build_transcript_segments

        restored = list(pipeline.segments())
        loaded_ids = {segment.video_id for segment in restored}
        for record in store.local_videos():
            if record["video_id"] in loaded_ids:
                continue
            try:
                stat = Path(record["video_path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (record["video_size"], record["video_mtime_ns"]):
                    raise ValueError("Stored video changed")
                entries = json.loads(record["segments_json"])
                segments = [VideoSegment(**entry) for entry in entries]
                if not segments or any(segment.video_id != record["video_id"] or
                                       segment.source_path != record["video_path"] for segment in segments):
                    raise ValueError("Stored segments are inconsistent")
                restored.extend(segments)
                loaded_ids.add(record["video_id"])
            except (OSError, ValueError, TypeError, KeyError):
                unavailable_videos[record["video_id"]] = Path(record["video_path"]).name
        if len(restored) > len(pipeline.segments()):
            pipeline = transcript_builder(restored)

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
        return jsonify({"status": "ok", "mode": mode, "retrieval_type": retrieval_type})

    @app.get("/api/library")
    def library():
        assignments = store.assignments()
        videos = pipeline.videos()
        for item in videos:
            item["course_id"] = assignments.get(item["id"])
            item["video_url"] = f"/api/videos/{item['id']}"
        return jsonify({"courses": store.courses(), "videos": videos,
                        "unavailable_saved_videos": len(unavailable_videos),
                        "unavailable_videos": [{"id": key, "title": value} for key, value in unavailable_videos.items()]})

    @app.delete("/api/library/unavailable-video/<video_id>")
    def forget_unavailable_video(video_id: str):
        if mode != "transcript":
            return jsonify({"error": "Only available in subtitle mode"}), 409
        with import_lock:
            if video_id not in unavailable_videos:
                return jsonify({"error": "Unavailable saved video not found"}), 404
            if not store.forget_video(video_id):
                return jsonify({"error": "Saved video not found"}), 404
            del unavailable_videos[video_id]
        return "", 204

    @app.delete("/api/library/videos/<video_id>")
    def forget_loaded_video(video_id: str):
        nonlocal pipeline
        if mode != "transcript":
            return jsonify({"error": "Only available in subtitle mode"}), 409
        with import_lock:
            current = pipeline.segments()
            if not any(segment.video_id == video_id for segment in current):
                return jsonify({"error": "Video not found in the loaded index"}), 404
            remaining = [segment for segment in current if segment.video_id != video_id]
            try:
                replacement = transcript_builder(remaining)
                # Sources passed at startup may not have a saved record.
                store.forget_video(video_id)
            except (sqlite3.Error, OSError, RuntimeError, ValueError):
                return jsonify({"error": "Could not rebuild or save the local index"}), 500
            pipeline = replacement
        return "", 204

    @app.post("/api/library/local-video")
    def import_local_video():
        nonlocal pipeline
        if mode != "transcript":
            return jsonify({"error": "Local import is only available in subtitle mode"}), 409
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400
        video = payload.get("video_path")
        subtitle = payload.get("subtitle_path", "")
        if not isinstance(video, str) or not video.strip() or not isinstance(subtitle, str):
            return jsonify({"error": "A local video path and optional subtitle path are required"}), 400
        if len(video) > 4096 or len(subtitle) > 4096:
            return jsonify({"error": "Path is too long"}), 400
        from video_rag.transcript_baseline import (
            find_subtitles,
            segments_from_subtitles,
        )
        try:
            video_path = Path(video.strip()).expanduser().resolve()
            subtitle_path = Path(subtitle.strip()).expanduser().resolve() if subtitle.strip() else find_subtitles(video_path)
            original_stat = video_path.stat()
            added = segments_from_subtitles(video_path, subtitle_path)
            current_stat = video_path.stat()
            if (original_stat.st_size, original_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns):
                return jsonify({"error": "Video changed while importing; try again"}), 409
            with import_lock:
                current = pipeline.segments()
                if any(segment.video_id == added[0].video_id for segment in current):
                    return jsonify({"error": "Video is already loaded"}), 409
                replacement = transcript_builder([*current, *added])
                store.remember_video(added[0].video_id, str(video_path), current_stat.st_size,
                                     current_stat.st_mtime_ns, json.dumps([segment.to_dict() for segment in added]))
                pipeline = replacement
            return jsonify({"video_id": added[0].video_id, "segment_count": len(added)}), 201
        except sqlite3.IntegrityError:
            return jsonify({"error": "Video is already saved; restore the original file and restart"}), 409
        except (OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

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
        items = store.bookmarks()
        for item in items:
            segment = pipeline.segment(item["segment_id"])
            item["available"] = bool(segment and Path(segment.source_path).is_file() and
                                     segment.video_id == item["video_id"] and
                                     segment.start_time == item["start_time"] and segment.end_time == item["end_time"])
        return jsonify({"bookmarks": items})

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

    @app.get("/api/feedback")
    def list_feedback():
        return jsonify({"feedback": store.recent_feedback()})

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
        course_id = payload.get("course_id")
        if course_id is not None and (type(course_id) is not int or course_id <= 0):
            return jsonify({"error": "course_id must be a positive integer"}), 400
        if video_id is not None and course_id is not None:
            return jsonify({"error": "Choose a video or a course, not both"}), 400
        if video_id is not None and (not isinstance(video_id, str) or not video_id):
            return jsonify({"error": "video_id must be a non-empty string"}), 400
        if video_id is not None and pipeline.video_path(video_id) is None:
            return jsonify({"error": "video not found in the loaded index"}), 404
        course_videos = None
        if course_id is not None:
            if not any(course["id"] == course_id for course in store.courses()):
                return jsonify({"error": "course not found"}), 404
            course_videos = {key for key, value in store.assignments().items() if value == course_id
                             and pipeline.video_path(key) is not None}
            if not course_videos:
                return jsonify({"error": "No loaded videos in this course"}), 400
        try:
            if course_videos is not None:
                result = pipeline.ask(question, video_ids=course_videos)
            elif video_id is not None:
                result = pipeline.ask(question, video_id=video_id)
            else:
                result = pipeline.ask(question)
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
