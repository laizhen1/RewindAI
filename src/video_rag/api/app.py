from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, render_template, request, send_file

from video_rag.api.library import LibraryStore
from video_rag.api.visual_sampling import sample_visual_segments
from video_rag.pipeline import VideoRAGPipeline
from video_rag.schemas import VideoSegment


def create_app(
    pipeline: VideoRAGPipeline,
    *,
    database_path: str | Path = "artifacts/rewindai/library.sqlite3",
    mode: str = "unknown",
    transcript_builder: Callable[[list[VideoSegment]], VideoRAGPipeline] | None = None,
    retrieval_type: str = "keyword",
    visual_builder: Callable[[list[VideoSegment]], object] | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    store = LibraryStore(database_path)
    import_lock = RLock()
    if mode not in {"demo", "full", "transcript", "unknown"}:
        raise ValueError("mode must be demo, full, transcript or unknown")
    unavailable_videos: dict[str, str] = {}
    visual_index = None
    visual_key = None
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
        return jsonify({"status": "ok", "mode": mode, "retrieval_type": retrieval_type,
                        "visual_enabled": visual_builder is not None})

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
        nonlocal pipeline, visual_index, visual_key
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
            visual_index = None
            visual_key = None
        return "", 204

    @app.post("/api/library/local-video")
    def import_local_video():
        nonlocal pipeline, visual_index, visual_key
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
            VIDEO_SUFFIXES,
            find_subtitles,
            segments_from_subtitles,
        )
        try:
            video_path = Path(video.strip()).expanduser().resolve()
            if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError("Video must be an existing local MP4, WebM, OGV, MOV or M4V file")
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
                visual_index = None
                visual_key = None
            return jsonify({"video_id": added[0].video_id, "segment_count": len(added)}), 201
        except sqlite3.IntegrityError:
            return jsonify({"error": "Video is already saved; restore the original file and restart"}), 409
        except (OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        except (sqlite3.Error, RuntimeError):
            return jsonify({"error": "Could not rebuild or save the local index; existing videos remain available"}), 500

    @app.post("/api/courses")
    def create_course():
        payload = request.get_json(silent=True)
        name = payload.get("name") if isinstance(payload, dict) else None
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            return jsonify({"error": "name must be between 1 and 120 characters"}), 400
        return jsonify(store.create_course(name.strip())), 201

    @app.patch("/api/courses/<int:course_id>")
    def rename_course(course_id: int):
        payload = request.get_json(silent=True)
        name = payload.get("name") if isinstance(payload, dict) else None
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            return jsonify({"error": "name must be between 1 and 120 characters"}), 400
        if not store.rename_course(course_id, name.strip()):
            return jsonify({"error": "course not found"}), 404
        return jsonify({"id": course_id, "name": name.strip()})

    @app.delete("/api/courses/<int:course_id>")
    def delete_course(course_id: int):
        with import_lock:
            if not store.delete_course(course_id):
                return jsonify({"error": "course not found"}), 404
        return "", 204

    @app.put("/api/videos/<video_id>/course")
    def assign_course(video_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "course_id" not in payload:
            return jsonify({"error": "course_id is required"}), 400
        course_id = payload["course_id"]
        if course_id is not None and (type(course_id) is not int or course_id <= 0):
            return jsonify({"error": "course_id must be a positive integer or null"}), 400
        with import_lock:
            if pipeline.video_path(video_id) is None:
                return jsonify({"error": "video not found in the loaded index"}), 404
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

    @app.get("/api/feedback/summary")
    def feedback_summary():
        groups = store.feedback_summary()
        return jsonify({"groups": groups, "total": sum(group["count"] for group in groups)})

    @app.delete("/api/feedback/<int:feedback_id>")
    def delete_feedback(feedback_id: int):
        if not store.delete_feedback(feedback_id):
            return jsonify({"error": "feedback not found"}), 404
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
        target = payload.get("target_type", "answer" if segment_id is None else "unknown_evidence")
        if target not in ("answer", "transcript", "visual", "unknown_evidence"):
            return jsonify({"error": "invalid feedback target"}), 400
        if (target == "answer") != (segment_id is None):
            return jsonify({"error": "feedback target does not match the segment"}), 400
        feedback_id = store.add_feedback(question.strip(), segment_id, rating, comment, target)
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

    @app.post("/api/visual-search")
    def visual_search():
        nonlocal visual_index, visual_key
        if mode != "transcript" or visual_builder is None:
            return jsonify({"error": "Experimental visual search is not enabled"}), 409
        payload = request.get_json(silent=True)
        question = payload.get("question") if isinstance(payload, dict) else None
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            return jsonify({"error": "question must contain 1 to 1000 characters"}), 400
        video_id, course_id = payload.get("video_id"), payload.get("course_id")
        if video_id is not None and (not isinstance(video_id, str) or not video_id):
            return jsonify({"error": "video_id must be a non-empty string"}), 400
        if course_id is not None and (type(course_id) is not int or course_id <= 0):
            return jsonify({"error": "course_id must be a positive integer"}), 400
        if video_id is not None and course_id is not None:
            return jsonify({"error": "Choose a video or a course, not both"}), 400
        try:
            with import_lock:
                segments = pipeline.segments()
                if video_id is not None:
                    if not any(segment.video_id == video_id for segment in segments):
                        return jsonify({"error": "video not found in the loaded index"}), 404
                    segments = tuple(segment for segment in segments if segment.video_id == video_id)
                if course_id is not None:
                    if not any(course["id"] == course_id for course in store.courses()):
                        return jsonify({"error": "course not found"}), 404
                    assignments = store.assignments()
                    segments = tuple(segment for segment in segments if assignments.get(segment.video_id) == course_id)
                if not segments:
                    return jsonify({"error": "No loaded videos in the selected scope"}), 400
                total = len(segments)
                try:
                    selected = sample_visual_segments(segments, limit=100)
                except ValueError as exc:
                    return jsonify({"error": str(exc)}), 400
                scope_key = tuple(sorted(segment.segment_id for segment in selected))
                if visual_index is None or visual_key != scope_key:
                    candidate = visual_builder(selected)
                    visual_index = candidate
                    visual_key = scope_key
                hits = visual_index.search(question.strip(), 3)
                lookup = {segment.segment_id: segment for segment in selected}
                evidence = [{"segment_id": hit.segment_id, "video_id": lookup[hit.segment_id].video_id,
                             "start_time": lookup[hit.segment_id].start_time,
                             "end_time": lookup[hit.segment_id].end_time, "score": hit.score}
                            for hit in hits if hit.segment_id in lookup]
        except (OSError, RuntimeError, ValueError, ImportError):
            return jsonify({"error": "Visual indexing or search failed; check local model and videos"}), 503
        return jsonify({"evidence": evidence, "source_segments": total, "indexed_segments": len(selected),
                        "sampled": len(selected) < total,
                        "warning": "Visual similarity is not proof of relevance; results can be wrong "
                                   "even when nothing matches. Unindexed segments are not searched."})

    @app.post("/api/compare-search")
    def compare_search():
        """Show transcript and visual candidates separately, without claiming an answer."""
        if mode != "transcript" or visual_builder is None:
            return jsonify({"error": "Experimental visual search is not enabled"}), 409
        payload = request.get_json(silent=True)
        question = payload.get("question") if isinstance(payload, dict) else None
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            return jsonify({"error": "question must contain 1 to 1000 characters"}), 400
        # The same lock prevents library or course assignments from changing
        # between the two searches. The visual handler re-enters this RLock.
        with import_lock:
            text_response = app.make_response(ask())
            if text_response.status_code != 200:
                return text_response
            visual_response = app.make_response(visual_search())
        if visual_response.status_code != 200:
            return jsonify({"transcript": text_response.get_json(), "visual": None,
                            "visual_error": (visual_response.get_json(silent=True) or {}).get("error", "Visual search failed")})
        return jsonify({"transcript": text_response.get_json(), "visual": visual_response.get_json(),
                        "visual_error": None})

    @app.get("/api/videos/<video_id>")
    def video(video_id: str):
        source = pipeline.video_path(video_id)
        if source is None:
            return jsonify({"error": "video not found"}), 404
        resolved = Path(source).resolve()
        if not resolved.is_file():
            return jsonify({"error": "video file is unavailable"}), 404
        return send_file(resolved, conditional=True)

    @app.get("/api/segments/<segment_id>/context")
    def segment_context(segment_id: str):
        if mode != "transcript":
            return jsonify({"error": "Context is only available in subtitle mode"}), 409
        snapshot = pipeline.segments()
        target = next((item for item in snapshot if item.segment_id == segment_id), None)
        if target is None:
            return jsonify({"error": "Segment not found"}), 404
        ordered = sorted((item for item in snapshot if item.video_id == target.video_id),
                         key=lambda item: (item.start_time, item.end_time, item.segment_id))
        index = next(i for i, item in enumerate(ordered) if item.segment_id == segment_id)
        context = []
        for pos, relation in ((index - 1, "previous"), (index + 1, "next")):
            if 0 <= pos < len(ordered):
                item = ordered[pos]
                context.append({"relation": relation, "segment_id": item.segment_id,
                                "video_id": item.video_id, "start_time": item.start_time,
                                "end_time": item.end_time, "transcript": item.transcript})
        return jsonify({"context": context})

    @app.get("/api/segments/<segment_id>/frame")
    def segment_frame(segment_id: str):
        if mode != "transcript":
            return jsonify({"error": "Frame preview is only available in subtitle mode"}), 409
        segment = pipeline.segment(segment_id)
        if segment is None:
            return jsonify({"error": "Segment not found"}), 404
        if not Path(segment.source_path).is_file():
            return jsonify({"error": "Video file is unavailable"}), 404
        try:
            import av

            from video_rag.api.frames import extract_frame_png

            image, actual = extract_frame_png(segment.source_path, (segment.start_time + segment.end_time) / 2)
        except ImportError:
            return jsonify({"error": "Frame preview requires the optional PyAV package"}), 503
        except (av.error.FFmpegError, OSError, OverflowError, RuntimeError, ValueError):
            # Do not expose local file paths in decoding errors.
            return jsonify({"error": "Could not decode a frame near this segment"}), 422
        return Response(image, mimetype="image/png", headers={"Cache-Control": "no-store",
                                                               "X-Frame-Seconds": f"{actual:.3f}"})

    return app
