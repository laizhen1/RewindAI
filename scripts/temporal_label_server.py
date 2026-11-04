"""Collect draft time-labeled questions beside a local video; never uploads media."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, jsonify, render_template, request, send_file

from scripts.diagnose_transcript import (
    comparison_by_question_type,
    diagnose,
    metrics,
    operating_metrics,
)
from scripts.prepare_temporal_questions import SPLITS, TYPES, prepare
from video_rag.evaluation.private_output import private_output
from video_rag.evaluation.provenance import fingerprint
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles

FIELDS = ("question_id", "video_id", "question_type", "answerable", "question", "reference_answer",
          "evidence_start_seconds", "evidence_end_seconds", "evidence_frame_seconds", "checked_scope",
          "annotation_status", "split", "notes")


def create_label_app(video: Path, output: Path, video_id: str, split: str, duration: float,
                     *, allowed_roots: tuple[Path, ...], subtitles: Path | None = None) -> Flask:
    video = Path(video).expanduser().resolve()
    output = private_output(output, roots=allowed_roots)
    if not video.is_file() or video.suffix.lower() not in {".mp4", ".webm", ".ogv", ".mov", ".m4v"}:
        raise ValueError("Provide an existing supported local video")
    if output.suffix.lower() != ".csv" or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", video_id) or split not in SPLITS:
        raise ValueError("Invalid output, video ID or split")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("A valid video duration is required")
    if subtitles is not None:
        subtitles = Path(subtitles).expanduser().resolve()
        if not subtitles.is_file() or subtitles.suffix.lower() not in {".srt", ".vtt"}:
            raise ValueError("Provide existing timed SRT/VTT subtitles")
    initial = video.stat()
    version = (initial.st_size, initial.st_mtime_ns)
    digest = hashlib.sha256()
    with video.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    after = video.stat()
    if (after.st_size, after.st_mtime_ns) != version:
        raise ValueError("Video changed while binding annotations")
    lock = Lock()
    output.parent.mkdir(parents=True, exist_ok=True)
    binding_path = output.with_suffix(output.suffix + ".source.json")
    binding = {"video_sha256": digest.hexdigest(), "video_id": video_id, "split": split}
    if not binding_path.exists():
        if output.exists():
            raise ValueError("Existing annotation CSV has no source binding; choose a new file")
        with binding_path.open("x", encoding="utf-8") as f:
            json.dump(binding, f)
    if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
        raise ValueError("Annotation file is bound to a different source or split")
    if not output.exists():
        try:
            with output.open("x", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        except FileExistsError:
            pass

    def source_unchanged() -> bool:
        try:
            st = video.stat()
            return video.is_file() and (st.st_size, st.st_mtime_ns) == version
        except OSError:
            return False

    def read_rows():
        raw = output.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected annotation CSV header")
        rows = list(reader)
        if any(row.get("video_id") != video_id or row.get("split") != split for row in rows):
            raise ValueError("Annotation file belongs to another video or split")
        return rows, hashlib.sha256(raw).hexdigest()

    read_rows()  # Fail before serving an incompatible existing file.
    app = Flask(__name__, template_folder=str(ROOT / "src" / "video_rag" / "api" / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024

    @app.before_request
    def local_only():
        try:
            host = urlsplit(f"http://{request.host}").hostname
            origin = request.headers.get("Origin")
            if host not in {"localhost", "127.0.0.1", "::1"}:
                return jsonify({"error": "Local access only"}), 403
            if origin:
                parsed = urlsplit(origin)
                if (parsed.scheme, parsed.netloc.lower()) != (request.scheme, request.host.lower()):
                    return jsonify({"error": "Cross-origin access is not allowed"}), 403
        except ValueError:
            return jsonify({"error": "Invalid request origin"}), 403

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("temporal_label.html")

    @app.get("/video")
    def media():
        if not source_unchanged():
            return jsonify({"error": "Video changed; restart after checking the source"}), 409
        return send_file(video, conditional=True)

    @app.get("/api/entries")
    def entries():
        try:
            rows, revision = read_rows()
        except (OSError, UnicodeError, ValueError):
            return jsonify({"error": "Unable to read the annotation file"}), 500
        return jsonify({"count": len(rows), "revision": revision, "split": split, "duration": duration,
                        "evaluation_available": subtitles is not None and split != "test"})

    @app.post("/api/evaluate")
    def evaluate_drafts():
        if subtitles is None or split == "test":
            return jsonify({"error": "Draft evaluation requires subtitles and is unavailable for the test split"}), 409
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A current annotation revision is required"}), 400
        try:
            rows, revision = read_rows()
            if payload.get("revision") != revision:
                return jsonify({"error": "Annotations changed; reload before evaluating"}), 409
            if not rows:
                return jsonify({"error": "Add at least one draft question first"}), 400
            if fingerprint(video) != binding["video_sha256"]:
                return jsonify({"error": "Video changed; check the source before evaluating"}), 409
            subtitle_hash = fingerprint(subtitles)
            if any(not isinstance(row.get("answerable"), str) or not isinstance(row.get("reference_answer"), str)
                   for row in rows):
                return jsonify({"error": "Invalid draft labels"}), 400
            if any((row["answerable"].strip().lower() == "false") == bool(row["reference_answer"].strip())
                   for row in rows):
                return jsonify({"error": "Check answers and answerability in the draft labels"}), 400
            if any(row["answerable"].strip().lower() == "false" and row.get("notes") != "scope_declared_entire_video"
                   for row in rows):
                return jsonify({"error": "Unanswerable labels require a declaration that the entire video was checked"}), 400
            questions = prepare(rows, video_id, split)["questions"]
            if any(item["expected_answerable"] and item["evidence_end_seconds"] > duration for item in questions):
                return jsonify({"error": "An evidence range exceeds the video duration"}), 400
            segments = segments_from_subtitles(video, subtitles)
            if max(segment.end_time for segment in segments) > duration + 1:
                return jsonify({"error": "Subtitle timestamps exceed the video duration"}), 400
            results = diagnose(build_transcript_segments(segments), questions)
            if (fingerprint(video) != binding["video_sha256"] or fingerprint(subtitles) != subtitle_hash
                    or read_rows()[1] != revision):
                return jsonify({"error": "An input changed during evaluation; retry after checking it"}), 409
            return jsonify({"warning": "Draft time-overlap diagnostic only; not answer accuracy or a verified test.",
                            "split": split, "metrics": metrics(results), "operating": operating_metrics(results),
                            "by_question_type": comparison_by_question_type(results),
                            "subtitle_last_end_seconds": max(segment.end_time for segment in segments),
                            "video_duration_seconds": duration})
        except (OSError, UnicodeError, KeyError, TypeError, ValueError, RuntimeError, csv.Error):
            return jsonify({"error": "Local evaluation failed; check draft labels and subtitle format"}), 400

    @app.post("/api/entries")
    def append_entry():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "A question is required"}), 400
        question, answer, scope = (payload.get(key, "") for key in ("question", "answer", "checked_scope"))
        kind, answerable = payload.get("question_type"), payload.get("answerable")
        if (not all(isinstance(value, str) and len(value) <= 4000 for value in (question, answer, scope))
                or not question.strip() or not isinstance(kind, str) or kind not in TYPES or type(answerable) is not bool):
            return jsonify({"error": "Provide a question and valid question type"}), 400
        row = dict.fromkeys(FIELDS, "")
        row.update(question_id=uuid.uuid4().hex, video_id=video_id, question_type=kind,
                   answerable=str(answerable).lower(), question=question.strip(), split=split,
                   annotation_status="draft_unverified", reference_answer=answer.strip(), checked_scope=scope.strip())
        if answerable:
            start, end = payload.get("start"), payload.get("end")
            if (not answer.strip() or type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration):
                return jsonify({"error": "Provide an answer and an evidence range within the video"}), 400
            row.update(evidence_start_seconds=str(start), evidence_end_seconds=str(end))
        elif not scope.strip() or answer.strip() or payload.get("checked_entire_video") is not True:
            return jsonify({"error": "Check the entire video, describe the scope, and leave the answer empty"}), 400
        else:
            row["notes"] = "scope_declared_entire_video"
        try:
            prepare([row], video_id, split)
            with lock:
                if not source_unchanged():
                    return jsonify({"error": "Video changed; restart after checking the source"}), 409
                rows, revision = read_rows()
                if payload.get("revision") != revision:
                    return jsonify({"error": "Annotations changed; reload the page before saving"}), 409
                rows.append(row)
                temp_name = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output.parent,
                                                     prefix=".annotations-", suffix=".tmp", delete=False) as f:
                        temp_name = f.name
                        writer = csv.DictWriter(f, fieldnames=FIELDS)
                        writer.writeheader()
                        writer.writerows(rows)
                    os.replace(temp_name, output)
                finally:
                    if temp_name and Path(temp_name).exists():
                        Path(temp_name).unlink()
                _, new_revision = read_rows()
        except (OSError, UnicodeError, ValueError, csv.Error):
            return jsonify({"error": "Could not save the local annotation"}), 500
        return jsonify({"count": len(rows), "revision": new_revision}), 201

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subtitles", type=Path, help="Optional local timed subtitles for draft evaluation")
    parser.add_argument("--port", type=int, default=5095)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    try:
        import av
    except ImportError:
        parser.exit(2, "Cannot start local labeler; install PyAV.\n")
    try:
        with av.open(str(args.video)) as container:
            duration = container.duration / av.time_base if container.duration else 0
        app = create_label_app(args.video, args.output, args.video_id, args.split, duration,
                               allowed_roots=(ROOT / "artifacts", ROOT / "local_data"), subtitles=args.subtitles)
    except (OSError, ValueError, av.error.FFmpegError) as exc:
        parser.exit(2, f"Cannot start local labeler ({type(exc).__name__}); check video and PyAV.\n")
    print(f"Local-only labeling: http://127.0.0.1:{args.port}/", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
