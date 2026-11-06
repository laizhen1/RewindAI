"""Review and correct timed subtitles beside a local video without uploading them."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, jsonify, render_template, request, send_file

from video_rag.evaluation.private_output import PRIVATE_ROOTS, private_output
from video_rag.evaluation.subtitle_metrics import reviewed_token_error
from video_rag.transcript_baseline import MAX_SUBTITLE_BYTES, parse_subtitles


def timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


TOKEN = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)
MAX_ALIGNMENT_TOKENS = 40_000


def align_alternative(cues, alternative: str) -> list[dict]:
    """Heuristically align untimed words to cues; neither transcript is ground truth."""
    words, owners = [], []
    for i, cue in enumerate(cues):
        tokens = TOKEN.findall(cue.text)
        words.extend(token.lower() for token in tokens)
        owners.extend([i] * len(tokens))
    other = TOKEN.findall(alternative)
    if not words or not other or max(len(words), len(other)) > MAX_ALIGNMENT_TOKENS:
        raise ValueError("Alternative or subtitles have no usable words or exceed the alignment limit")
    blocks = [[] for _ in cues]
    differences = [False] * len(cues)
    matcher = SequenceMatcher(None, words, [word.lower() for word in other], autojunk=True)
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag == 'equal':
            for offset in range(b - a):
                blocks[owners[a + offset]].append(other[c + offset])
        else:
            for i in range(a, b):
                differences[owners[i]] = True
            for j in range(c, d):
                position = a + (j - c) * max(1, b - a) // max(1, d - c)
                owner = owners[min(position, len(owners) - 1)]
                differences[owner] = True
                blocks[owner].append(other[j])
    aligned = []
    for block, difference in zip(blocks, differences, strict=True):
        text = " ".join(block)
        aligned.append({"alternative": text[:2000], "alternative_truncated": len(text) > 2000,
                        "differs": difference})
    return aligned


def create_review_app(video: Path, subtitles: Path, output_dir: Path, *, alternative: Path | None = None, allowed_roots=PRIVATE_ROOTS) -> Flask:
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
    alternative_digest = None
    aligned = None
    if alternative is not None:
        alternative = Path(alternative).expanduser().resolve()
        if not alternative.is_file() or alternative.suffix.lower() != ".txt" or alternative.stat().st_size > MAX_SUBTITLE_BYTES:
            raise ValueError("Provide an existing UTF-8 text transcript under 10 MiB")
        alternative_bytes = alternative.read_bytes()
        alternative_digest = hashlib.sha256(alternative_bytes).hexdigest()
        aligned = align_alternative(cues, alternative_bytes.decode("utf-8-sig"))
    version = (video.stat().st_size, video.stat().st_mtime_ns)
    source_digest = hashlib.sha256(source).hexdigest()
    app = Flask(__name__, template_folder=str(ROOT / "src" / "video_rag" / "api" / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = MAX_SUBTITLE_BYTES

    def unchanged() -> bool:
        try:
            current = video.stat()
            return ((current.st_size, current.st_mtime_ns) == version and
                    hashlib.sha256(subtitles.read_bytes()).hexdigest() == source_digest and
                    (alternative is None or hashlib.sha256(alternative.read_bytes()).hexdigest() == alternative_digest))
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
        return jsonify(cues=[{"start": c.start_time, "end": c.end_time, "text": c.text,
                              **(aligned[i] if aligned is not None else {})} for i, c in enumerate(cues)])

    def parse_edits(edits):
        if not isinstance(edits, dict) or len(edits) > len(cues):
            raise ValueError("Invalid corrections")
        checked = {}
        for key, value in edits.items():
            if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                raise ValueError("Invalid cue index")
            i = int(key)
            if i >= len(cues) or not isinstance(value, str) or not value.strip() or len(value) > 5000:
                raise ValueError("Invalid cue text")
            value = " ".join(value.split())
            if "-->" in value:
                raise ValueError("Cue text cannot contain a timing arrow")
            checked[i] = value
        return checked

    @app.post("/api/metrics")
    def metrics():
        if not unchanged():
            return jsonify(error="Source changed; restart review"), 409
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or data.get("confirm_reviewed") is not True:
            return jsonify(error="Confirm the selected cues were reviewed against the recording"), 400
        try:
            checked = parse_edits(data.get("edits"))
            result = reviewed_token_error(cues, checked, data.get("reviewed"))
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        if not unchanged():
            return jsonify(error="Source changed; restart review"), 409
        save_report = data.get("save_report", False)
        if type(save_report) is not bool:
            return jsonify(error="Save option must be true or false"), 400
        warning = ("User-declared reviewed cues only; not independently verified. "
                   "Punctuation and case ignored; English words and CJK characters are tokens. "
                   "Scores are not representative of unchecked subtitles.")
        response = {"metrics": result, "warning": warning}
        if save_report:
            selected = sorted(data["reviewed"])
            references = [[i, checked.get(i, cues[i].text)] for i in selected]
            reference_bytes = json.dumps(references, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            report = {"status": "user_declared_review_not_independently_verified", "metrics": result,
                      "warning": warning, "source_sha256": source_digest,
                      "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                      "reference_text_included": False, "reviewed_cue_indices": selected,
                      "reproducibility": "Export corrected subtitles separately to preserve edited reference text."}
            if not unchanged():
                return jsonify(error="Source changed; restart review"), 409
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / f"subtitle-metrics-{uuid.uuid4().hex}.json"
            with target.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            response["filename"] = target.name
        return jsonify(response)

    @app.post("/api/export")
    def export():
        if not unchanged():
            return jsonify(error="Source changed; restart review"), 409
        data = request.get_json(silent=True)
        try:
            checked = parse_edits(data.get("edits") if isinstance(data, dict) else None)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
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
    parser.add_argument("--alternative", type=Path, help="Optional untimed UTF-8 TXT transcript for approximate comparison")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "corrected_subtitles")
    parser.add_argument("--port", type=int, default=5096)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    try:
        app = create_review_app(args.video, args.subtitles, args.output_dir, alternative=args.alternative)
    except (OSError, ValueError, UnicodeError) as exc:
        parser.error(str(exc))
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
