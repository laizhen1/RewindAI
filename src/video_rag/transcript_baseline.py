"""Offline, model-free retrieval over user-provided SRT/VTT subtitles.

This is a lexical search baseline, not speech recognition, answer generation,
video understanding, or a replacement for the multimodal model pipeline.
"""

from __future__ import annotations

import hashlib
import html
import re
from dataclasses import replace
from pathlib import Path

from video_rag.adapters import EvidenceGenerator
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval import InMemoryLexicalRetriever
from video_rag.schemas import SearchHit, TimedText, VideoSegment

VIDEO_SUFFIXES = {".mp4", ".webm", ".ogv", ".mov", ".m4v"}
SUBTITLE_SUFFIXES = {".srt", ".vtt"}
MAX_SUBTITLE_BYTES = 10 * 1024 * 1024
MAX_CUES = 100_000
_TIMESTAMP = r"(?:(?P<{p}hour>\d{{2,}}):)?(?P<{p}minute>\d{{2}}):(?P<{p}second>\d{{2}})[.,](?P<{p}ms>\d{{3}})"
_TIMING_LINE = re.compile(
    r"^\s*" + _TIMESTAMP.format(p="start_") + r"\s+-->\s+"
    + _TIMESTAMP.format(p="end_") + r"(?:\s+.*)?\s*$"
)
_TAG = re.compile(r"<[^>]*>")
_QUERY_PARTS = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.IGNORECASE)
_COMMON_ENGLISH = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does",
    "explain", "find", "for", "from", "how", "in", "is", "it", "me", "of", "on", "or",
    "please", "show", "tell", "the", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "work", "works", "would", "you", "your",
}
_COMMON_CHINESE = {
    "什么", "么是", "有什", "是什", "如何", "怎么", "为何", "为什", "请问", "介绍", "讲解",
    "解释", "视频", "课程", "这个", "那个", "哪些", "可以", "一下", "一个", "我们",
}


def _content_terms(text: str) -> set[str]:
    """Extract conservative exact-match terms, excluding common question words."""
    terms: set[str] = set()
    for word in _QUERY_PARTS.findall(text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            terms.update(word[i:i + 2] for i in range(len(word) - 1) if word[i:i + 2] not in _COMMON_CHINESE)
        elif len(word) >= 2 and word not in _COMMON_ENGLISH:
            terms.add(word)
    return terms


class TranscriptLexicalRetriever:
    """Require content overlap so generic question words cannot produce results."""

    name = "transcript_lexical"

    def __init__(self) -> None:
        self.base = InMemoryLexicalRetriever()
        self.documents: dict[str, set[str]] = {}

    def build(self, segments: list[VideoSegment]) -> None:
        self.documents = {segment.segment_id: _content_terms(segment.transcript) for segment in segments}
        # Index the same normalized terms used by the overlap check. This keeps
        # mixed-language strings such as "CLIP模型" searchable for "CLIP".
        self.base.build([
            replace(segment, transcript=" ".join(sorted(self.documents[segment.segment_id])))
            for segment in segments
        ])

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        terms = _content_terms(query)
        if not terms or top_k <= 0:
            return []
        minimum_overlap = min(2, len(terms))
        ranked = self.base.search(" ".join(sorted(terms)), len(self.documents))
        accepted: list[SearchHit] = []
        for hit in ranked:
            if len(terms & self.documents[hit.segment_id]) >= minimum_overlap:
                accepted.append(SearchHit(hit.segment_id, hit.score, hit.source, len(accepted) + 1))
                if len(accepted) >= top_k:
                    break
        return accepted


class TranscriptReranker:
    def score(self, query: str, segments: list[VideoSegment]) -> list[float]:
        terms = _content_terms(query)
        if not terms:
            return [0.0 for _ in segments]
        return [len(terms & _content_terms(segment.transcript)) / len(terms) for segment in segments]


def _time(match: re.Match[str], prefix: str) -> float:
    hour = int(match.group(prefix + "hour") or 0)
    minute = int(match.group(prefix + "minute"))
    second = int(match.group(prefix + "second"))
    millisecond = int(match.group(prefix + "ms"))
    if minute >= 60 or second >= 60:
        raise ValueError("Subtitle timestamp minutes and seconds must be below 60")
    return hour * 3600 + minute * 60 + second + millisecond / 1000


def parse_subtitles(text: str) -> list[TimedText]:
    """Read timestamped SRT/WebVTT cues, failing on malformed time ranges."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[TimedText] = []
    for block in re.split(r"\n[ \t]*\n", text.strip()):
        lines = block.splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        if first.startswith(("WEBVTT", "NOTE ")) or first in {"STYLE", "REGION", "NOTE"}:
            continue
        time_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        match = _TIMING_LINE.fullmatch(lines[time_index])
        if not match:
            raise ValueError("Invalid subtitle timestamp line")
        start, end = _time(match, "start_"), _time(match, "end_")
        # Drop subtitle presentation tags and decode encoded text, keeping text
        # untrusted: the browser renders all captions via textContent.
        content = html.unescape(_TAG.sub("", " ".join(lines[time_index + 1:]))).strip()
        if end <= start:
            raise ValueError("Subtitle cue end must be after its start")
        if content:
            cues.append(TimedText(start, end, content))
            if len(cues) > MAX_CUES:
                raise ValueError("Subtitle file contains too many cues")
    if not cues:
        raise ValueError("No timestamped subtitle cues found")
    return sorted(cues, key=lambda cue: (cue.start_time, cue.end_time))


def find_subtitles(video: Path) -> Path:
    """Find a same-name sidecar without guessing when both types exist."""
    matches = [video.with_suffix(ext) for ext in (".srt", ".vtt") if video.with_suffix(ext).is_file()]
    if not matches:
        raise ValueError("No same-name .srt or .vtt file; pass --subtitles explicitly")
    if len(matches) > 1:
        raise ValueError("Both .srt and .vtt sidecars exist; choose one with --subtitles")
    return matches[0]


def group_subtitle_cues(cues: list[TimedText]) -> list[list[TimedText]]:
    """Group time-ordered cues using the workspace's subtitle segmentation."""
    groups: list[list[TimedText]] = []
    for cue in cues:
        if not groups or cue.end_time - groups[-1][0].start_time > 20 or cue.start_time - groups[-1][-1].end_time > 5:
            groups.append([cue])
        else:
            groups[-1].append(cue)
    return groups


def segments_from_subtitles(video_path: str | Path, subtitles_path: str | Path) -> list[VideoSegment]:
    video = Path(video_path).expanduser().resolve()
    subtitles = Path(subtitles_path).expanduser().resolve()
    if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Video must be an existing local MP4, WebM, OGV, MOV or M4V file")
    if subtitles.suffix.lower() == ".txt":
        raise ValueError("Plain text transcripts are not supported; use timed SRT/VTT or transcribe locally")
    if not subtitles.is_file() or subtitles.suffix.lower() not in SUBTITLE_SUFFIXES:
        raise ValueError("Subtitles must be an existing local .srt or .vtt file")
    if subtitles.stat().st_size > MAX_SUBTITLE_BYTES:
        raise ValueError("Subtitle file exceeds the 10 MiB limit")
    try:
        subtitle_text = subtitles.read_text(encoding="utf-8-sig")
        cues = parse_subtitles(subtitle_text)
    except UnicodeDecodeError as exc:
        raise ValueError("Subtitles must be encoded as UTF-8") from exc

    # Group successive cues while retaining the first and last real timestamps.
    # The video itself is not decoded or uploaded in this mode.
    groups = group_subtitle_cues(cues)
    video_id = "video_" + hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:16]
    stat = video.stat()
    revision = hashlib.sha256(
        (subtitle_text + f"\0{stat.st_size}\0{stat.st_mtime_ns}").encode("utf-8")
    ).hexdigest()[:16]
    return [
        VideoSegment(
            segment_id=f"{video_id}_{revision}_{index:05d}", video_id=video_id,
            source_path=str(video), start_time=group[0].start_time,
            end_time=max(cue.end_time for cue in group),
            transcript=" ".join(cue.text for cue in group),
        )
        for index, group in enumerate(groups)
    ]


def build_transcript_pipeline(video_path: str | Path, subtitles_path: str | Path) -> VideoRAGPipeline:
    return build_transcript_library([(video_path, subtitles_path)])


def build_transcript_library(sources: list[tuple[str | Path, str | Path]]) -> VideoRAGPipeline:
    """Index multiple distinct local videos in one in-memory learning library."""
    if not sources:
        raise ValueError("At least one video is required")
    paths = [Path(video).expanduser().resolve() for video, _ in sources]
    if len(set(paths)) != len(paths):
        raise ValueError("Each video may only be loaded once")
    segments = [segment for video, subtitles in sources
                for segment in segments_from_subtitles(video, subtitles)]
    return build_transcript_segments(segments)


def build_transcript_segments(segments: list[VideoSegment], *, generator=None) -> VideoRAGPipeline:
    """Build a new index before replacing the currently serving index."""
    pipeline = VideoRAGPipeline(
        retrievers=[TranscriptLexicalRetriever()],
        reranker=TranscriptReranker(),
        generator=generator if generator is not None else EvidenceGenerator(),
        minimum_rerank_score=0.01,
    )
    pipeline.build(segments)
    return pipeline
