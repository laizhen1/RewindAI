"""Construct optional visual-only segments for gaps between subtitle segments."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from video_rag.schemas import VideoSegment


def video_duration_seconds(path: str | Path) -> float:
    """Prefer video-stream duration over a container that may include audio-only tail."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError('The local file has no video stream')
            stream = container.streams.video[0]
            stream_duration = (float(stream.duration * stream.time_base)
                               if stream.duration is not None and stream.time_base is not None else 0)
            duration = (stream_duration if math.isfinite(stream_duration) and stream_duration > 0 else
                        container.duration / av.time_base if container.duration else 0)
    except (av.error.FFmpegError, OSError) as exc:
        raise ValueError('Could not read the local video duration') from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('The local video has no finite positive duration')
    return float(duration)


def cover_subtitle_gaps(segments: list[VideoSegment], duration: float, *, window_seconds: float = 20.0,
                        max_gap_segments: int = 10_000) -> list[VideoSegment]:
    """Keep subtitle segments unchanged and cover their uncovered video intervals.

    The added segments contain no transcript and must not be represented as
    textual evidence. Full time coverage does not guarantee that midpoint frame
    sampling sees an event within a segment.
    """
    if (not segments or not math.isfinite(duration) or duration <= 0
            or not math.isfinite(window_seconds) or window_seconds <= 0
            or type(max_gap_segments) is not int or max_gap_segments <= 0):
        raise ValueError('Provide segments, a finite positive duration and window, and a positive limit')
    duration, window_seconds = float(duration), float(window_seconds)
    video_id, source_path = segments[0].video_id, segments[0].source_path
    if (any(s.video_id != video_id or s.source_path != source_path for s in segments)
            or len({s.segment_id for s in segments}) != len(segments)
            or any(not math.isfinite(s.start_time) or not math.isfinite(s.end_time)
                   or s.start_time < 0 or s.start_time >= duration or s.end_time <= s.start_time
                   or s.end_time > duration + 1 for s in segments)):
        raise ValueError('Subtitle segments must be unique, finite and from the same video')
    ordered = sorted(segments, key=lambda s: (s.start_time, s.end_time, s.segment_id))
    # Existing subtitle IDs include the source revision. Binding gap IDs to them
    # prevents a restored visual bookmark from silently pointing into a new index.
    revision = hashlib.sha256(json.dumps(sorted(s.segment_id for s in segments), ensure_ascii=True,
                                         separators=(',', ':')).encode('utf-8')).hexdigest()
    gaps = []
    cursor = 0.0
    for segment in ordered:
        if segment.start_time > cursor:
            gaps.append((cursor, segment.start_time))
        cursor = max(cursor, min(segment.end_time, duration))
    if cursor < duration:
        gaps.append((cursor, duration))
    count = 0
    for start, end in gaps:
        windows = (end - start) / window_seconds
        if not math.isfinite(windows) or windows > max_gap_segments:
            raise ValueError('Visual-only gap segment limit exceeded')
        count += math.ceil(windows)
        if count > max_gap_segments:
            raise ValueError('Visual-only gap segment limit exceeded')
    result = list(segments)
    ids = {s.segment_id for s in segments}
    for start, end in gaps:
        # Calculate endpoints from the original start to avoid cumulative drift.
        n = math.ceil((end - start) / window_seconds)
        for index in range(n):
            left = start + index * window_seconds
            right = min(end, start + (index + 1) * window_seconds)
            if right <= left:
                raise ValueError('Window size is too small for this timeline')
            left, right = float(left), float(right)
            key = json.dumps([video_id, source_path, revision, duration, window_seconds, left, right],
                             ensure_ascii=True, separators=(',', ':'), allow_nan=False)
            segment_id = 'visual_gap_' + hashlib.sha256(key.encode('utf-8')).hexdigest()
            if segment_id in ids:
                raise ValueError('Visual-only segment ID collision')
            ids.add(segment_id)
            result.append(VideoSegment(segment_id=segment_id, video_id=video_id, source_path=source_path,
                                       start_time=left, end_time=right, transcript=''))
    return sorted(result, key=lambda s: (s.start_time, s.end_time, s.segment_id))
