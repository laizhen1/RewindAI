"""Deterministic bounded frame sampling for local visual search."""
from __future__ import annotations

from collections import defaultdict

from video_rag.schemas import VideoSegment


def sample_visual_segments(segments: tuple[VideoSegment, ...], limit: int = 100) -> list[VideoSegment]:
    """Spread the budget across videos, retaining at least one segment per video."""
    if limit <= 0:
        raise ValueError("Sampling limit must be positive")
    groups: dict[str, list[VideoSegment]] = defaultdict(list)
    for segment in segments:
        groups[segment.video_id].append(segment)
    if len(groups) > limit:
        raise ValueError("The selected scope contains too many videos")
    for group in groups.values():
        group.sort(key=lambda s: (s.start_time, s.end_time, s.segment_id))
    if len(segments) <= limit:
        return [segment for key in sorted(groups) for segment in groups[key]]
    # Guarantee one per video, then allocate remaining slots in proportion to
    # each video's remaining segments using the largest-remainder method.
    remaining = limit - len(groups)
    spare = sum(len(group) - 1 for group in groups.values())
    quotas = {key: remaining * (len(group) - 1) / spare for key, group in groups.items()}
    counts = {key: 1 + int(quota) for key, quota in quotas.items()}
    left = limit - sum(counts.values())
    for key in sorted(groups, key=lambda item: (-(quotas[item] % 1), item))[:left]:
        counts[key] += 1
    selected = []
    for key in sorted(groups):
        group, count = groups[key], counts[key]
        positions = ([(len(group) - 1) // 2] if count == 1 else
                     [i * (len(group) - 1) // (count - 1) for i in range(count)])
        selected.extend(group[index] for index in positions)
    return selected
