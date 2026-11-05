"""Check source-bound test draft structure without claiming independent review."""
from __future__ import annotations

import csv

from scripts.compare_temporal_multimodal import load_questions


def validate_test_labels(video, labels, video_id, video_hash):
    data, _, _ = load_questions(labels, video_id, 'test', video_hash, allow_test=True)
    if any(q['expected_answerable'] and not q.get('expected_answer', '').strip() for q in data['questions']):
        raise ValueError('Every answerable test question needs a reference answer')
    import av

    with av.open(str(video)) as container:
        duration = container.duration / av.time_base if container.duration is not None else None
    if duration is None or duration <= 0 or any(q['expected_answerable'] and q['evidence_end_seconds'] > duration
                                                 for q in data['questions']):
        raise ValueError('Video duration is unavailable or a reference interval exceeds the video')
    with labels.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if any(row.get('answerable', '').strip().lower() == 'false' and row.get('notes') != 'scope_declared_entire_video'
           for row in rows if row.get('video_id', '').strip() == video_id):
        raise ValueError('Unanswerable questions require a declared whole-video check')
