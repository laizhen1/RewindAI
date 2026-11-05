import csv
import hashlib
import json

import pytest

from scripts.start_answer_review import create_draft
from scripts.temporal_label_server import FIELDS


@pytest.fixture
def sources(tmp_path):
    root = tmp_path / 'private'; root.mkdir()
    video = tmp_path / 'video.mp4'; video.write_bytes(b'synthetic')
    subs = tmp_path / 'video.srt'; subs.write_text('1\n00:00:01,000 --> 00:00:03,000\nBinary search halves the range.\n', encoding='utf-8')
    model = tmp_path / 'model.gguf'; model.write_bytes(b'model')
    labels = root / 'labels.csv'
    row = {key: '' for key in FIELDS}
    row.update(question_id='q1', video_id='sample', split='development', question_type='audio',
               answerable='true', question='What does binary search do to the range?',
               reference_answer='Halves it', evidence_start_seconds='1', evidence_end_seconds='3',
               annotation_status='draft_unverified')
    with labels.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerow(row)
    binding = {'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'video_id': 'sample', 'split': 'development'}
    labels.with_suffix('.csv.source.json').write_text(json.dumps(binding), encoding='utf-8')
    return video, subs, labels, model, root


def test_create_draft_from_bound_local_labels(sources, monkeypatch):
    video, subs, labels, model, root = sources

    class Generator:
        def __init__(self, path):
            assert path == model

        def generate(self, query, segments):
            return 'It halves the range.'

    monkeypatch.setattr('scripts.start_answer_review.LocalAnswerGenerator', Generator)
    target = create_draft(video, subs, labels, 'sample', 'development', model, root / 'runs', allowed_roots=(root,))
    report = json.loads(target.read_text(encoding='utf-8'))
    assert report['status'] == 'unreviewed_answer_drafts' and report['binding'] == 'video_matched'
    assert report['rows'][0]['human_answer_verdict'] is None
    assert report['rows'][0]['answer'] == 'It halves the range.'
    assert report['input_sha256']['video'] == hashlib.sha256(video.read_bytes()).hexdigest()


def test_rejects_wrong_binding_and_test_split_without_outputs(sources):
    video, subs, labels, model, root = sources
    with pytest.raises(ValueError, match='reserved'):
        create_draft(video, subs, labels, 'sample', 'test', model, root / 'runs', allowed_roots=(root,))
    video.write_bytes(b'different video')
    with pytest.raises(ValueError, match='match'):
        create_draft(video, subs, labels, 'sample', 'development', model, root / 'runs', allowed_roots=(root,))
    assert not (root / 'runs').exists()
