import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import freeze_answer_test, score_answer_review, start_answer_review
from scripts.review_answers import create_review_app
from scripts.score_answer_review import summarize
from scripts.temporal_label_server import FIELDS
from video_rag.adapters.local_answer import LocalAnswerGenerator
from video_rag.evaluation import frozen_answers


def test_frozen_answer_test_requires_manifest_and_human_verdicts(tmp_path, monkeypatch):
    root = tmp_path / 'private'; root.mkdir()
    video = tmp_path / 'video.mp4'; video.write_bytes(b'synthetic')
    subtitles = tmp_path / 'video.srt'
    subtitles.write_text('1\n00:00:01,000 --> 00:00:03,000\nBinary search halves the range.\n', encoding='utf-8')
    labels = root / 'test.csv'; row = {key: '' for key in FIELDS}
    row.update(question_id='q', video_id='v', split='test', question_type='audio', answerable='true',
               question='What does binary search do to the range?', reference_answer='Halves it',
               evidence_start_seconds='1', evidence_end_seconds='3')
    with labels.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerow(row)
    binding = {'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'video_id': 'v', 'split': 'test'}
    labels.with_suffix('.csv.source.json').write_text(json.dumps(binding), encoding='utf-8')
    model = tmp_path / 'model.gguf'; model.write_bytes(b'synthetic model')
    code = tmp_path / 'code.py'; code.write_text('original')
    monkeypatch.setattr(frozen_answers, 'ROOT', tmp_path)
    monkeypatch.setattr(frozen_answers, 'CODE_FILES', ('code.py',))
    monkeypatch.setattr(freeze_answer_test, 'private_output', lambda path: path)
    monkeypatch.setattr(freeze_answer_test, 'write_output',
                        lambda path, data, roots: Path(path).write_text(json.dumps(data), encoding='utf-8'))

    class Container:
        duration = 10_000_000
        def __enter__(self): return self
        def __exit__(self, *_): pass

    monkeypatch.setitem(sys.modules, 'av', SimpleNamespace(open=lambda _: Container(), time_base=1_000_000))
    manifest = root / 'answer-manifest.json'
    args = ['freeze', '--video', str(video), '--subtitles', str(subtitles), '--labels', str(labels),
            '--model', str(model), '--video-id', 'v', '--output', str(manifest)]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit):
        freeze_answer_test.main()
    assert not manifest.exists()
    monkeypatch.setattr(sys, 'argv', [*args, '--confirm-labels-reviewed'])
    assert freeze_answer_test.main() == 0

    class Generator:
        model_input = staticmethod(LocalAnswerGenerator.model_input)
        def __init__(self, path): pass
        def metadata(self): return {'prompt_version': 'explicit-evidence-v2'}
        def generate(self, query, segments): return 'It halves the range.'

    monkeypatch.setattr(start_answer_review, 'LocalAnswerGenerator', Generator)
    with pytest.raises(ValueError, match='reserved'):
        start_answer_review.create_draft(video, subtitles, labels, 'v', 'test', model, root / 'runs', allowed_roots=(root,))
    draft = start_answer_review.create_draft(video, subtitles, labels, 'v', 'test', model, root / 'runs',
                                             allowed_roots=(root,), frozen_test_manifest=manifest)
    report = json.loads(draft.read_text(encoding='utf-8'))
    assert report['test_declaration'] == 'frozen_answer_test_declared_not_independently_verified'
    frozen_answers.verify_review_manifest(report, manifest)
    altered = {**report, 'input_sha256': {**report['input_sha256'], 'video': 'wrong'}}
    with pytest.raises(ValueError, match='fingerprints'):
        frozen_answers.verify_review_manifest(altered, manifest)
    with pytest.raises(ValueError, match='verdicts'):
        summarize(report)
    with pytest.raises(ValueError, match='manifest'):
        create_review_app(video, draft, allowed_roots=(root,))
    client = create_review_app(video, draft, allowed_roots=(root,), frozen_test_manifest=manifest).test_client()
    response = client.post('/api/save', json={'confirm_human_review': True,
                                             'judgments': [{'answer': 'correct', 'evidence': 'supported'}]})
    assert response.status_code == 201 and response.json['metrics']['f1'] == 1.0
    reviewed = json.loads((draft.parent / response.json['filename']).read_text(encoding='utf-8'))
    assert summarize(reviewed)['tp'] == 1
    reviewed_path = draft.parent / response.json['filename']
    summary_path = root / 'summary.json'
    monkeypatch.setattr(score_answer_review, 'private_output', lambda path: path)
    scoring = ['score', '--review', str(reviewed_path), '--output', str(summary_path), '--confirm-human-review']
    monkeypatch.setattr(sys, 'argv', scoring)
    with pytest.raises(SystemExit):
        score_answer_review.main()
    assert not summary_path.exists()
    monkeypatch.setattr(sys, 'argv', [*scoring, '--frozen-test-manifest', str(manifest)])
    assert score_answer_review.main() == 0
    assert json.loads(summary_path.read_text(encoding='utf-8'))['frozen_manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='inputs'):
        start_answer_review.create_draft(video, subtitles, labels, 'v', 'test', model, root / 'runs',
                                         allowed_roots=(root,), frozen_test_manifest=manifest)
