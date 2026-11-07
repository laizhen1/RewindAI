import csv
import hashlib
import json

import pytest

from scripts.start_answer_review import create_draft
from scripts.temporal_label_server import FIELDS
from video_rag.adapters.local_answer import LocalAnswerGenerator


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
        model_input = staticmethod(LocalAnswerGenerator.model_input)

        def __init__(self, path):
            assert path == model

        def metadata(self):
            return {'prompt_version': 'test'}

        def generate(self, query, segments):
            return 'It halves the range.'

    monkeypatch.setattr('scripts.start_answer_review.LocalAnswerGenerator', Generator)
    target = create_draft(video, subs, labels, 'sample', 'development', model, root / 'runs', allowed_roots=(root,))
    report = json.loads(target.read_text(encoding='utf-8'))
    assert report['status'] == 'unreviewed_answer_drafts' and report['binding'] == 'video_matched'
    assert report['generation_config']['prompt_version'] == 'test'
    assert report['retrieval_config'] == {'type': 'keyword', 'threshold': None}
    assert report['rows'][0]['human_answer_verdict'] is None
    assert report['rows'][0]['answer'] == 'It halves the range.'
    assert report['rows'][0]['model_question'] == report['rows'][0]['question']
    assert report['rows'][0]['evidence'][0]['model_excerpt'] == 'Binary search halves the range.'
    assert report['input_sha256']['video'] == hashlib.sha256(video.read_bytes()).hexdigest()


def test_generation_failure_is_preserved_in_private_draft(sources, monkeypatch):
    from scripts.review_answers import create_review_app
    from video_rag.adapters.local_answer import LocalAnswerGenerationError

    video, subs, labels, model, root = sources

    class Generator:
        model_input = staticmethod(LocalAnswerGenerator.model_input)
        def __init__(self, path):
            pass
        def metadata(self):
            return {'prompt_version': 'test'}
        def generate(self, query, segments):
            raise LocalAnswerGenerationError('Model failed')

    monkeypatch.setattr('scripts.start_answer_review.LocalAnswerGenerator', Generator)
    target = create_draft(video, subs, labels, 'sample', 'development', model, root / 'runs', allowed_roots=(root,))
    report = json.loads(target.read_text(encoding='utf-8'))
    assert report['rows'][0]['generation_error'] == 'Model failed'
    assert report['rows'][0]['evidence'] and 'model_excerpt' not in report['rows'][0]['evidence'][0]
    client = create_review_app(video, target, allowed_roots=(root,)).test_client()
    assert client.get('/api/review').json['rows'][0]['generation_error'] == 'Model failed'
    response = client.post('/api/save', json={'confirm_human_review': True,
                           'judgments': [{'answer': 'abstain', 'evidence': 'not_applicable'}]})
    assert response.status_code == 400 and 'Generation failed' in response.json['error']


def test_create_semantic_draft_records_models_and_configuration(sources, monkeypatch):
    import numpy as np

    from video_rag.retrieval import onnx_semantic

    video, subs, labels, model, root = sources
    semantic = root / 'encoder'; (semantic / 'onnx').mkdir(parents=True)
    (semantic / 'tokenizer.json').write_text('{}')
    (semantic / 'onnx' / 'model_quantized.onnx').write_bytes(b'encoder')

    class Generator:
        model_input = staticmethod(LocalAnswerGenerator.model_input)

        def __init__(self, path):
            pass
        def metadata(self):
            return {'prompt_version': 'test'}
        def generate(self, query, segments):
            return 'It halves the range.'

    class Encoder:
        def __init__(self, path):
            assert path == semantic
        def encode(self, texts):
            return np.ones((len(texts), 1), dtype=np.float32)

    monkeypatch.setattr('scripts.start_answer_review.LocalAnswerGenerator', Generator)
    monkeypatch.setattr(onnx_semantic, 'OnnxSentenceEncoder', Encoder)
    target = create_draft(video, subs, labels, 'sample', 'development', model, root / 'semantic-runs',
                          allowed_roots=(root,), semantic_model=semantic, semantic_threshold=0.4)
    report = json.loads(target.read_text(encoding='utf-8'))
    assert report['retrieval_config'] == {'type': 'semantic_experimental', 'threshold': 0.4}
    assert report['input_sha256']['semantic_model'] == hashlib.sha256(b'encoder').hexdigest()
    assert report['rows'][0]['answer'] == 'It halves the range.'
    assert report['rows'][0]['model_question'] == report['rows'][0]['question']
    assert report['rows'][0]['evidence'][0]['model_excerpt'] == 'Binary search halves the range.'


def test_rejects_wrong_binding_and_test_split_without_outputs(sources):
    video, subs, labels, model, root = sources
    with pytest.raises(ValueError, match='reserved'):
        create_draft(video, subs, labels, 'sample', 'test', model, root / 'runs', allowed_roots=(root,))
    video.write_bytes(b'different video')
    with pytest.raises(ValueError, match='match'):
        create_draft(video, subs, labels, 'sample', 'development', model, root / 'runs', allowed_roots=(root,))
    assert not (root / 'runs').exists()
