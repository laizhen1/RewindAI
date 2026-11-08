import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import evaluate_local_segment_retrieval as evaluator
from scripts import freeze_local_segment_test as freezer
from scripts.review_retrieval_relevance import create_relevance_app
from scripts.temporal_label_server import FIELDS
from video_rag.retrieval import onnx_clip
from video_rag.schemas import SearchHit


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    private = tmp_path / 'private'; private.mkdir()
    video = tmp_path / 'test.mp4'; video.write_bytes(b'synthetic test video')
    subtitles = tmp_path / 'test.srt'
    subtitles.write_text('1\n00:00:01,000 --> 00:00:04,000\nSynthetic evidence.\n', encoding='utf-8')
    labels = private / 'test.csv'
    row = {field: '' for field in FIELDS}
    row.update(question_id='q1', video_id='test-video', split='test', question_type='audio', answerable='true',
               question='Private question?', reference_answer='private answer', evidence_start_seconds='1',
               evidence_end_seconds='4', annotation_status='draft_unverified')
    with labels.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS); writer.writeheader(); writer.writerow(row)
    binding = {'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'video_id': 'test-video', 'split': 'test'}
    labels.with_suffix('.csv.source.json').write_text(json.dumps(binding), encoding='utf-8')
    client = create_relevance_app(video, subtitles, labels, private / 'reviews', 'test-video', 'test', 10,
                                  allowed_roots=(private,)).test_client()
    segment_id = client.get('/api/data').json['segments'][0]['id']
    response = client.post('/api/save', json={'judgments': [{'question_id': 'q1', 'relevant_segment_ids': [segment_id],
                                                             'verified': True, 'exhaustive': True}]})
    gold = private / 'reviews' / response.json['filename']
    other = private / 'development-gold.json'
    other.write_text(json.dumps({'schema_version': 1, 'status': 'relevance_draft_not_independently_verified',
                                 'split': 'development', 'source_sha256': {'video': 'a' * 64}}), encoding='utf-8')
    embedding = tmp_path / 'embedding.gguf'; embedding.write_bytes(b'synthetic embedding')
    visual = tmp_path / 'visual'; (visual / 'onnx').mkdir(parents=True)
    (visual / 'tokenizer.json').write_text('{}', encoding='utf-8')
    (visual / 'onnx' / 'model_quantized.onnx').write_bytes(b'synthetic visual')
    class Retriever:
        def __init__(self, encoder):
            pass
        def build(self, segments):
            self.ids = [segment.segment_id for segment in segments]
        def search(self, query, top_k):
            return [SearchHit(value, 1, 'fake', rank) for rank, value in enumerate(self.ids[:top_k], 1)]
    versions = {name: 'synthetic' for name in ('llama-cpp-python', 'numpy', 'onnxruntime', 'tokenizers', 'av')}
    monkeypatch.setattr(evaluator, 'private_output', lambda path: Path(path))
    monkeypatch.setattr(freezer, 'private_output', lambda path: Path(path))
    monkeypatch.setattr(evaluator, 'dependency_versions', lambda: versions.copy())
    monkeypatch.setattr(evaluator, 'load_semantic_encoder', lambda path: object())
    monkeypatch.setattr(evaluator, 'SemanticRetriever', Retriever)
    monkeypatch.setattr(onnx_clip, 'OnnxClipEncoder', lambda path: object())
    monkeypatch.setattr(onnx_clip, 'ClipFrameRetriever', Retriever)
    common = ['--video', str(video), '--subtitles', str(subtitles), '--labels', str(labels), '--gold', str(gold),
              '--semantic-model', str(embedding), '--visual-model', str(visual), '--video-id', 'test-video',
              '--other-gold', str(other)]
    return private, video, gold, other, common


def test_freeze_and_evaluate_test_only_with_matching_declared_inputs(experiment, monkeypatch):
    private, _video, _gold, other, common = experiment
    manifest, output = private / 'frozen.json', private / 'result.json'
    freeze_args = ['freeze', *common, '--output', str(manifest), '--confirm-reviewed', '--confirm-test-isolation']
    monkeypatch.setattr(sys, 'argv', freeze_args[:-1])
    with pytest.raises(SystemExit):
        freezer.main()
    assert not manifest.exists()
    monkeypatch.setattr(sys, 'argv', freeze_args)
    assert freezer.main() == 0
    frozen = json.loads(manifest.read_text(encoding='utf-8'))
    assert frozen['status'] == 'frozen_segment_test_declared_not_independently_verified'
    args = ['evaluate', *common, '--split', 'test', '--frozen-test-manifest', str(manifest),
            '--output', str(output), '--confirm-reviewed']
    monkeypatch.setattr(sys, 'argv', args)
    assert evaluator.main() == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['test_declaration'] == frozen['status']
    assert result['metrics']['text']['overall']['recall@5'] == 1
    assert 'Private question' not in output.read_text(encoding='utf-8')
    other.write_text(other.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    second = private / 'changed.json'
    monkeypatch.setattr(sys, 'argv', [str(second) if item == str(output) else item for item in args])
    with pytest.raises(SystemExit):
        evaluator.main()
    assert not second.exists()


def test_freeze_rejects_same_video_and_test_requires_manifest(experiment, monkeypatch):
    private, video, _gold, other, common = experiment
    data = json.loads(other.read_text(encoding='utf-8'))
    data['source_sha256']['video'] = hashlib.sha256(video.read_bytes()).hexdigest()
    other.write_text(json.dumps(data), encoding='utf-8')
    manifest = private / 'rejected.json'
    monkeypatch.setattr(sys, 'argv', ['freeze', *common, '--output', str(manifest),
                                     '--confirm-reviewed', '--confirm-test-isolation'])
    with pytest.raises(SystemExit):
        freezer.main()
    assert not manifest.exists()
    monkeypatch.setattr(sys, 'argv', ['evaluate', *common, '--split', 'test', '--output', str(private / 'no.json'),
                                     '--confirm-reviewed'])
    with pytest.raises(SystemExit):
        evaluator.main()
    assert not (private / 'no.json').exists()


def test_frozen_test_rejects_changed_gold_model_and_runtime_before_scoring(experiment, monkeypatch):
    private, _video, gold, _other, common = experiment
    manifest, output = private / 'frozen.json', private / 'result.json'
    monkeypatch.setattr(sys, 'argv', ['freeze', *common, '--output', str(manifest),
                                     '--confirm-reviewed', '--confirm-test-isolation'])
    assert freezer.main() == 0
    args = ['evaluate', *common, '--split', 'test', '--frozen-test-manifest', str(manifest),
            '--output', str(output), '--confirm-reviewed']
    monkeypatch.setattr(sys, 'argv', args)
    original_gold = gold.read_bytes()
    gold.write_bytes(original_gold + b' ')
    with pytest.raises(SystemExit):
        evaluator.main()
    assert not output.exists()
    gold.write_bytes(original_gold)
    embedding = Path(common[common.index('--semantic-model') + 1])
    original_model = embedding.read_bytes()
    embedding.write_bytes(b'changed model')
    with pytest.raises(SystemExit):
        evaluator.main()
    assert not output.exists()
    embedding.write_bytes(original_model)
    monkeypatch.setattr(evaluator, 'dependency_versions', lambda: {'numpy': 'different'})
    with pytest.raises(SystemExit):
        evaluator.main()
    assert not output.exists()
