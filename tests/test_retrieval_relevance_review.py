import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.review_retrieval_relevance import create_relevance_app
from scripts.temporal_label_server import FIELDS


@pytest.fixture
def sources(tmp_path):
    private = tmp_path / 'private'; private.mkdir()
    video = tmp_path / 'video.mp4'; video.write_bytes(b'synthetic video')
    subtitles = tmp_path / 'video.srt'
    subtitles.write_text('1\n00:00:01,000 --> 00:00:03,000\nFirst fact.\n\n'
                         '2\n00:00:30,000 --> 00:00:34,000\nSecond fact.\n', encoding='utf-8')
    labels = private / 'labels.csv'
    rows = []
    for index, start in enumerate((1, 30), 1):
        row = {field: '' for field in FIELDS}
        row.update(question_id=f'q{index}', video_id='sample', split='validation', question_type='audio',
                   answerable='true', question=f'Private question {index}?', reference_answer='private answer',
                   evidence_start_seconds=str(start), evidence_end_seconds=str(start + 2), annotation_status='draft_unverified')
        rows.append(row)
    with labels.open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    binding = {'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'video_id': 'sample', 'split': 'validation'}
    labels.with_suffix('.csv.source.json').write_text(json.dumps(binding), encoding='utf-8')
    return video, subtitles, labels, private


def app_for(sources):
    video, subtitles, labels, private = sources
    return create_relevance_app(video, subtitles, labels, private / 'reviews', 'sample', 'validation', 100,
                                allowed_roots=(private,))


def test_review_saves_private_judgments_in_question_order_and_restores(sources):
    _video, _subtitles, _labels, private = sources
    client = app_for(sources).test_client()
    response = client.get('/api/data')
    assert response.status_code == 200 and response.headers['Cache-Control'] == 'no-store'
    data = response.get_json()
    assert len(data['questions']) == len(data['segments']) == 2
    ids = [segment['id'] for segment in data['segments']]
    rows = [{'question_id': 'q2', 'relevant_segment_ids': [ids[1]], 'verified': True, 'exhaustive': False},
            {'question_id': 'q1', 'relevant_segment_ids': [ids[0]], 'verified': True, 'exhaustive': True}]
    saved = client.post('/api/save', json={'judgments': rows})
    assert saved.status_code == 201
    path = private / 'reviews' / saved.json['filename']
    raw = path.read_text(encoding='utf-8')
    assert 'Private question' not in raw and 'private answer' not in raw and 'First fact' not in raw
    restored = client.get('/api/latest').get_json()['judgments']
    assert [row['question_id'] for row in restored] == ['q1', 'q2']
    assert restored[0]['exhaustive'] is True and restored[1]['exhaustive'] is False
    assert client.get('/video', headers={'Range': 'bytes=0-3'}).status_code == 206
    assert client.get('/api/data', headers={'Host': 'outside.example'}).status_code == 403
    assert client.post('/api/save', json={'judgments': rows}, headers={'Origin': 'https://outside.example'}).status_code == 403


def test_rejects_invalid_or_unconfirmed_labels_and_preserves_sources(sources):
    _video, subtitles, _labels, private = sources
    client = app_for(sources).test_client()
    data = client.get('/api/data').json
    ids = [segment['id'] for segment in data['segments']]
    rows = [{'question_id': 'q1', 'relevant_segment_ids': [ids[0]], 'verified': True, 'exhaustive': True},
            {'question_id': 'q2', 'relevant_segment_ids': [], 'verified': False, 'exhaustive': False}]
    invalid = [
        [{**rows[0], 'relevant_segment_ids': ['unknown']}, rows[1]],
        [rows[0], {**rows[1], 'question_id': 'q1'}],
        [rows[0], {**rows[1], 'verified': True}],
        [rows[0], {**rows[1], 'exhaustive': True}],
        [{**rows[0], 'verified': 1}, rows[1]],
    ]
    for candidate in invalid:
        assert client.post('/api/save', json={'judgments': candidate}).status_code == 400
    assert not (private / 'reviews').exists()
    subtitles.write_text('changed', encoding='utf-8')
    assert client.post('/api/save', json={'judgments': rows}).status_code == 409
    assert not (private / 'reviews').exists()


def test_source_binding_and_private_output_are_enforced(sources, tmp_path):
    video, subtitles, labels, private = sources
    with pytest.raises(ValueError, match='Private'):
        create_relevance_app(video, subtitles, labels, tmp_path / 'public', 'sample', 'validation', 100,
                             allowed_roots=(private,))
    with pytest.raises(ValueError, match='match'):
        create_relevance_app(video, subtitles, labels, private / 'reviews', 'other', 'validation', 100,
                             allowed_roots=(private,))


def test_unanswerable_questions_are_explicitly_excluded(sources):
    _video, _subtitles, labels, _private = sources
    row = {field: '' for field in FIELDS}
    row.update(question_id='q3', video_id='sample', split='validation', question_type='audio',
               answerable='false', question='Unanswerable?', checked_scope='entire video')
    with labels.open('a', encoding='utf-8', newline='') as out:
        csv.DictWriter(out, fieldnames=FIELDS).writerow(row)
    data = app_for(sources).test_client().get('/api/data').json
    assert data['unanswerable_excluded'] == 1
    assert [item['question_id'] for item in data['questions']] == ['q1', 'q2']


def test_scoring_requires_declared_complete_labels_and_exact_segmentation(sources):
    from scripts.evaluate_local_segment_retrieval import validate_gold
    from video_rag.transcript_baseline import segments_from_subtitles

    video, subtitles, _labels, private = sources
    client = app_for(sources).test_client()
    data = client.get('/api/data').json
    ids = [segment['id'] for segment in data['segments']]
    rows = [{'question_id': 'q1', 'relevant_segment_ids': [ids[0]], 'verified': True, 'exhaustive': True},
            {'question_id': 'q2', 'relevant_segment_ids': [ids[1]], 'verified': True, 'exhaustive': False}]
    saved = client.post('/api/save', json={'judgments': rows}).json
    gold = json.loads((private / 'reviews' / saved['filename']).read_text(encoding='utf-8'))
    segments = segments_from_subtitles(video, subtitles)
    kwargs = {'hashes': gold['source_sha256'], 'video_id': 'sample', 'split': 'validation',
              'questions': data['questions'], 'segments': segments}
    with pytest.raises(ValueError, match='exhaustively'):
        validate_gold(gold, **kwargs)
    gold['judgments'][1]['exhaustive'] = True
    assert validate_gold(gold, **kwargs) == {'q1': [ids[0]], 'q2': [ids[1]]}
    with pytest.raises(ValueError, match='segmentation'):
        validate_gold(gold, **{**kwargs, 'segments': [replace(segments[0], transcript='changed'), segments[1]]})
    with pytest.raises(ValueError, match='sources'):
        validate_gold(gold, **{**kwargs, 'hashes': {**gold['source_sha256'], 'video': '0' * 64}})


def test_paired_comparison_uses_fractional_recall_and_does_not_assume_fusion_improves():
    from scripts.evaluate_local_segment_retrieval import compare_rankings
    from video_rag.schemas import SearchHit

    class Retriever:
        def __init__(self, ids):
            self.ids = ids
        def search(self, query, top_k):
            return [SearchHit(value, 1, 'synthetic', i) for i, value in enumerate(self.ids[:top_k], 1)]
    questions = [{'question_id': 'q', 'question': 'Question?', 'question_type': 'audio', 'expected_answerable': True}]
    result = compare_rankings(questions, {'q': ['a', 'b']}, Retriever(['a', 'c', 'd', 'e', 'f']),
                              Retriever(['b', 'c', 'd', 'e', 'f']), set('abcdef'))
    assert result['text']['overall']['recall@5'] == 0.5
    assert result['text']['overall']['hit@5'] == 1
    assert result['rrf']['overall']['mrr@5'] == 0.2
    with pytest.raises(ValueError, match='invalid'):
        compare_rankings(questions, {'q': ['a']}, Retriever(['unknown']), Retriever(['b']), set('ab'))


def test_optional_reranker_reorders_deeper_same_text_candidates_in_bounded_batches():
    from types import SimpleNamespace

    from scripts.evaluate_local_segment_retrieval import compare_rankings
    from video_rag.schemas import SearchHit

    class Retriever:
        def __init__(self, ids):
            self.ids = ids
        def search(self, query, top_k):
            return [SearchHit(value, 1, 'fake', rank) for rank, value in enumerate(self.ids[:top_k], 1)]
    class Reranker:
        def __init__(self):
            self.batch_sizes = []
        def score(self, query, candidates):
            self.batch_sizes.append(len(candidates))
            return [1.0 if candidate.segment_id == 'g' else 0.0 for candidate in candidates]
    ids = list('abcdefg')
    segments = {value: SimpleNamespace(segment_id=value, transcript='synthetic') for value in ids}
    questions = [{'question_id': 'q', 'question': 'Question?', 'question_type': 'audio', 'expected_answerable': True}]
    reranker = Reranker()
    result = compare_rankings(questions, {'q': ['g']}, Retriever(ids), Retriever(ids), set(ids),
                              reranker=reranker, segments_by_id=segments)
    assert result['text']['overall']['recall@5'] == 0
    assert result['text_reranked']['overall']['recall@5'] == 1
    assert result['text_reranked']['overall']['mrr@5'] == 1
    assert reranker.batch_sizes == [4, 3]


def test_local_comparison_saves_only_aggregate_and_detects_source_mutation(sources, tmp_path, monkeypatch):
    from scripts import evaluate_local_segment_retrieval as module
    from video_rag.retrieval import onnx_clip
    from video_rag.schemas import SearchHit

    video, subtitles, labels, private = sources
    negative = {field: '' for field in FIELDS}
    negative.update(question_id='q3', video_id='sample', split='validation', question_type='audio',
                    answerable='false', question='Unanswerable?', checked_scope='entire video')
    with labels.open('a', encoding='utf-8', newline='') as out:
        csv.DictWriter(out, fieldnames=FIELDS).writerow(negative)
    client = app_for(sources).test_client()
    data = client.get('/api/data').json
    ids = [segment['id'] for segment in data['segments']]
    judgments = [{'question_id': f'q{i + 1}', 'relevant_segment_ids': [ids[i]],
                  'verified': True, 'exhaustive': True} for i in range(2)]
    gold = private / 'reviews' / client.post('/api/save', json={'judgments': judgments}).json['filename']
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
            return [SearchHit(value, 1, 'fake', i) for i, value in enumerate(self.ids, 1)]
    monkeypatch.setattr(module, 'private_output', lambda path: Path(path))
    monkeypatch.setattr(module, 'load_semantic_encoder', lambda path: object())
    monkeypatch.setattr(module, 'SemanticRetriever', Retriever)
    monkeypatch.setattr(onnx_clip, 'OnnxClipEncoder', lambda path: object())
    monkeypatch.setattr(onnx_clip, 'ClipFrameRetriever', Retriever)
    output = private / 'comparison.json'
    args = ['evaluate', '--video', str(video), '--subtitles', str(subtitles), '--labels', str(labels),
            '--gold', str(gold), '--semantic-model', str(embedding), '--visual-model', str(visual),
            '--output', str(output), '--video-id', 'sample', '--split', 'validation', '--confirm-reviewed']
    monkeypatch.setattr(sys, 'argv', args)
    assert module.main() == 0
    text = output.read_text(encoding='utf-8')
    report = json.loads(text)
    assert report['settings']['device'] == 'cpu' and report['answerable_questions'] == 2
    assert report['unanswerable_questions_excluded'] == 1
    assert set(report['metrics']) == {'text', 'frame', 'rrf'}
    assert 'Private question' not in text and 'private answer' not in text and 'First fact' not in text
    assert report['metrics']['text']['overall']['recall@5'] == 1
    with pytest.raises(SystemExit):
        module.main()
    from video_rag.adapters import qwen

    reranker_dir = tmp_path / 'reranker'; reranker_dir.mkdir()
    (reranker_dir / 'config.json').write_text('{}', encoding='utf-8')
    (reranker_dir / 'model.safetensors').write_bytes(b'synthetic weights')
    class FakeReranker:
        def __init__(self, model, *, device_map, max_length, local_files_only):
            assert device_map == 'cpu' and max_length == 1024 and local_files_only is True
        def score(self, query, candidates):
            assert all(self._document(item) == item.transcript for item in candidates)
            return [0.5] * len(candidates)
    monkeypatch.setattr(qwen, 'Qwen3Reranker', FakeReranker)
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '0')
    reranked_output = private / 'reranked.json'
    rerank_args = [str(reranked_output) if value == str(output) else value for value in args]
    monkeypatch.setattr(sys, 'argv', [*rerank_args, '--reranker-model', str(reranker_dir)])
    assert module.main() == 0
    reranked = json.loads(reranked_output.read_text(encoding='utf-8'))
    assert 'text_reranked' in reranked['metrics'] and reranked['settings']['rerank_candidate_depth'] == 20
    assert 'text_reranked_minus_text' in reranked['paired_comparisons']
    assert 'reranker:model.safetensors' in reranked['source_sha256']
    assert module.os.environ['HF_HUB_OFFLINE'] == module.os.environ['TRANSFORMERS_OFFLINE'] == '1'
    output2 = private / 'changed.json'
    class MutatingRetriever(Retriever):
        def search(self, query, top_k):
            embedding.write_bytes(b'changed embedding')
            return super().search(query, top_k)
    monkeypatch.setattr(module, 'SemanticRetriever', MutatingRetriever)
    monkeypatch.setattr(sys, 'argv', [str(output2) if value == str(output) else value for value in args])
    with pytest.raises(SystemExit):
        module.main()
    assert not output2.exists()


def test_paired_comparison_counts_improvements_regressions_and_ties():
    from scripts.evaluate_local_segment_retrieval import paired_differences

    gold = {'q1': ['a'], 'q2': ['b'], 'q3': ['c']}
    rankings = {'text': {'q1': ['x'], 'q2': ['b'], 'q3': ['c']},
                'frame': {'q1': ['a'], 'q2': ['x'], 'q3': ['c']}}
    result = paired_differences(rankings, gold)['frame_minus_text']
    assert result['question_count'] == 3
    for metric in ('recall@5', 'hit@5', 'mrr@5'):
        assert result['metrics'][metric] == {'mean_difference': 0, 'improved': 1, 'worsened': 1, 'tied': 1}
    with pytest.raises(ValueError):
        paired_differences({'text': {'q1': ['a']}}, gold)
