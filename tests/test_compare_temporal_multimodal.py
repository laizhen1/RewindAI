import pytest

from scripts.compare_temporal_multimodal import (
    EvidenceOnly,
    FusedCandidates,
    compare,
    load_questions,
)
from video_rag.api import create_app
from video_rag.schemas import SearchHit, VideoSegment
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


def test_fused_comparison_uses_same_questions_and_reports_negative_candidates():
    segments = [VideoSegment('a', 'v', 'local', 0, 10, transcript='attention keys queries'),
                VideoSegment('b', 'v', 'local', 20, 30, transcript='gradient descent')]

    class NoGeneration:
        def generate(self, query, segments):
            raise AssertionError('Candidate evaluation must not generate answers')

    class Visual:
        def search(self, query, top_k):
            return [SearchHit('b' if query == 'diagram' else 'a', 0.2, 'visual', 1)]

    pipeline = build_transcript_segments(segments, generator=NoGeneration())
    result = compare([
        {'question': 'attention', 'question_type': 'audio', 'expected_answerable': True,
         'evidence_start_seconds': 2, 'evidence_end_seconds': 8},
        {'question': 'diagram', 'question_type': 'visual', 'expected_answerable': True,
         'evidence_start_seconds': 22, 'evidence_end_seconds': 28},
        {'question': 'unrelated elephants', 'question_type': 'audio', 'expected_answerable': False},
    ], EvidenceOnly(pipeline), FusedCandidates(pipeline, Visual(), segments))
    assert result['transcript']['metrics']['tp'] == 1
    assert result['fused_experimental']['metrics']['tp'] == 2
    assert result['fused_experimental']['operating']['unanswerable_returned_candidates'] == 1
    assert result['fused_experimental']['temporal_recall_at_3'] == {
        'hits': 2, 'answerable_questions': 2, 'rate': 1.0, 'min_reference_coverage': 0.5}
    assert result['paired_answerable_top1']['fused_only_hit'] == 1
    assert result['paired_unanswerable']['transcript_only_abstains'] == 1
    assert result['results'][1]['fused']['returned_sources'] == [['visual']]
    assert result['results'][0]['fused']['returned_sources'][0] == ['transcript', 'visual']
    assert result['fused_experimental']['by_question_type']['visual']['metrics']['tp'] == 1
    assert result['results'][1]['reference_range'] == [22.0, 28.0]
    assert result['results'][2]['reference_range'] is None


def test_comparison_rejects_invalid_threshold_and_empty_questions():
    with pytest.raises(ValueError):
        compare([], None, None)
    with pytest.raises(ValueError):
        compare([{'question': 'x'}], None, None, min_coverage=float('nan'))


def test_experimental_visual_threshold_can_remove_candidates():
    segments = [VideoSegment('a', 'v', 'local', 0, 5, transcript='attention')]

    class Visual:
        def search(self, query, top_k):
            return [SearchHit('a', 0.2, 'visual', 1)]

    pipeline = build_transcript_segments(segments)
    unfiltered = FusedCandidates(pipeline, Visual(), segments).ask('elephants').to_dict()
    filtered = FusedCandidates(pipeline, Visual(), segments, min_visual_score=0.3).ask('elephants').to_dict()
    assert unfiltered['evidence'] and not unfiltered['abstained']
    assert filtered['evidence'] == [] and filtered['abstained']
    assert FusedCandidates(pipeline, Visual(), segments, min_visual_score=0.3).ask('attention').to_dict()['evidence']
    with pytest.raises(ValueError, match='threshold'):
        FusedCandidates(pipeline, Visual(), segments, min_visual_score=float('nan'))


def test_fused_evaluation_matches_workspace_candidate_ranking(tmp_path):
    video = tmp_path / 'video.mp4'; video.write_bytes(b'synthetic')
    subtitle = tmp_path / 'video.srt'
    subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nAttention queries.\n\n'
                        '2\n00:00:20,000 --> 00:00:23,000\nAnother concept.\n', encoding='utf-8')
    segments = segments_from_subtitles(video, subtitle)

    class Visual:
        def search(self, query, top_k):
            return [SearchHit(segments[1].segment_id, 0.8, 'visual', 1),
                    SearchHit(segments[0].segment_id, 0.5, 'visual', 2)]

    pipeline = build_transcript_segments(segments)
    result = FusedCandidates(pipeline, Visual(), segments).ask('attention').to_dict()
    client = create_app(pipeline, mode='transcript', visual_builder=lambda _: Visual(),
                        database_path=tmp_path / 'state.db').test_client()
    workspace = client.post('/api/hybrid-search', json={'question': 'attention'}).json
    assert [(x['start_time'], x['end_time']) for x in result['evidence']] == [
        (x['start_time'], x['end_time']) for x in workspace['evidence']]


def test_csv_labels_require_matching_video_and_non_test_split(tmp_path):
    import csv
    import json

    from scripts.temporal_label_server import FIELDS

    path = tmp_path / 'labels.csv'
    item = {key: '' for key in FIELDS}
    item.update(question_id='q', video_id='v', split='development', question_type='audio',
                answerable='true', question='What is attention?', reference_answer='Mapping',
                evidence_start_seconds='1', evidence_end_seconds='3')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(item)
    sidecar = path.with_suffix('.csv.source.json')
    sidecar.write_text(json.dumps({'video_sha256': 'video-hash', 'video_id': 'v', 'split': 'development'}))
    data, binding, _ = load_questions(path, 'v', 'development', 'video-hash')
    assert data['questions'][0]['question_type'] == 'audio'
    assert binding == 'annotation_video_matched_subtitles_not_bound'
    with pytest.raises(ValueError, match='binding'):
        load_questions(path, 'v', 'development', 'different-hash')
    with pytest.raises(ValueError, match='development or validation'):
        load_questions(path, 'v', 'test', 'video-hash')


def test_json_questions_reject_test_and_unknown_splits(tmp_path):
    import json

    path = tmp_path / 'questions.json'
    for split in ('test', 'Test', 'unknown'):
        path.write_text(json.dumps({'split': split, 'questions': []}), encoding='utf-8')
        with pytest.raises(ValueError, match='reserved'):
            load_questions(path, None, None, 'hash')
    path.write_text(json.dumps({'split': 'validation', 'questions': []}), encoding='utf-8')
    data, binding, _ = load_questions(path, None, None, 'hash')
    assert data['split'] == 'validation' and binding is None


def test_gap_temporal_fusion_matches_workspace_without_changing_text_index(tmp_path, monkeypatch):
    from video_rag.api import visual_timeline

    video = tmp_path / 'video.mp4'; video.write_bytes(b'synthetic')
    subtitle = tmp_path / 'video.srt'
    subtitle.write_text('1\n00:00:10,000 --> 00:00:12,000\nAttention queries.\n', encoding='utf-8')
    segments = segments_from_subtitles(video, subtitle)
    timeline = visual_timeline.cover_subtitle_gaps(segments, 40)
    gap = next(s for s in timeline if not s.transcript)
    monkeypatch.setattr(visual_timeline, 'video_duration_seconds', lambda _path: 40.0)

    class Visual:
        def search(self, _query, _top_k):
            return [SearchHit(gap.segment_id, 0.8, 'visual', 1)]

    pipeline = build_transcript_segments(segments)
    result = FusedCandidates(pipeline, Visual(), timeline).ask('attention').to_dict()
    client = create_app(pipeline, mode='transcript', visual_builder=lambda _: Visual(), visual_cover_gaps=True,
                        database_path=tmp_path / 'state.db').test_client()
    workspace = client.post('/api/hybrid-search', json={'question': 'attention'}).json
    assert [(x['start_time'], x['end_time']) for x in result['evidence']] == [
        (x['start_time'], x['end_time']) for x in workspace['evidence']]
    assert all(item.segment.transcript for item in pipeline.ask('attention', generate_answer=False).evidence)


def test_gap_temporal_diagnostic_does_not_run_on_unfrozen_test(monkeypatch, capsys):
    import sys

    from scripts import compare_temporal_multimodal as module

    monkeypatch.setattr(sys, 'argv', ['compare', '--video', 'x', '--subtitles', 'x', '--questions', 'x',
                                     '--visual-model', 'x', '--output', str(module.ROOT / 'artifacts' / 'gap-test-invalid.json'),
                                     '--split', 'test', '--cover-gaps'])
    with pytest.raises(SystemExit):
        module.main()
    assert 'reserved for development and validation' in capsys.readouterr().err
