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
    assert result['paired_answerable_top1']['fused_only_hit'] == 1
    assert result['fused_experimental']['by_question_type']['visual']['metrics']['tp'] == 1


def test_comparison_rejects_invalid_threshold_and_empty_questions():
    with pytest.raises(ValueError):
        compare([], None, None)
    with pytest.raises(ValueError):
        compare([{'question': 'x'}], None, None, min_coverage=float('nan'))


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
