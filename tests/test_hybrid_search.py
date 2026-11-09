from video_rag.api import create_app
from video_rag.schemas import SearchHit
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


class NoGeneration:
    def generate(self, query, segments):
        raise AssertionError('Hybrid search must not generate an answer')


def test_hybrid_fuses_ranked_candidates_and_never_generates(tmp_path):
    video = tmp_path / 'sample.mp4'; video.write_bytes(b'synthetic')
    subtitle = tmp_path / 'sample.srt'
    subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nAttention maps query to keys.\n\n'
                        '2\n00:00:30,000 --> 00:00:34,000\nA slide describes gradients.\n', encoding='utf-8')
    segments = segments_from_subtitles(video, subtitle)
    pipeline = build_transcript_segments(segments, generator=NoGeneration())

    class Visual:
        def search(self, query, top_k):
            return [SearchHit(segments[0].segment_id, 0.7, 'visual', 1),
                    SearchHit(segments[1].segment_id, 0.6, 'visual', 2)]

    client = create_app(pipeline, mode='transcript', visual_builder=lambda values: Visual(),
                        database_path=tmp_path / 'library.db').test_client()
    result = client.post('/api/hybrid-search', json={'question': 'attention'}).json
    assert result['visual_available'] and result['evidence'][0]['sources'] == ['transcript', 'visual']
    assert result['evidence'][1]['sources'] == ['visual']
    assert result['evidence'][1]['transcript'] == ''
    assert all('source_path' not in item for item in result['evidence'])
    assert result['warning'].endswith('No answer generated.')
    assert client.post('/api/hybrid-search', json={'question': 'attention', 'video_id': 'missing'}).status_code == 404
    compared = client.post('/api/compare-search', json={'question': 'attention'}).json
    assert compared['transcript']['evidence'] and compared['transcript']['answer'] == ''
    assert compared['visual']['evidence']


def test_hybrid_degrades_to_transcript_when_visual_fails(tmp_path):
    video = tmp_path / 'sample.mp4'; video.write_bytes(b'synthetic')
    subtitle = tmp_path / 'sample.srt'; subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nAttention maps keys.\n')
    pipeline = build_transcript_segments(segments_from_subtitles(video, subtitle), generator=NoGeneration())

    def fail(_segments):
        raise ValueError('private path')

    client = create_app(pipeline, mode='transcript', visual_builder=fail, database_path=tmp_path / 'library.db').test_client()
    result = client.post('/api/hybrid-search', json={'question': 'attention'}).json
    assert not result['visual_available'] and result['evidence'][0]['sources'] == ['transcript']
    assert 'private path' not in result['visual_error']


def test_optional_gap_visual_search_supports_frames_bookmarks_and_restart(tmp_path, monkeypatch):
    from video_rag.api import frames, visual_timeline

    video = tmp_path / 'sample.mp4'; video.write_bytes(b'synthetic')
    subtitle = tmp_path / 'sample.srt'
    subtitle.write_text('1\n00:00:10,000 --> 00:00:12,000\nAttention maps keys.\n', encoding='utf-8')
    monkeypatch.setattr(visual_timeline, 'video_duration_seconds', lambda _path: 40.0)
    monkeypatch.setattr(frames, 'extract_frame_png', lambda _path, seconds: (b'synthetic PNG', seconds))

    class Visual:
        def __init__(self, segments):
            self.gaps = [s for s in segments if not s.transcript]
        def search(self, _query, top_k):
            return [SearchHit(s.segment_id, .8, 'visual', i + 1, {'frame_time': (s.start_time + s.end_time) / 2})
                    for i, s in enumerate(self.gaps[:top_k])]

    database = tmp_path / 'library.db'
    def app():
        return create_app(build_transcript_segments([]), mode='transcript', visual_builder=Visual,
                          visual_cover_gaps=True, database_path=database)

    client = app().test_client()
    assert client.get('/api/health').json['visual_cover_gaps'] is True
    added = client.post('/api/library/local-video', json={'video_path': str(video), 'subtitle_path': str(subtitle)})
    assert added.status_code == 201
    video_id = added.json['video_id']
    query = {'question': 'attention', 'video_id': video_id}
    text_before = client.post('/api/ask', json=query).json['evidence']
    visual = client.post('/api/visual-search', json=query).json
    gap = visual['evidence'][0]
    assert visual['visual_cover_gaps'] and gap['start_time'] == 0 and gap['end_time'] == 10
    assert gap['frame_time'] == 5
    assert client.get(f"/api/segments/{gap['segment_id']}/frame").status_code == 200
    hybrid = client.post('/api/hybrid-search', json=query).json
    assert any(item['segment_id'] == gap['segment_id'] and item['sources'] == ['visual'] and item['frame_time'] == 5
               for item in hybrid['evidence'])
    assert client.post('/api/bookmarks', json={'video_id': video_id, 'segment_id': gap['segment_id']}).status_code == 201
    assert client.post('/api/feedback', json={'question': 'attention', 'segment_id': gap['segment_id'],
                                               'rating': 'helpful', 'target_type': 'visual'}).status_code == 201
    assert client.post('/api/ask', json=query).json['evidence'] == text_before

    restored = app().test_client()
    assert restored.get('/api/bookmarks').json['bookmarks'][0]['available'] is True
    subtitle.write_text('1\n00:00:10,000 --> 00:00:12,000\nChanged subtitle.\n', encoding='utf-8')
    assert restored.put(f'/api/library/videos/{video_id}/subtitles', json={'subtitle_path': str(subtitle)}).status_code == 200
    assert restored.get('/api/bookmarks').json['bookmarks'][0]['available'] is False
    assert restored.get(f"/api/segments/{gap['segment_id']}/frame").status_code == 404


def test_gap_mode_requires_visual_transcript_workspace(tmp_path):
    import pytest

    with pytest.raises(ValueError, match='requires transcript mode'):
        create_app(build_transcript_segments([]), mode='transcript', visual_cover_gaps=True,
                   database_path=tmp_path / 'library.db')


def test_changed_video_disables_gap_search_and_saved_evidence(tmp_path, monkeypatch):
    from video_rag.api import visual_timeline

    video = tmp_path / 'sample.mp4'; video.write_bytes(b'synthetic')
    subtitle = tmp_path / 'sample.srt'
    subtitle.write_text('1\n00:00:10,000 --> 00:00:12,000\nAttention.\n', encoding='utf-8')
    monkeypatch.setattr(visual_timeline, 'video_duration_seconds', lambda _path: 40.0)

    class Visual:
        def __init__(self, segments):
            self.gap = next(s for s in segments if not s.transcript)
        def search(self, _query, _top_k):
            return [SearchHit(self.gap.segment_id, .8, 'visual', 1, {'frame_time': 5.0})]

    client = create_app(build_transcript_segments(segments_from_subtitles(video, subtitle)), mode='transcript',
                        visual_builder=Visual, visual_cover_gaps=True,
                        database_path=tmp_path / 'library.db').test_client()
    result = client.post('/api/visual-search', json={'question': 'attention'}).json
    gap = result['evidence'][0]
    assert client.post('/api/bookmarks', json={'video_id': gap['video_id'], 'segment_id': gap['segment_id']}).status_code == 201
    video.write_bytes(b'changed synthetic data')
    assert client.post('/api/visual-search', json={'question': 'attention'}).status_code == 409
    assert client.get('/api/bookmarks').json['bookmarks'][0]['available'] is False
    assert client.get(f"/api/segments/{gap['segment_id']}/frame").status_code in (404, 409)


def test_invalid_visual_frame_timestamp_is_not_exposed(tmp_path):
    video = tmp_path / 'sample.mp4'; video.write_bytes(b'synthetic')
    subtitle = tmp_path / 'sample.srt'
    subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nAttention.\n', encoding='utf-8')
    segments = segments_from_subtitles(video, subtitle)

    class Visual:
        def search(self, _query, _top_k):
            return [SearchHit(segments[0].segment_id, .8, 'visual', 1, {'frame_time': float('nan')})]

    client = create_app(build_transcript_segments(segments), mode='transcript', visual_builder=lambda _: Visual(),
                        database_path=tmp_path / 'library.db').test_client()
    result = client.post('/api/visual-search', json={'question': 'attention'}).json
    assert result['evidence'][0]['frame_time'] is None


def test_gap_visual_search_respects_video_and_course_scopes_and_sampling_limit(tmp_path, monkeypatch):
    from video_rag.api import visual_timeline

    segments = []
    for index in range(2):
        video = tmp_path / f'video-{index}.mp4'; video.write_bytes(b'synthetic')
        subtitle = tmp_path / f'video-{index}.srt'
        subtitle.write_text('1\n00:00:10,000 --> 00:00:12,000\nAttention.\n', encoding='utf-8')
        segments.extend(segments_from_subtitles(video, subtitle))
    video_ids = [segments[0].video_id, segments[1].video_id]
    monkeypatch.setattr(visual_timeline, 'video_duration_seconds', lambda _path: 3000.0)
    built_scopes = []

    class Visual:
        def __init__(self, candidates):
            assert len(candidates) <= 100
            self.candidates = candidates
            built_scopes.append({s.video_id for s in candidates})
        def search(self, _query, top_k):
            return [SearchHit(s.segment_id, .8, 'visual', rank + 1)
                    for rank, s in enumerate(self.candidates[:top_k])]

    client = create_app(build_transcript_segments(segments), mode='transcript', visual_builder=Visual,
                        visual_cover_gaps=True, database_path=tmp_path / 'library.db').test_client()
    course = client.post('/api/courses', json={'name': 'Synthetic course'}).json['id']
    assert client.put(f'/api/videos/{video_ids[1]}/course', json={'course_id': course}).status_code == 200
    one = client.post('/api/visual-search', json={'question': 'slide', 'video_id': video_ids[0]}).json
    assert one['sampled'] and one['indexed_segments'] == 100
    assert {item['video_id'] for item in one['evidence']} == {video_ids[0]}
    course_result = client.post('/api/visual-search', json={'question': 'slide', 'course_id': course}).json
    assert {item['video_id'] for item in course_result['evidence']} == {video_ids[1]}
    both = client.post('/api/visual-search', json={'question': 'slide'}).json
    assert both['sampled'] and both['indexed_segments'] == 100
    assert built_scopes[-1] == set(video_ids)
    assert built_scopes[0] == {video_ids[0]} and built_scopes[1] == {video_ids[1]}


def test_visual_scope_limits_reject_before_model_build(tmp_path, monkeypatch):
    import importlib

    from video_rag.api import visual_timeline

    api_module = importlib.import_module('video_rag.api.app')
    segments = []
    for index in range(2):
        video = tmp_path / f'video-{index}.mp4'; video.write_bytes(b'synthetic')
        subtitle = tmp_path / f'video-{index}.srt'
        subtitle.write_text('1\n00:00:10,000 --> 00:00:12,000\nAttention.\n', encoding='utf-8')
        segments.extend(segments_from_subtitles(video, subtitle))
    calls = {'duration': 0, 'model': 0}
    def duration(_path):
        calls['duration'] += 1
        return 40.0
    def visual(_segments):
        calls['model'] += 1
        raise AssertionError('Visual model must not be built for an oversized scope')
    monkeypatch.setattr(visual_timeline, 'video_duration_seconds', duration)
    client = create_app(build_transcript_segments(segments), mode='transcript', visual_builder=visual,
                        visual_cover_gaps=True, database_path=tmp_path / 'library.db').test_client()
    monkeypatch.setattr(api_module, 'MAX_VISUAL_SCOPE_VIDEOS', 1)
    response = client.post('/api/visual-search', json={'question': 'slide'})
    assert response.status_code == 400 and 'narrow' in response.json['error']
    assert calls == {'duration': 0, 'model': 0}
    monkeypatch.setattr(api_module, 'MAX_VISUAL_SCOPE_VIDEOS', 100)
    monkeypatch.setattr(api_module, 'MAX_VISUAL_TIMELINE_SEGMENTS', 1)
    response = client.post('/api/visual-search', json={'question': 'slide', 'video_id': segments[0].video_id})
    assert response.status_code == 400 and 'narrow' in response.json['error']
    assert calls == {'duration': 1, 'model': 0}
