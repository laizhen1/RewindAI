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
