from scripts.review_subtitles import align_alternative, create_review_app
from video_rag.transcript_baseline import parse_subtitles


def test_alternative_alignment_is_a_local_comparison_not_a_reference(tmp_path):
    video = tmp_path / 'video.mp4'; video.write_bytes(b'video')
    subtitle = tmp_path / 'source.srt'; subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nA red car\n\n2\n00:00:04,000 --> 00:00:06,000\nThe car moves\n', encoding='utf-8')
    alternative = tmp_path / 'alternative.txt'; alternative.write_text('A blue car. The car moves.', encoding='utf-8')
    output = tmp_path / 'private' / 'out'
    client = create_review_app(video, subtitle, output, alternative=alternative, allowed_roots=(tmp_path / 'private',)).test_client()
    cues = client.get('/api/cues').json['cues']
    assert cues[0]['alternative'].lower() == 'a blue car' and cues[0]['differs']
    assert cues[1]['alternative'].lower() == 'the car moves' and not cues[1]['differs']
    alternative.write_text('changed alternative', encoding='utf-8')
    assert client.get('/api/cues').status_code == 409
    assert client.post('/api/export', json={'edits': {'0': 'A blue car'}}).status_code == 409


def test_alignment_requires_usable_words():
    import pytest
    cues = parse_subtitles('1\n00:00:01,000 --> 00:00:03,000\nhello world\n')
    with pytest.raises(ValueError, match='usable words'):
        align_alternative(cues, '...')
