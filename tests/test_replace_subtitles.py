import sqlite3

import pytest

from video_rag.api.app import create_app
from video_rag.transcript_baseline import build_transcript_segments


@pytest.fixture
def workspace(tmp_path):
    video = tmp_path / 'sample.mp4'
    video.write_bytes(b'synthetic video')
    subtitle = tmp_path / 'original.srt'
    subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nOriginal words\n', encoding='utf-8')
    db = tmp_path / 'library.sqlite3'
    client = create_app(build_transcript_segments([]), mode='transcript', database_path=db).test_client()
    result = client.post('/api/library/local-video', json={'video_path': str(video), 'subtitle_path': str(subtitle)})
    assert result.status_code == 201
    video_id = result.json['video_id']
    evidence = client.post('/api/ask', json={'question': 'original words'}).json['evidence'][0]
    bookmark = client.post('/api/bookmarks', json={'video_id': video_id, 'segment_id': evidence['segment_id'], 'note': 'keep'})
    assert bookmark.status_code == 201
    return client, video, subtitle, db, video_id


def test_corrected_subtitles_replace_search_and_preserve_exact_bookmarks(workspace, tmp_path):
    client, _video, subtitle, db, video_id = workspace
    corrected = tmp_path / 'corrected.srt'
    corrected.write_text('1\n00:00:01,000 --> 00:00:03,000\nCorrected phrase\n', encoding='utf-8')
    result = client.put(f'/api/library/videos/{video_id}/subtitles', json={'subtitle_path': str(corrected)})
    assert result.status_code == 200
    bookmark = client.get('/api/bookmarks').json['bookmarks'][0]
    assert bookmark['available'] and bookmark['note'] == 'keep'
    assert client.post('/api/ask', json={'question': 'corrected phrase'}).json['evidence']
    reopened = create_app(build_transcript_segments([]), mode='transcript', database_path=db).test_client()
    assert reopened.get('/api/bookmarks').json['bookmarks'][0]['available']
    assert reopened.post('/api/ask', json={'question': 'corrected phrase'}).json['evidence']
    assert subtitle.read_text(encoding='utf-8').endswith('Original words\n')


def test_shifted_ranges_keep_note_but_not_claim_playable(workspace, tmp_path):
    client, _video, _subtitle, _db, video_id = workspace
    changed = tmp_path / 'changed.srt'
    changed.write_text('1\n00:00:02,000 --> 00:00:04,000\nChanged words\n', encoding='utf-8')
    assert client.put(f'/api/library/videos/{video_id}/subtitles', json={'subtitle_path': str(changed)}).status_code == 200
    bookmark = client.get('/api/bookmarks').json['bookmarks'][0]
    assert not bookmark['available'] and bookmark['note'] == 'keep'


def test_failed_replacement_leaves_existing_index(workspace, tmp_path):
    client, video, subtitle, db, video_id = workspace
    assert client.put(f'/api/library/videos/{video_id}/subtitles', json={'subtitle_path': str(tmp_path / 'missing.srt')}).status_code == 400
    assert client.post('/api/ask', json={'question': 'original words'}).json['evidence']
    video.write_bytes(b'changed synthetic video')
    assert client.put(f'/api/library/videos/{video_id}/subtitles', json={'subtitle_path': str(subtitle)}).status_code == 409
    reopened = create_app(build_transcript_segments([]), mode='transcript', database_path=db).test_client()
    assert reopened.get('/api/library').json['unavailable_saved_videos'] == 1


def test_concurrent_saved_index_change_does_not_overwrite(workspace, tmp_path):
    client, _video, subtitle, db, video_id = workspace
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE local_transcript_videos SET segments_json = ? WHERE video_id = ?", ("[]", video_id))
    result = client.put(f'/api/library/videos/{video_id}/subtitles', json={'subtitle_path': str(subtitle)})
    assert result.status_code == 409
    assert client.post('/api/ask', json={'question': 'original words'}).json['evidence']
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT segments_json FROM local_transcript_videos WHERE video_id = ?", (video_id,)).fetchone()[0] == "[]"


def test_database_failure_rolls_back_index_and_bookmarks(workspace, tmp_path):
    client, _video, _subtitle, db, video_id = workspace
    corrected = tmp_path / 'corrected.srt'
    corrected.write_text('1\n00:00:01,000 --> 00:00:03,000\nCorrected words\n', encoding='utf-8')
    with sqlite3.connect(db) as connection:
        original = connection.execute("SELECT segments_json FROM local_transcript_videos WHERE video_id = ?", (video_id,)).fetchone()[0]
        connection.execute("CREATE TRIGGER reject_remap BEFORE UPDATE ON bookmarks BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    result = client.put(f'/api/library/videos/{video_id}/subtitles', json={'subtitle_path': str(corrected)})
    assert result.status_code == 500
    assert client.get('/api/bookmarks').json['bookmarks'][0]['available']
    assert client.post('/api/ask', json={'question': 'original words'}).json['evidence']
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT segments_json FROM local_transcript_videos WHERE video_id = ?", (video_id,)).fetchone()[0] == original


def test_private_workspace_responses_are_not_cached(workspace):
    client, _video, _subtitle, _db, video_id = workspace
    for path in ('/', '/api/library', '/api/bookmarks', '/api/feedback', f'/api/videos/{video_id}'):
        assert client.get(path).headers['Cache-Control'] == 'no-store'
    assert client.post('/api/ask', json={'question': 'original words'}).headers['Cache-Control'] == 'no-store'
