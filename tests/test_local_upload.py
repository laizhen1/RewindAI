import io
from pathlib import Path
from types import SimpleNamespace

from video_rag.api import create_app
from video_rag.transcript_baseline import build_transcript_segments


def client(tmp_path):
    app = create_app(build_transcript_segments([]), mode='transcript',
                     database_path=tmp_path / 'library.sqlite3', upload_dir=tmp_path / 'uploads')
    return app.test_client()


def payload(**changes):
    data = {'permission': 'yes', 'local_storage': 'yes',
            'video': (io.BytesIO(b'synthetic video'), 'sample.mp4'),
            'subtitle': (io.BytesIO(b'1\n00:00:01,000 --> 00:00:03,000\nSynthetic lecture.\n'), 'sample.srt')}
    data.update(changes)
    return data


def test_upload_requires_permission_and_local_storage_acknowledgment(tmp_path):
    c = client(tmp_path)
    for field in ('permission', 'local_storage'):
        response = c.post('/api/library/upload', data=payload(**{field: 'no'}))
        assert response.status_code == 400
        assert not (tmp_path / 'uploads').exists()
    assert c.post('/api/library/upload', data=payload(), headers={'Origin': 'https://example.org'}).status_code == 403


def test_upload_saves_and_indexes_private_local_copy(tmp_path):
    c = client(tmp_path)
    response = c.post('/api/library/upload', data=payload())
    assert response.status_code == 201 and response.json['state'] == 'ready'
    assert 'video_path' not in response.json
    library = c.get('/api/library').json
    assert len(library['videos']) == 1 and library['videos'][0]['title'] == 'sample.mp4'
    files = list((tmp_path / 'uploads').glob('*/*'))
    assert {p.name for p in files} == {'sample.mp4', 'captions.srt'}
    result = c.post('/api/ask', json={'question': 'synthetic lecture'}).json
    assert result['evidence'][0]['start_time'] == 1


def test_failed_upload_cleans_only_its_new_copy(tmp_path):
    c = client(tmp_path)
    data = payload(subtitle=(io.BytesIO(b'invalid subtitle'), 'sample.srt'))
    assert c.post('/api/library/upload', data=data).status_code == 400
    assert list((tmp_path / 'uploads').iterdir()) == []
    assert c.get('/api/library').json['videos'] == []
    data = payload(subtitle=(io.BytesIO(b'bad'), 'captions.txt'))
    assert c.post('/api/library/upload', data=data).status_code == 400


def test_changed_upload_is_rejected_and_cleaned_after_indexing(tmp_path):
    def builder(segments):
        Path(segments[0].source_path).write_bytes(b'video changed while indexing')
        return build_transcript_segments(segments)

    app = create_app(build_transcript_segments([]), mode='transcript', transcript_builder=builder,
                     database_path=tmp_path / 'library.sqlite3', upload_dir=tmp_path / 'uploads')
    c = app.test_client()
    response = c.post('/api/library/upload', data=payload())
    assert response.status_code == 409
    assert 'changed while importing' in response.json['error']
    assert list((tmp_path / 'uploads').iterdir()) == []
    assert c.get('/api/library').json['videos'] == []


def test_upload_without_subtitles_starts_local_transcription(tmp_path, monkeypatch):
    from video_rag.api.transcription_job import LocalTranscriptionJob

    def start(job, video, language):
        assert video.is_file() and language == 'en'
        job._state = {'state': 'transcribing', 'message': 'synthetic'}
        return True

    monkeypatch.setattr(LocalTranscriptionJob, 'start', start)
    c = client(tmp_path)
    data = payload(language='en')
    del data['subtitle']
    response = c.post('/api/library/upload', data=data)
    assert response.status_code == 202 and response.json['state'] == 'transcribing'
    assert len(list((tmp_path / 'uploads').glob('*/*.mp4'))) == 1


def test_failed_upload_never_removes_preexisting_directory(tmp_path, monkeypatch):
    from video_rag.api import app as app_module

    c = client(tmp_path)
    existing = tmp_path / 'uploads' / 'already-there'
    existing.mkdir(parents=True)
    monkeypatch.setattr(app_module.uuid, 'uuid4', lambda: SimpleNamespace(hex='already-there'))
    response = c.post('/api/library/upload', data=payload())
    assert response.status_code == 400
    assert existing.is_dir()
    assert list(existing.iterdir()) == []
