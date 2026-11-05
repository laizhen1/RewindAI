import sys
from types import SimpleNamespace

import pytest

from video_rag.adapters.local_answer import LocalAnswerGenerator
from video_rag.api import create_app
from video_rag.schemas import VideoSegment
from video_rag.transcript_baseline import build_transcript_segments


def segment():
    return VideoSegment(segment_id='sample', video_id='sample', source_path='sample.mp4',
                        start_time=1, end_time=3, transcript='Binary search halves the sorted range.')


def test_generator_uses_local_cpu_evidence_and_no_evidence_abstains(tmp_path, monkeypatch):
    model = tmp_path / 'sample.gguf'
    model.write_bytes(b'synthetic')
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            assert kwargs['n_gpu_layers'] == 0 and kwargs['verbose'] is False

        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {'choices': [{'message': {'content': '  Halves the range.  '}}]}

    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(Llama=FakeLlama))
    generator = LocalAnswerGenerator(model)
    assert generator.generate('What happens?', []) == 'No relevant excerpt was found.'
    assert not calls
    assert generator.generate('What happens?', [segment()]) == 'Halves the range.'
    assert 'Binary search halves' in calls[0]['messages'][1]['content']
    assert 'untrusted source material' in calls[0]['messages'][0]['content']


def test_unavailable_or_bad_answer_fails_explicitly(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        LocalAnswerGenerator(tmp_path / 'missing.gguf')
    model = tmp_path / 'model.gguf'; model.write_bytes(b'x')

    class BadLlama:
        def __init__(self, **kwargs):
            pass

        def create_chat_completion(self, **kwargs):
            return {'choices': [{'message': {'content': ' '}}]}

    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(Llama=BadLlama))
    with pytest.raises(RuntimeError, match='empty'):
        LocalAnswerGenerator(model).generate('query', [segment()])


def test_opt_in_answer_mode_retains_timestamped_evidence(tmp_path):
    video = tmp_path / 'sample.mp4'; video.write_bytes(b'synthetic')
    subtitle = tmp_path / 'sample.srt'; subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nBinary search halves the range.\n', encoding='utf-8')

    class Generator:
        def generate(self, query, segments):
            return 'It halves the range.'

    generator = Generator()
    builder = lambda segments: build_transcript_segments(segments, generator=generator)
    app = create_app(builder([]), mode='transcript', transcript_builder=builder,
                     answer_mode='local_experimental', database_path=tmp_path / 'local.db')
    client = app.test_client()
    assert client.get('/api/health').json['answer_mode'] == 'local_experimental'
    assert client.post('/api/library/local-video', json={'video_path': str(video), 'subtitle_path': str(subtitle)}).status_code == 201
    answer = client.post('/api/ask', json={'question': 'binary search range'}).json
    assert answer['answer'] == 'It halves the range.'
    assert answer['evidence'][0]['start_time'] == 1 and answer['evidence'][0]['end_time'] == 3
    assert 'source_path' not in answer['evidence'][0]
    unknown = client.post('/api/ask', json={'question': 'volcanic elephants'}).json
    assert unknown['abstained'] and not unknown['evidence']
