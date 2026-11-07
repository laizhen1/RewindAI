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
    assert generator.metadata()['max_question_characters'] == 1000


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


def test_exact_model_refusal_is_distinct_from_retrieval_abstention(tmp_path):
    video = tmp_path / 'sample.mp4'; video.write_bytes(b'synthetic')
    subtitle = tmp_path / 'sample.srt'
    subtitle.write_text('1\n00:00:01,000 --> 00:00:03,000\nBinary search halves the range.\n', encoding='utf-8')

    class Generator:
        def generate(self, query, segments):
            return '  Insufficient evidence in the provided excerpts.  '

    generator = Generator()
    builder = lambda segments: build_transcript_segments(segments, generator=generator)
    client = create_app(builder([]), mode='transcript', transcript_builder=builder,
                        answer_mode='local_experimental', database_path=tmp_path / 'local.db').test_client()
    assert client.post('/api/library/local-video', json={'video_path': str(video), 'subtitle_path': str(subtitle)}).status_code == 201
    result = client.post('/api/ask', json={'question': 'binary search range'}).json
    assert result['evidence'] and not result['abstained'] and result['exact_model_refusal']
    missing = client.post('/api/ask', json={'question': 'volcanic elephants'}).json
    assert missing['abstained'] and not missing['exact_model_refusal']


def test_local_answer_rejects_questions_that_would_be_silently_truncated(tmp_path):
    class Generator:
        def generate(self, query, segments):
            return query

    question = 'a' * 1000
    evidence = VideoSegment(segment_id='sample', video_id='sample', source_path='sample.mp4',
                            start_time=1, end_time=3, transcript=question)
    app = create_app(build_transcript_segments([evidence], generator=Generator()), mode='unknown',
                     answer_mode='local_experimental', database_path=tmp_path / 'local.db')
    client = app.test_client()
    response = client.post('/api/ask', json={'question': 'a' * 1001})
    assert response.status_code == 400
    assert 'at most 1000 characters' in response.json['error']
    response = client.post('/api/ask', json={'question': question})
    assert response.status_code == 200
    assert response.json['answer'] == question
    assert response.json['evidence'][0]['model_excerpt'] == question[:500]
    assert response.json['evidence'][0]['transcript'] == question
