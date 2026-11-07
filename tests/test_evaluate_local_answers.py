import pytest

from scripts.evaluate_local_answers import prepare_review
from video_rag.schemas import Answer, Evidence, VideoSegment


class StubPipeline:
    def ask(self, question):
        segment = VideoSegment(segment_id='s', video_id='v', source_path='private.mp4',
                               start_time=1, end_time=2, transcript='Evidence text')
        return Answer(answer='Generated answer', evidence=(Evidence(segment, 0.5, 0.6),), abstained=False)


def test_review_does_not_score_or_expose_source_path():
    result = prepare_review(StubPipeline(), {'questions': [{'question': 'Why?', 'expected_answerable': True}]})
    assert result[0]['human_answer_verdict'] is None
    assert result[0]['human_evidence_verdict'] is None
    assert result[0]['evidence'] == [{'start': 1, 'end': 2, 'text': 'Evidence text'}]
    assert result[0]['retrieval_abstained'] is False


def test_review_preserves_question_type_without_inventing_one():
    result = prepare_review(StubPipeline(), {'questions': [{'question': 'Why?', 'question_type': 'audio'},
                                                            {'question': 'How?'}]})
    assert result[0]['question_type'] == 'audio'
    assert 'question_type' not in result[1]
    with pytest.raises(ValueError, match='question type'):
        prepare_review(StubPipeline(), {'questions': [{'question': 'Why?', 'question_type': 'invalid'}]})


def test_generation_failure_keeps_other_questions_without_inventing_model_input():
    from video_rag.adapters.local_answer import LocalAnswerGenerationError

    class Pipeline:
        def ask(self, question, *, generate_answer=True):
            if question == 'Fails?' and generate_answer:
                raise LocalAnswerGenerationError('Model context limit exceeded')
            segment = VideoSegment('s', 'v', 'video.mp4', 0, 3, transcript='Candidate')
            return Answer(answer='Supported answer' if generate_answer else '',
                          evidence=(Evidence(segment, 0.5, 0.6),))

    rows = prepare_review(Pipeline(), {'questions': [{'question': 'Fails?'}, {'question': 'Works?'}]},
                          model_input_builder=lambda q, s: {'question': q, 'excerpts': [s[0].transcript]})
    assert len(rows) == 2 and rows[0]['generation_error'] == 'Model context limit exceeded'
    assert rows[0]['answer'] == '' and rows[0]['evidence'][0]['text'] == 'Candidate'
    assert 'model_question' not in rows[0] and 'model_excerpt' not in rows[0]['evidence'][0]
    assert rows[1]['answer'] == 'Supported answer' and rows[1]['model_question'] == 'Works?'


def test_rejects_test_split_and_invalid_questions():
    with pytest.raises(ValueError, match='reserved'):
        prepare_review(StubPipeline(), {'split': 'test', 'questions': [{'question': 'Why?'}]})
    for data in ({'questions': []}, {'questions': [{'question': ''}]},
                 {'questions': [{'question': 'Why?', 'expected_answerable': 'false'}]}):
        with pytest.raises(ValueError):
            prepare_review(StubPipeline(), data)


def test_review_records_exact_local_model_inputs_and_omitted_context(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from video_rag.adapters.local_answer import LocalAnswerGenerator

    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            pass

        def create_chat_completion(self, **kwargs):
            calls.append(kwargs['messages'][1]['content'])
            return {'choices': [{'message': {'content': 'Answer'}}]}

    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(Llama=FakeLlama))
    model = tmp_path / 'model.gguf'
    model.write_bytes(b'synthetic')
    generator = LocalAnswerGenerator(model)
    segments = [VideoSegment(str(i), 'v', 'v.mp4', i, i + 1, transcript=str(i) + 'x' * 600)
                for i in range(4)]

    class Pipeline:
        def ask(self, question):
            result = generator.generate(question.strip(), segments)
            return Answer(answer=result, evidence=tuple(Evidence(s, 0.5, 0.6) for s in segments), abstained=False)

    question = '  ' + 'q' * 1100 + '  '
    row = prepare_review(Pipeline(), {'questions': [{'question': question}]},
                         model_input_builder=generator.model_input)[0]
    assert row['model_question'] == 'q' * 1000
    assert row['evidence'][0]['model_excerpt'] == segments[0].transcript[:500]
    assert row['evidence'][3]['model_excerpt'] is None
    assert f"Question: {row['model_question']}" in calls[0]
    assert f"[1] {row['evidence'][0]['model_excerpt']}" in calls[0]
    assert '[4]' not in calls[0]
