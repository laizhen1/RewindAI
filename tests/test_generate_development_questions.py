from threading import Lock
from types import SimpleNamespace

import pytest

from scripts.generate_development_questions import generate_suggestions, parse_suggestion


def test_suggestions_remain_unverified_and_use_source_times():
    class Model:
        def create_chat_completion(self, **kwargs):
            return {'choices': [{'message': {'content': '{"question":"What color?","answer":"Red"}'}}]}
    generator = SimpleNamespace(_lock=Lock(), _model=Model())
    segment = SimpleNamespace(transcript='A red car', start_time=1, end_time=3)
    result = generate_suggestions([segment], generator, 1)
    assert result[0]['status'] == 'model_generated_unverified'
    assert result[0]['start_seconds'] == 1 and result[0]['end_seconds'] == 3


def test_bad_model_output_is_not_accepted():
    for value in ('not json', '{}', '{"question":"?","answer":""}'):
        with pytest.raises(ValueError):
            parse_suggestion(value)
    assert parse_suggestion('```json\n{"question":"Q?","answer":"A"}\n```')['answer'] == 'A'
