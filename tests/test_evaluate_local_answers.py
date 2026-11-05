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


def test_rejects_test_split_and_invalid_questions():
    with pytest.raises(ValueError, match='reserved'):
        prepare_review(StubPipeline(), {'split': 'test', 'questions': [{'question': 'Why?'}]})
    for data in ({'questions': []}, {'questions': [{'question': ''}]},
                 {'questions': [{'question': 'Why?', 'expected_answerable': 'false'}]}):
        with pytest.raises(ValueError):
            prepare_review(StubPipeline(), data)
