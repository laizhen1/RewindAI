"""Generate local answer-review drafts; do not infer correctness from model output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from video_rag.adapters.local_answer import LocalAnswerGenerationError, LocalAnswerGenerator
from video_rag.evaluation.private_output import private_output
from video_rag.evaluation.provenance import fingerprint, verify_sources
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


def prepare_review(pipeline, data: dict, *, allow_frozen_test: bool = False, model_input_builder=None) -> list[dict]:
    """Preserve model output for a human reviewer without assigning scores."""
    if not isinstance(data, dict) or (data.get('split') == 'test' and not allow_frozen_test):
        raise ValueError('Final test questions are reserved; use development or validation')
    questions = data.get('questions')
    if not isinstance(questions, list) or not questions or len(questions) > 1000:
        raise ValueError('Provide 1 to 1000 questions')
    rows = []
    for item in questions:
        if not isinstance(item, dict) or not isinstance(item.get('question'), str) or not item['question'].strip():
            raise ValueError('Each question needs text')
        answerable = item.get('expected_answerable', True)
        if type(answerable) is not bool:
            raise ValueError('expected_answerable must be a boolean')
        question_type = item.get('question_type')
        if question_type is not None and (not isinstance(question_type, str) or
                                          question_type not in {'audio', 'visual', 'ocr', 'multimodal', 'unknown_route'}):
            raise ValueError('Invalid question type')
        started = perf_counter()
        generation_error = None
        try:
            answer = pipeline.ask(item['question'])
        except LocalAnswerGenerationError as exc:
            generation_error = str(exc)
            # Preserve retrievable context, but do not misrepresent a retry as
            # the evidence actually supplied to the failed generation attempt.
            answer = pipeline.ask(item['question'], generate_answer=False)
        model_input = (model_input_builder(item['question'].strip(), [hit.segment for hit in answer.evidence])
                       if model_input_builder is not None and not answer.abstained and generation_error is None else None)
        evidence = [{'start': hit.segment.start_time, 'end': hit.segment.end_time,
                     'text': hit.segment.transcript} for hit in answer.evidence]
        if model_input is not None:
            for index, entry in enumerate(evidence):
                entry['model_excerpt'] = (model_input['excerpts'][index]
                                          if index < len(model_input['excerpts']) else None)
        row = {
            'question_id': item.get('question_id'), 'question': item['question'],
            'expected_answerable': answerable, 'expected_answer': item.get('expected_answer'),
            'annotation_source': item.get('annotation_source', 'not_recorded'),
            'reference_interval': [item.get('evidence_start_seconds'), item.get('evidence_end_seconds')],
            'answer': answer.answer, 'retrieval_abstained': answer.abstained,
            'evidence': evidence,
            'latency_ms': round((perf_counter() - started) * 1000, 1),
            'human_answer_verdict': None, 'human_evidence_verdict': None,
        }
        if generation_error is not None:
            row['generation_error'] = generation_error
        if model_input is not None:
            row['model_question'] = model_input['question']
        if question_type is not None:
            row['question_type'] = question_type
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('video', 'subtitles', 'questions', 'model', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    try:
        target = private_output(args.output)
        if target.suffix.lower() != '.json' or target.exists():
            raise ValueError('Choose a new JSON output under artifacts/ or local_data/')
        sources = {'video': args.video, 'subtitles': args.subtitles,
                   'questions': args.questions, 'model': args.model}
        before = {key: fingerprint(path) for key, path in sources.items()}
        data = json.loads(args.questions.read_text(encoding='utf-8-sig'))
        binding = verify_sources(data, args.video, args.subtitles)
        if data.get('split') == 'test':
            raise ValueError('Final test questions are reserved')
        generator = LocalAnswerGenerator(args.model)
        segments = segments_from_subtitles(args.video, args.subtitles)
        pipeline = build_transcript_segments(segments, generator=generator)
        rows = prepare_review(pipeline, data, model_input_builder=generator.model_input)
        if any(fingerprint(path) != before[key] for key, path in sources.items()):
            raise ValueError('An input changed during evaluation; discard this run')
        report = {'status': 'unreviewed_answer_drafts', 'binding': binding,
                  'input_sha256': before, 'generation_config': generator.metadata(),
                  'split': data.get('split'), 'rows': rows,
                  'warning': 'Private draft output; judge answers and evidence against the video. No scores are assigned.'}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
            out.write('\n')
        print('Local review draft saved; no answer accuracy has been calculated.')
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(2, f'Could not create local review draft: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
