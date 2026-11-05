"""Compare local transcript and fused candidate timestamps on the same draft questions.

This evaluates temporal coverage only, not visual relevance or answer correctness.
Private videos, labels, model inputs, fingerprints and reports must stay local.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.diagnose_transcript import (
    comparison_by_question_type,
    diagnose,
    latency_summary,
    metrics,
    operating_metrics,
)
from scripts.prepare_temporal_questions import prepare, write_output
from video_rag.api.visual_sampling import sample_visual_segments
from video_rag.evaluation.private_output import private_output
from video_rag.evaluation.provenance import fingerprint, verify_sources
from video_rag.retrieval.fusion import reciprocal_rank_fusion
from video_rag.schemas import SearchHit
from video_rag.transcript_baseline import build_transcript_segments, segments_from_subtitles


class EvidenceOnly:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def ask(self, query):
        return self.pipeline.ask(query, generate_answer=False)


def load_questions(path, video_id, split, video_hash, *, allow_test=False):
    """Load JSON drafts or video-bound local labeler CSVs without changing them."""
    if path.suffix.lower() == '.csv':
        allowed = {'development', 'validation', 'test'} if allow_test else {'development', 'validation'}
        if not video_id or split not in allowed:
            raise ValueError('CSV requires --video-id and a development or validation split, or an explicitly frozen test')
        sidecar = path.with_suffix(path.suffix + '.source.json')
        binding = json.loads(sidecar.read_text(encoding='utf-8-sig'))
        if binding != {'video_sha256': video_hash, 'video_id': video_id, 'split': split}:
            raise ValueError('CSV source binding does not match this video and split')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            data = prepare(list(csv.DictReader(handle)), video_id, split)
        return data, 'annotation_video_matched_subtitles_not_bound', sidecar
    if video_id is not None or split is not None:
        raise ValueError('--video-id and --split are only used for CSV labels')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise TypeError('Question file must contain a JSON object')
    if data.get('split') not in (None, 'development', 'validation'):
        raise ValueError('Use development or validation; final test and unknown splits are reserved')
    return data, None, None


class FusedCandidates:
    """Mirror the workspace's equal-weight RRF of the two top-three rankings."""
    def __init__(self, transcript, visual, segments, *, min_visual_score=None):
        if min_visual_score is not None and (not math.isfinite(min_visual_score) or not -1 <= min_visual_score <= 1):
            raise ValueError('Visual similarity threshold must be finite and in [-1, 1]')
        self.transcript, self.visual = transcript, visual
        self.min_visual_score = min_visual_score
        self.lookup = {item.segment_id: item for item in segments}

    def ask(self, query):
        text = self.transcript.ask(query, generate_answer=False)
        text_hits = [SearchHit(item.segment.segment_id, 0.0, 'transcript', rank)
                     for rank, item in enumerate(text.evidence, 1)]
        visual = self.visual.search(query, 3)
        if self.min_visual_score is not None:
            visual = [hit for hit in visual if hit.score >= self.min_visual_score]
        visual = [SearchHit(hit.segment_id, hit.score, 'visual', rank)
                  for rank, hit in enumerate(visual, 1)]
        fused = reciprocal_rank_fusion([text_hits, visual], k=60, top_k=3)
        evidence = [{'start_time': self.lookup[hit.segment_id].start_time,
                     'end_time': self.lookup[hit.segment_id].end_time,
                     'sources': hit.source.split('+')} for hit in fused]

        class Result:
            def to_dict(self):
                return {'evidence': evidence, 'abstained': not evidence}

        return Result()


def compare(questions, transcript, fused, min_coverage=0.5):
    if not isinstance(questions, list) or not questions:
        raise ValueError('A nonempty question list is required')
    if not math.isfinite(min_coverage) or not 0 < min_coverage <= 1:
        raise ValueError('Coverage threshold must be in (0, 1]')
    left, right = diagnose(transcript, questions), diagnose(fused, questions)
    pairs = {'both_hit': 0, 'transcript_only_hit': 0, 'fused_only_hit': 0, 'neither_hit': 0}
    negatives = {'both_abstain': 0, 'transcript_only_abstains': 0,
                 'fused_only_abstains': 0, 'neither_abstains': 0}
    rows = []
    for question, a, b in zip(questions, left, right, strict=True):
        if a['expected_answerable']:
            ah = not a['abstained'] and a['first_hit_reference_coverage'] >= min_coverage
            bh = not b['abstained'] and b['first_hit_reference_coverage'] >= min_coverage
            key = 'both_hit' if ah and bh else 'transcript_only_hit' if ah else 'fused_only_hit' if bh else 'neither_hit'
            pairs[key] += 1
        else:
            aa, ba = a['abstained'], b['abstained']
            key = 'both_abstain' if aa and ba else 'transcript_only_abstains' if aa else 'fused_only_abstains' if ba else 'neither_abstains'
            negatives[key] += 1
        reference = ([float(question['evidence_start_seconds']), float(question['evidence_end_seconds'])]
                     if a['expected_answerable'] else None)
        rows.append({'question_id': question.get('question_id'), 'question': a['question'],
                     'reference_range': reference, 'transcript': a, 'fused': b})

    def summarize(items):
        operating = operating_metrics(items, min_coverage)
        return {'metrics': metrics(items, min_coverage), 'operating': operating,
                'temporal_recall_at_3': {'hits': operating['any_returned_temporal_hits'],
                                         'answerable_questions': operating['answerable_questions'],
                                         'rate': operating['any_returned_temporal_hit_rate'],
                                         'min_reference_coverage': min_coverage},
                'query_latency': latency_summary(items), 'by_question_type': comparison_by_question_type(items, min_coverage)}

    return {'transcript': summarize(left), 'fused_experimental': summarize(right),
            'paired_answerable_top1': pairs, 'paired_unanswerable': negatives, 'results': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'subtitles', 'questions', 'visual-model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--semantic-model', type=Path, help='Use local MiniLM; default is keyword retrieval')
    parser.add_argument('--video-id', help='Required for a local labeler CSV')
    parser.add_argument('--split', choices=('development', 'validation', 'test'), help='Required for a local labeler CSV')
    parser.add_argument('--frozen-test-manifest', type=Path, help='Required only for a declared frozen test CSV')
    parser.add_argument('--semantic-threshold', type=float, default=0.4)
    parser.add_argument('--visual-min-score', type=float,
                        help='Optional uncalibrated visual cutoff for evaluation only; workspace applies none')
    parser.add_argument('--min-coverage', type=float, default=0.5)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    try:
        from video_rag.retrieval.onnx_clip import ClipFrameRetriever, OnnxClipEncoder

        target = private_output(args.output)
        if target.suffix.lower() != '.json' or target.exists():
            raise ValueError('Choose a new local JSON output')
        if args.visual_min_score is not None and (not math.isfinite(args.visual_min_score) or
                                                   not -1 <= args.visual_min_score <= 1):
            raise ValueError('Visual similarity threshold must be finite and in [-1, 1]')
        is_test = args.split == 'test'
        if is_test != (args.frozen_test_manifest is not None):
            raise ValueError('A frozen-test manifest is required only when --split test is selected')
        paths = {'video': args.video, 'subtitles': args.subtitles, 'questions': args.questions,
                 'visual_tokenizer': args.visual_model / 'tokenizer.json',
                 'visual_model': args.visual_model / 'onnx' / 'model_quantized.onnx'}
        if args.semantic_model:
            paths.update({'semantic_tokenizer': args.semantic_model / 'tokenizer.json',
                          'semantic_model': args.semantic_model / 'onnx' / 'model_quantized.onnx'})
        if args.questions.suffix.lower() == '.csv':
            paths['annotation_binding'] = args.questions.with_suffix(args.questions.suffix + '.source.json')
        before = {name: fingerprint(path) for name, path in paths.items()}
        if is_test:
            from video_rag.evaluation.frozen_temporal import config, input_paths, read_and_verify

            manifest_path = private_output(args.frozen_test_manifest)
            settings = config(args.video_id, args.semantic_model, args.semantic_threshold,
                              args.visual_min_score, args.min_coverage)
            expected_paths = input_paths(args.video, args.subtitles, args.questions, args.visual_model, args.semantic_model)
            read_and_verify(manifest_path, expected_paths, settings)
            before['frozen_manifest'] = fingerprint(manifest_path)
            paths['frozen_manifest'] = manifest_path
        data, csv_binding, _ = load_questions(args.questions, args.video_id, args.split, before['video'], allow_test=is_test)
        binding = csv_binding or verify_sources(data, args.video, args.subtitles)
        segments = segments_from_subtitles(args.video, args.subtitles)
        if args.semantic_model:
            from video_rag.retrieval.onnx_semantic import (
                OnnxSentenceEncoder,
                build_semantic_pipeline,
            )

            transcript = build_semantic_pipeline(segments, OnnxSentenceEncoder(args.semantic_model), args.semantic_threshold)
        else:
            transcript = build_transcript_segments(segments)
        selected = sample_visual_segments(tuple(segments), limit=100)
        visual = ClipFrameRetriever(OnnxClipEncoder(args.visual_model))
        started = perf_counter()
        visual.build(selected)
        build_seconds = perf_counter() - started
        result = compare(data['questions'], EvidenceOnly(transcript),
                         FusedCandidates(transcript, visual, segments, min_visual_score=args.visual_min_score),
                         args.min_coverage)
        if csv_binding is None:
            verify_sources(data, args.video, args.subtitles)
        if is_test:
            read_and_verify(manifest_path, expected_paths, settings)
        if any(fingerprint(path) != before[name] for name, path in paths.items()):
            raise ValueError('An input or model file changed during evaluation')
        if args.aggregate_only:
            result.pop('results')
        report = {'status': data.get('status', 'not recorded'), 'split': data.get('split'), 'binding': binding,
                  'test_declaration': 'frozen_test_declared_not_independently_verified' if is_test else None,
                  'input_sha256': before, 'transcript_retrieval': 'semantic_experimental' if args.semantic_model else 'keyword',
                  'semantic_threshold': args.semantic_threshold if args.semantic_model else None,
                  'visual_min_score_experimental': args.visual_min_score,
                  'min_coverage': args.min_coverage, 'rrf_k': 60, 'top_k': 3,
                  'source_segments': len(segments), 'sampled_visual_segments': len(selected),
                  'visual_index_build_seconds': build_seconds, 'aggregate_only': args.aggregate_only,
                  'warning': 'Candidate timestamp coverage only; no visual relevance or answer correctness is assessed. '
                             'The workspace has no visual rejection threshold. Any experimental cutoff is uncalibrated. '
                             'Keep all course-derived reports and hashes local.',
                  'comparison': result}
        write_output(target, report, (ROOT / 'artifacts', ROOT / 'local_data'))
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError, csv.Error) as exc:
        parser.exit(2, f'Comparison failed: {exc}\n')
    print('Saved a local provisional comparison. Keep course-derived results and fingerprints private.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
