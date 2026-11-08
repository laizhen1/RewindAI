"""Compare local text, frame, RRF and optional reranking using declared relevance.

Development/validation or a declared frozen test. This is a top-five offline ranking experiment,
not the workspace's top-three thresholded answer pipeline. All inputs stay local.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.evaluate_retrieval import metrics_with_breakdown
from scripts.prepare_temporal_questions import prepare
from scripts.review_retrieval_relevance import segment_fingerprint
from video_rag.api.visual_sampling import sample_visual_segments
from video_rag.evaluation.private_output import private_output
from video_rag.evaluation.provenance import fingerprint
from video_rag.evaluation.retrieval import hit_at_k, recall_at_k, reciprocal_rank
from video_rag.retrieval.fusion import reciprocal_rank_fusion
from video_rag.retrieval.onnx_semantic import SemanticRetriever
from video_rag.retrieval.semantic_models import load_semantic_encoder, semantic_model_files
from video_rag.schemas import SearchHit
from video_rag.transcript_baseline import segments_from_subtitles

CODE_FILES = (
    'scripts/evaluate_local_segment_retrieval.py', 'scripts/freeze_local_segment_test.py',
    'scripts/review_retrieval_relevance.py', 'src/video_rag/api/templates/relevance_review.html',
    'scripts/prepare_temporal_questions.py', 'scripts/evaluate_retrieval.py',
    'src/video_rag/evaluation/retrieval.py', 'src/video_rag/evaluation/provenance.py',
    'src/video_rag/retrieval/semantic_models.py', 'src/video_rag/retrieval/qwen_semantic.py',
    'src/video_rag/retrieval/onnx_semantic.py', 'src/video_rag/retrieval/onnx_clip.py',
    'src/video_rag/retrieval/fusion.py', 'src/video_rag/api/visual_sampling.py',
    'src/video_rag/api/frames.py', 'src/video_rag/transcript_baseline.py', 'src/video_rag/schemas.py',
    'src/video_rag/adapters/qwen.py',
)


def dependency_versions() -> dict:
    versions = {}
    for name in ('llama-cpp-python', 'numpy', 'onnxruntime', 'tokenizers', 'av', 'torch', 'transformers', 'accelerate'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def path_versions(paths: dict) -> dict:
    versions = {}
    for name, path in paths.items():
        stat = Path(path).stat()
        versions[name] = (stat.st_size, stat.st_mtime_ns)
    return versions


def local_reranker_files(directory: Path) -> dict:
    if not directory.is_dir():
        raise ValueError('Provide an existing local reranker model directory')
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    if not files or len(files) > 512:
        raise ValueError('The reranker directory must contain 1 to 512 model files')
    return {f'reranker:{path.relative_to(directory).as_posix()}': path for path in files}


def validate_gold(gold: dict, *, hashes: dict, video_id: str, split: str, questions: list, segments: list) -> dict:
    """Require declared complete labels; do not infer relevance from time overlap."""
    if (not isinstance(gold, dict) or gold.get('schema_version') != 1
            or gold.get('status') != 'relevance_draft_not_independently_verified'
            or gold.get('source_sha256') != hashes or gold.get('video_id') != video_id
            or gold.get('split') != split or gold.get('segment_count') != len(segments)
            or gold.get('segment_sha256') != segment_fingerprint(segments)):
        raise ValueError('Relevance labels do not match the current sources and segmentation')
    positive = [item for item in questions if item['expected_answerable']]
    expected = {item['question_id'] for item in positive}
    rows = gold.get('judgments')
    valid_segments = {item.segment_id for item in segments}
    if not positive or not isinstance(rows, list) or len(rows) != len(positive):
        raise ValueError('One declared relevance entry is required for each answerable question')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'question_id', 'relevant_segment_ids', 'verified', 'exhaustive'}:
            raise ValueError('Invalid relevance entry')
        qid, ids = row['question_id'], row['relevant_segment_ids']
        if (not isinstance(qid, str) or qid not in expected or qid in result
                or not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids)
                or len(set(ids)) != len(ids) or not set(ids) <= valid_segments
                or row['verified'] is not True or row['exhaustive'] is not True):
            raise ValueError('Each answerable question needs verified, exhaustively declared relevant segments')
        result[qid] = ids
    return result


def experiment_settings(args) -> dict:
    return {'video_id': args.video_id, 'split': args.split, 'retrieval_depth': 5, 'rrf_k': 60,
            'text_threshold': None, 'visual_threshold': None, 'device': 'cpu',
            'semantic_document_token_limit': 512, 'clip_query_token_limit': 77,
            'rerank_candidate_depth': 20 if args.reranker_model is not None else None,
            'rerank_batch_size': 4 if args.reranker_model is not None else None,
            'rerank_max_length': 1024 if args.reranker_model is not None else None,
            'frame_sampling': 'one midpoint per sampled segment', 'frame_sample_limit': 100}


def prepare_inputs(args) -> dict:
    """Validate provenance, labels and fixed model inputs without running inference."""
    if args.semantic_model.suffix.lower() != '.gguf':
        raise ValueError('Provide the local Qwen3-Embedding-0.6B GGUF file')
    labels = private_output(args.labels)
    gold_path = private_output(args.gold)
    binding_path = labels.with_suffix(labels.suffix + '.source.json')
    source_paths = {'video': args.video, 'subtitles': args.subtitles, 'annotations': labels,
                    'annotation_binding': binding_path}
    paths = {**source_paths, 'gold': gold_path, **semantic_model_files(args.semantic_model),
             'visual_tokenizer': args.visual_model / 'tokenizer.json',
             'visual_model': args.visual_model / 'onnx' / 'model_quantized.onnx',
             **{f'code:{path}': ROOT / path for path in CODE_FILES}}
    model_dir = None
    reranker_paths = {}
    if args.reranker_model is not None:
        model_dir = args.reranker_model.expanduser().resolve()
        reranker_paths = local_reranker_files(model_dir)
        paths.update(reranker_paths)
    versions_before = path_versions(paths)
    before = {name: fingerprint(path) for name, path in paths.items()}
    if versions_before != path_versions(paths):
        raise ValueError('An input or model changed during preparation')
    versions = dependency_versions()
    binding = json.loads(binding_path.read_text(encoding='utf-8-sig'))
    if binding != {'video_sha256': before['video'], 'video_id': args.video_id, 'split': args.split}:
        raise ValueError('Annotation binding does not match this video and split')
    with labels.open(encoding='utf-8-sig', newline='') as source:
        data = prepare(list(csv.DictReader(source)), args.video_id, args.split)
    segments = segments_from_subtitles(args.video, args.subtitles)
    gold = json.loads(gold_path.read_text(encoding='utf-8-sig'))
    ground_truth = validate_gold(gold, hashes={name: before[name] for name in source_paths},
                                 video_id=args.video_id, split=args.split, questions=data['questions'], segments=segments)
    return {'paths': paths, 'hashes': before, 'path_versions': versions_before,
            'dependency_versions': versions, 'model_dir': model_dir, 'reranker_paths': reranker_paths,
            'data': data, 'segments': segments, 'ground_truth': ground_truth,
            'segment_sha256': segment_fingerprint(segments), 'settings': experiment_settings(args)}


def verify_inputs(prepared: dict) -> None:
    paths = prepared['paths']
    if ((prepared['model_dir'] is not None and
         prepared['reranker_paths'] != local_reranker_files(prepared['model_dir']))
            or prepared['path_versions'] != path_versions(paths)
            or prepared['hashes'] != {name: fingerprint(path) for name, path in paths.items()}
            or prepared['path_versions'] != path_versions(paths)
            or prepared['dependency_versions'] != dependency_versions()):
        raise ValueError('An input, model, code or runtime changed; discard this run')


def comparison_gold_metadata(paths: list[Path], test_video_hash: str) -> list[dict]:
    """Check recorded video hashes for the declared development/validation sets."""
    if not paths:
        raise ValueError('Provide at least one development or validation relevance file')
    records = []
    seen_videos = set()
    for supplied in paths:
        path = private_output(supplied)
        before = fingerprint(path)
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, dict):
            raise TypeError('Comparison relevance file must be an object')
        source = data.get('source_sha256')
        video_hash = source.get('video') if isinstance(source, dict) else None
        if (data.get('schema_version') != 1 or data.get('status') != 'relevance_draft_not_independently_verified'
                or data.get('split') not in {'development', 'validation'}
                or not isinstance(video_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', video_hash)):
            raise ValueError('Comparison relevance file needs a recorded development or validation video hash')
        if video_hash == test_video_hash or video_hash in seen_videos:
            raise ValueError('Each declared split must use a distinct video')
        if fingerprint(path) != before:
            raise ValueError('A comparison relevance file changed while reading')
        seen_videos.add(video_hash)
        records.append({'sha256': before, 'video_sha256': video_hash, 'split': data['split']})
    return sorted(records, key=lambda item: (item['video_sha256'], item['sha256']))


def frozen_snapshot(prepared: dict, comparison_gold: list[dict]) -> dict:
    return {'input_sha256': prepared['hashes'], 'dependency_versions': prepared['dependency_versions'],
            'segment_sha256': prepared['segment_sha256'], 'settings': prepared['settings'],
            'comparison_gold': comparison_gold}


def paired_differences(predictions: dict, gold: dict) -> dict:
    """Compare each route against text on the same questions; output aggregates only."""
    if not gold or 'text' not in predictions or any(set(rows) != set(gold) for rows in predictions.values()):
        raise ValueError('Paired comparisons require the same nonempty set of questions for every route')
    pairs = {}
    functions = {
        'recall@5': lambda ranking, relevant: recall_at_k(ranking, relevant, 5),
        'hit@5': lambda ranking, relevant: hit_at_k(ranking, relevant, 5),
        'mrr@5': lambda ranking, relevant: reciprocal_rank(ranking[:5], relevant),
    }
    for route in ('frame', 'rrf', 'text_reranked'):
        if route not in predictions:
            continue
        comparisons = {}
        for name, metric in functions.items():
            diffs = [metric(predictions[route][qid], set(relevant))
                     - metric(predictions['text'][qid], set(relevant)) for qid, relevant in gold.items()]
            tied = sum(math.isclose(value, 0, abs_tol=1e-12) for value in diffs)
            improved = sum(value > 1e-12 for value in diffs)
            comparisons[name] = {'mean_difference': sum(diffs) / len(diffs), 'improved': improved,
                                 'worsened': len(diffs) - tied - improved, 'tied': tied}
        pairs[f'{route}_minus_text'] = {'question_count': len(gold), 'metrics': comparisons}
    return pairs


def compare_rankings(questions: list, gold: dict, text_retriever, visual_retriever, known_ids: set[str],
                     *, reranker=None, segments_by_id: dict | None = None, paired_out: dict | None = None) -> dict:
    """Compare fixed top-five rankings without using unanswerable questions."""
    positive = [item for item in questions if item['expected_answerable']]
    if not positive or set(gold) != {item['question_id'] for item in positive}:
        raise ValueError('Gold questions must match all answerable questions')
    predictions = {'text': {}, 'frame': {}, 'rrf': {}}
    if reranker is not None:
        if segments_by_id is None or set(segments_by_id) != known_ids:
            raise ValueError('Reranking requires the same segment set')
        predictions['text_reranked'] = {}
    scored_questions = []
    for item in positive:
        qid, query = item['question_id'], item['question']
        text_candidates = text_retriever.search(query, 20 if reranker is not None else 5)
        frame_candidates = visual_retriever.search(query, 5)
        for hits, limit in ((text_candidates, 20 if reranker is not None else 5), (frame_candidates, 5)):
            ids = [hit.segment_id for hit in hits]
            if len(ids) > limit or len(set(ids)) != len(ids) or not set(ids) <= known_ids:
                raise ValueError('Retriever returned invalid or duplicate segment IDs')
        text_hits = [SearchHit(hit.segment_id, hit.score, 'text', rank)
                     for rank, hit in enumerate(text_candidates[:5], 1)]
        frame_hits = [SearchHit(hit.segment_id, hit.score, 'frame', rank)
                      for rank, hit in enumerate(frame_candidates, 1)]
        fused = reciprocal_rank_fusion([text_hits, frame_hits], k=60, top_k=5)
        predictions['text'][qid] = [hit.segment_id for hit in text_hits]
        predictions['frame'][qid] = [hit.segment_id for hit in frame_hits]
        predictions['rrf'][qid] = [hit.segment_id for hit in fused]
        if reranker is not None:
            candidates = [segments_by_id[hit.segment_id] for hit in text_candidates]
            scores = []
            for offset in range(0, len(candidates), 4):
                batch = candidates[offset:offset + 4]
                batch_scores = reranker.score(query, batch)
                if len(batch_scores) != len(batch) or any(not math.isfinite(value) for value in batch_scores):
                    raise ValueError('Reranker must return one finite score per candidate')
                scores.extend(batch_scores)
            ordered = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
            predictions['text_reranked'][qid] = [segment.segment_id for segment, _ in ordered[:5]]
        scored_questions.append({'question_id': qid, 'question_type': item.get('question_type', 'unlabeled'),
                                 'relevant_segment_ids': gold[qid]})
    if paired_out is not None:
        paired_out.update(paired_differences(predictions, gold))
    return {name: metrics_with_breakdown(ranking, scored_questions, cutoffs=(1, 5))
            for name, ranking in predictions.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video', 'subtitles', 'labels', 'gold', 'semantic-model', 'visual-model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--split', choices=('development', 'validation', 'test'), required=True)
    parser.add_argument('--reranker-model', type=Path, help='Optional local Qwen3-Reranker-0.6B model directory (CPU)')
    parser.add_argument('--frozen-test-manifest', type=Path, help='Required only for a declared frozen test')
    parser.add_argument('--other-gold', type=Path, action='append', default=[],
                        help='Development/validation relevance file used when freezing the test')
    parser.add_argument('--confirm-reviewed', action='store_true',
                        help='Declare that the selected relevant segments and completeness were checked against the video')
    args = parser.parse_args()
    try:
        if not args.confirm_reviewed:
            raise ValueError('Declare that relevance labels were reviewed before scoring')
        target = private_output(args.output)
        if target.suffix.lower() != '.json' or target.exists():
            raise ValueError('Choose a new local JSON output')
        is_test = args.split == 'test'
        if is_test != (args.frozen_test_manifest is not None) or (not is_test and args.other_gold):
            raise ValueError('A frozen manifest and comparison relevance files are only used for a test')
        prepared = prepare_inputs(args)
        data, segments, ground_truth = prepared['data'], prepared['segments'], prepared['ground_truth']
        manifest_path = manifest_hash = None
        other = []
        if is_test:
            manifest_path = private_output(args.frozen_test_manifest)
            manifest_hash = fingerprint(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
            other = comparison_gold_metadata(args.other_gold, prepared['hashes']['video'])
            expected = frozen_snapshot(prepared, other)
            if (not isinstance(manifest, dict) or manifest.get('status') != 'frozen_segment_test_declared_not_independently_verified'
                    or any(manifest.get(key) != value for key, value in expected.items())):
                raise ValueError('Frozen test inputs, settings, code or runtime do not match')
            if fingerprint(manifest_path) != manifest_hash:
                raise ValueError('Frozen test manifest changed while reading')
        from video_rag.retrieval.onnx_clip import ClipFrameRetriever, OnnxClipEncoder

        print('Building local text and frame indexes; no video data is uploaded.', file=sys.stderr, flush=True)
        text = SemanticRetriever(load_semantic_encoder(args.semantic_model))
        text.build(segments)
        sampled = sample_visual_segments(tuple(segments), 100)
        visual = ClipFrameRetriever(OnnxClipEncoder(args.visual_model))
        visual.build(sampled)
        reranker = None
        if args.reranker_model is not None:
            print('Preparing the optional local CPU reranker.', file=sys.stderr, flush=True)
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            from video_rag.adapters.qwen import Qwen3Reranker

            class TranscriptOnlyReranker(Qwen3Reranker):
                @staticmethod
                def _document(segment):
                    return segment.transcript

            reranker = TranscriptOnlyReranker(str(prepared['model_dir']), device_map='cpu', max_length=1024,
                                             local_files_only=True)
        segment_map = {item.segment_id: item for item in segments}
        paired = {}
        results = compare_rankings(data['questions'], ground_truth, text, visual, set(segment_map),
                                   reranker=reranker, segments_by_id=segment_map, paired_out=paired)
        verify_inputs(prepared)
        if is_test and (fingerprint(manifest_path) != manifest_hash
                        or comparison_gold_metadata(args.other_gold, prepared['hashes']['video']) != other):
            raise ValueError('Frozen test manifest or comparison relevance changed during evaluation')
        report = {'status': 'declared_local_relevance_not_independently_verified',
                  'test_declaration': 'frozen_segment_test_declared_not_independently_verified' if is_test else None,
                  'frozen_manifest_sha256': manifest_hash,
                  'split': args.split, 'source_sha256': prepared['hashes'],
                  'segment_sha256': prepared['segment_sha256'],
                  'dependency_versions': prepared['dependency_versions'],
                  'answerable_questions': len(ground_truth),
                  'unanswerable_questions_excluded': sum(not item['expected_answerable'] for item in data['questions']),
                  'settings': {**prepared['settings'], 'sampled_segments': len(sampled),
                               'total_segments': len(segments)},
                  'metrics': results, 'paired_comparisons': paired,
                  'warning': 'Labels and completeness are user declarations, not independently verified. '
                             'Top-five unthresholded ranking differs from the workspace top-three pipeline. '
                             'Frame sampling can omit evidence; code hashes and dependency versions do not capture '
                             'every runtime detail. Paired differences are descriptive, not proof of generalization. '
                             'Keep this report and fingerprints local.'}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as out:
            json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write('\n')
        print('Private local comparison saved; review its label and sampling limitations.')
        return 0
    except (ImportError, OSError, UnicodeError, KeyError, TypeError, ValueError, RuntimeError, csv.Error) as exc:
        parser.exit(2, f'Local retrieval comparison failed: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
