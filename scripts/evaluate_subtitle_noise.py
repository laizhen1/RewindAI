"""Compare clean versus synthetic word-deletion retrieval on public transcript data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from scripts.evaluate_ytcommentqa_full_transcript import evaluate
from video_rag.evaluation.private_output import private_output
from video_rag.evaluation.provenance import fingerprint
from video_rag.retrieval.onnx_semantic import OnnxSentenceEncoder, build_semantic_pipeline
from video_rag.transcript_baseline import build_transcript_segments


def delete_words(segments):
    """Delete approximately 10% of whitespace tokens deterministically, preserving times."""
    result = []
    for segment in segments:
        words = segment.transcript.split()
        kept = []
        for index, word in enumerate(words):
            key = json.dumps([segment.video_id, segment.segment_id, index], ensure_ascii=False).encode('utf-8')
            if int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 10:
                kept.append(word)
        result.append(replace(segment, transcript=' '.join(kept)))
    return result


def paired(clean, noisy):
    left, right = clean.pop('_outcomes'), noisy.pop('_outcomes')
    if left.keys() != right.keys():
        raise ValueError('Clean and noisy runs did not evaluate the same questions')
    counts = Counter()
    for key in left:
        a, b = left[key][1], right[key][1]
        label = 'both_overlap' if a and b else 'clean_only' if a else 'noisy_only' if b else 'neither_overlap'
        counts[label] += 1
    return {key: counts[key] for key in ('both_overlap', 'clean_only', 'noisy_only', 'neither_overlap')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--semantic-model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sample', choices=('sample', 'other'), default='sample')
    args = parser.parse_args()
    try:
        output = private_output(args.output)
        if output.suffix.lower() != '.json' or output.exists():
            raise ValueError('Choose a new local JSON output')
        sources = {'dataset': args.dataset, 'tokenizer': args.semantic_model / 'tokenizer.json',
                   'model': args.semantic_model / 'onnx' / 'model_quantized.onnx'}
        before = {key: fingerprint(path) for key, path in sources.items()}
        data = json.loads(args.dataset.read_text(encoding='utf-8-sig'))
        encoder = OnnxSentenceEncoder(args.semantic_model)
        builders = {'keyword': build_transcript_segments,
                    'semantic': lambda segments: build_semantic_pipeline(segments, encoder, 0.4)}
        results = {}
        for name, builder in builders.items():
            clean = evaluate(data, sample=args.sample, builder=builder, include_rows=True, segmentation='workspace')
            token_counts = Counter()

            def noisy_builder(segments, build=builder, counts=token_counts):
                changed = delete_words(segments)
                counts['original'] += sum(len(item.transcript.split()) for item in segments)
                counts['remaining'] += sum(len(item.transcript.split()) for item in changed)
                return build(changed)

            noisy = evaluate(data, sample=args.sample, builder=noisy_builder, include_rows=True, segmentation='workspace')
            pairs = paired(clean, noisy)
            results[name] = {'clean': clean, 'synthetic_deletion': noisy, 'paired_top3': pairs,
                             'tokens': dict(token_counts)}
        if any(fingerprint(path) != before[key] for key, path in sources.items()):
            raise ValueError('Input changed during evaluation; discard results')
        report = {'sample': args.sample, 'source_sha256': before, 'deletion_rule': 'sha256-token-index-modulo-10',
                  'semantic_threshold': 0.4, 'results': results,
                  'warning': 'Synthetic word deletion is not an ASR error model, answer correctness, or lecture performance.'}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        print('Saved local aggregate. Keep private input fingerprints and results local.')
    except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.exit(2, f'Noise evaluation failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
