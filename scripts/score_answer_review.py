"""Summarize explicitly human-rated answer drafts; keep all results local."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from video_rag.evaluation.private_output import private_output


def summarize(report: dict) -> dict:
    """Apply question-level micro metrics to declared human answer judgments."""
    if not isinstance(report, dict) or report.get('status') != 'unreviewed_answer_drafts':
        raise ValueError('Expected an answer review draft')
    if report.get('split') == 'test':
        raise ValueError('Final test split is reserved')
    rows = report.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('No review rows')
    tp = fp = fn = tn = 0
    generated = unrecorded = 0
    for row in rows:
        if not isinstance(row, dict) or type(row.get('expected_answerable')) is not bool:
            raise ValueError('Every row must have a boolean reference label')
        origin = row.get('annotation_source', 'not_recorded')
        if not isinstance(origin, str) or not origin:
            raise ValueError('Invalid question origin')
        generated += int(origin == 'model_generated_candidate_unverified')
        unrecorded += int(origin == 'not_recorded')
        verdict = row.get('human_answer_verdict')
        evidence = row.get('human_evidence_verdict')
        if verdict not in {'correct', 'incorrect', 'abstain'} or evidence not in {'supported', 'unsupported', 'not_applicable'}:
            raise ValueError('Every row requires answer and evidence verdicts')
        if verdict == 'correct' and (not row['expected_answerable'] or evidence != 'supported'):
            raise ValueError('A correct answer requires an answerable question and supported evidence')
        if verdict == 'abstain' and evidence != 'not_applicable':
            raise ValueError('An abstention requires evidence verdict not_applicable')
        if verdict == 'correct':
            tp += 1
        elif verdict == 'incorrect':
            fp += 1
            fn += int(row['expected_answerable'])
        elif row['expected_answerable']:
            fn += 1
        else:
            tn += 1
    return {'question_count': len(rows), 'model_generated_questions': generated,
            'question_origin_not_recorded': unrecorded, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--confirm-human-review', action='store_true',
                        help='Declare that a human reviewed each answer, evidence and reference against the video')
    args = parser.parse_args()
    try:
        if not args.confirm_human_review:
            raise ValueError('Human review of every row and reference must be declared')
        source = private_output(args.review)
        target = private_output(args.output)
        if source == target or target.suffix.lower() != '.json' or target.exists():
            raise ValueError('Choose a new local JSON output')
        raw = source.read_bytes()
        report = json.loads(raw.decode('utf-8-sig'))
        scores = summarize(report)
        if source.read_bytes() != raw:
            raise ValueError('Review changed during scoring; try again')
        result = {'status': 'human_review_declared_not_independently_verified',
                  'split': report.get('split'), 'binding': report.get('binding'),
                  'generation_config': report.get('generation_config'),
                  'retrieval_config': report.get('retrieval_config'),
                  'review_sha256': hashlib.sha256(raw).hexdigest(), 'metrics': scores,
                  'method': 'Correct supported answer=TP; wrong answer=FP and, if answerable, FN; '
                            'abstention on answerable=FN; abstention on unanswerable=TN.',
                  'warning': 'Metrics reflect declared labels and judgments, which this script cannot verify; '
                             'draft or unbound questions do not become a validated test set.'}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as out:
            json.dump(result, out, ensure_ascii=False, indent=2)
            out.write('\n')
        print('Private summary saved. Review declaration and label quality are not independently verified.')
        return 0
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        parser.exit(2, f'Could not score review: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
