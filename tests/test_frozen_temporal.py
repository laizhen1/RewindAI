import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import compare_temporal_multimodal, freeze_temporal_test
from scripts.temporal_label_server import FIELDS
from video_rag.evaluation import frozen_temporal
from video_rag.schemas import SearchHit, VideoSegment


def test_frozen_manifest_binds_inputs_configuration_code_and_runtime(tmp_path, monkeypatch):
    video = tmp_path / 'video.mp4'; video.write_bytes(b'synthetic')
    subtitles = tmp_path / 'video.srt'; subtitles.write_text('synthetic')
    csv_path = tmp_path / 'questions.csv'
    row = {key: '' for key in FIELDS}
    row.update(question_id='q', video_id='v', split='test', question_type='audio', answerable='true',
               question='Synthetic question?', reference_answer='Synthetic answer',
               evidence_start_seconds='1', evidence_end_seconds='3')
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerow(row)
    binding = {'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'video_id': 'v', 'split': 'test'}
    csv_path.with_suffix('.csv.source.json').write_text(json.dumps(binding), encoding='utf-8')
    model = tmp_path / 'clip'; (model / 'onnx').mkdir(parents=True)
    (model / 'tokenizer.json').write_text('{}')
    (model / 'onnx/model_quantized.onnx').write_bytes(b'model')
    code = tmp_path / 'code.py'; code.write_text('original')
    monkeypatch.setattr(frozen_temporal, 'ROOT', tmp_path)
    monkeypatch.setattr(frozen_temporal, 'CODE_FILES', ('code.py',))
    monkeypatch.setattr(freeze_temporal_test, 'private_output', lambda path: path)
    monkeypatch.setattr(freeze_temporal_test, 'write_output',
                        lambda path, data, roots: Path(path).write_text(json.dumps(data), encoding='utf-8'))

    class Container:
        duration = 10_000_000
        def __enter__(self): return self
        def __exit__(self, *_): pass

    monkeypatch.setitem(sys.modules, 'av', SimpleNamespace(open=lambda _: Container(), time_base=1_000_000))
    manifest = tmp_path / 'manifest.json'
    args = ['freeze', '--video', str(video), '--subtitles', str(subtitles), '--questions', str(csv_path),
            '--visual-model', str(model), '--video-id', 'v', '--output', str(manifest)]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit):
        freeze_temporal_test.main()
    assert not manifest.exists()
    monkeypatch.setattr(sys, 'argv', [*args, '--confirm-labels-reviewed'])
    assert freeze_temporal_test.main() == 0
    paths = frozen_temporal.input_paths(video, subtitles, csv_path, model)
    settings = frozen_temporal.config('v', None, 0.4, None, 0.5)
    frozen_temporal.read_and_verify(manifest, paths, settings)
    with pytest.raises(ValueError, match='configuration'):
        frozen_temporal.read_and_verify(manifest, paths, {**settings, 'visual_min_score': 0.25})

    from video_rag.retrieval import onnx_clip

    class Visual:
        def __init__(self, encoder): pass
        def build(self, segments): pass
        def search(self, query, top_k): return [SearchHit('s', 0.5, 'visual', 1)]

    monkeypatch.setattr(onnx_clip, 'OnnxClipEncoder', lambda path: None)
    monkeypatch.setattr(onnx_clip, 'ClipFrameRetriever', Visual)
    monkeypatch.setattr(compare_temporal_multimodal, 'segments_from_subtitles',
                        lambda *_: [VideoSegment('s', 'v', 'local', 1, 3, transcript='Synthetic question answer')])
    monkeypatch.setattr(compare_temporal_multimodal, 'private_output', lambda path: path)
    monkeypatch.setattr(compare_temporal_multimodal, 'write_output',
                        lambda path, data, roots: Path(path).write_text(json.dumps(data), encoding='utf-8'))
    output = tmp_path / 'result.json'
    evaluate_args = ['compare', '--video', str(video), '--subtitles', str(subtitles), '--questions', str(csv_path),
                     '--visual-model', str(model), '--video-id', 'v', '--split', 'test', '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', evaluate_args)
    with pytest.raises(SystemExit):
        compare_temporal_multimodal.main()
    assert not output.exists()
    monkeypatch.setattr(sys, 'argv', [*evaluate_args, '--frozen-test-manifest', str(manifest)])
    assert compare_temporal_multimodal.main() == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['split'] == 'test' and result['test_declaration'] == 'frozen_test_declared_not_independently_verified'
    assert result['comparison']['fused_experimental']['temporal_recall_at_3']['answerable_questions'] == 1
    changed_args = [*evaluate_args[:-1], str(tmp_path / 'changed.json'), '--frozen-test-manifest', str(manifest),
                    '--visual-min-score', '0.25']
    monkeypatch.setattr(sys, 'argv', changed_args)
    with pytest.raises(SystemExit):
        compare_temporal_multimodal.main()
    assert not (tmp_path / 'changed.json').exists()
    original_labels = csv_path.read_bytes()
    csv_path.write_bytes(original_labels + b'\n')
    with pytest.raises(ValueError, match='inputs'):
        frozen_temporal.read_and_verify(manifest, paths, settings)
    csv_path.write_bytes(original_labels)
    code.write_text('changed')
    with pytest.raises(ValueError, match='code'):
        frozen_temporal.read_and_verify(manifest, paths, settings)
