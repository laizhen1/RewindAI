from __future__ import annotations

import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "transcribe_local.py"
spec = importlib.util.spec_from_file_location("transcribe_local", SCRIPT)
assert spec and spec.loader
transcribe_local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transcribe_local)


class LocalTranscribeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.media = self.root / "lecture.mp4"
        self.media.touch()
        self.model = self.root / "model-cache"
        self.model.mkdir()
        (self.model / "tokenizer.json").touch()
        self.outputs = self.root / "transcripts"
        self.root_patch = patch.object(transcribe_local, "OUTPUT_ROOT", self.outputs)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_local_model_and_srt_conversion(self):
        calls = {}

        class Model:
            def __init__(self, *args, **kwargs):
                calls["init"] = args, kwargs

            def transcribe(self, path, language):
                calls["transcribe"] = path, language
                segments = [types.SimpleNamespace(start=1.1, end=3.45, text=" hello world ")]
                return iter(segments), None

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = Model
        fake_utils = types.ModuleType("faster_whisper.utils")
        fake_utils.download_model = lambda name, local_files_only: str(self.model)
        with patch.dict("sys.modules", {"faster_whisper": fake, "faster_whisper.utils": fake_utils}), patch.dict(os.environ, {}, clear=False):
            result = transcribe_local.main([str(self.media), "--language", "en"])
        self.assertEqual(result, 0)
        self.assertEqual(calls["init"], ((str(self.model),), {
            "device": "cpu", "compute_type": "int8", "local_files_only": True,
        }))
        self.assertEqual(calls["transcribe"], (str(self.media.resolve()), "en"))
        self.assertEqual((self.outputs / "lecture.srt").read_text(encoding="utf-8"),
                         "1\n00:00:01,100 --> 00:00:03,450\nhello world\n")

    def test_download_opt_in_and_chinese(self):
        calls = {}

        class Model:
            def __init__(self, *args, **kwargs):
                calls.update(kwargs)

            def transcribe(self, path, language):
                self.assert_language = language
                return iter([types.SimpleNamespace(start=0, end=1, text="你好")]), None

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = Model
        fake_utils = types.ModuleType("faster_whisper.utils")
        def fake_download(name, local_files_only):
            calls["download_local_only"] = local_files_only
            return str(self.model)
        fake_utils.download_model = fake_download
        with patch.dict("sys.modules", {"faster_whisper": fake, "faster_whisper.utils": fake_utils}), patch.dict(os.environ, {}, clear=False):
            result = transcribe_local.main([str(self.media), "--language", "zh", "--allow-model-download"])
        self.assertEqual(result, 0)
        self.assertFalse(calls["download_local_only"])
        self.assertTrue(calls["local_files_only"])
        self.assertIn("你好", (self.outputs / "lecture.srt").read_text(encoding="utf-8"))

    def test_existing_output_and_outside_paths_are_never_overwritten(self):
        self.outputs.mkdir()
        target = self.outputs / "lecture.srt"
        target.write_text("original", encoding="utf-8")
        self.assertEqual(transcribe_local.main([str(self.media)]), 2)
        self.assertEqual(target.read_text(encoding="utf-8"), "original")
        self.assertEqual(transcribe_local.main([str(self.media), "--output", str(self.root / "out.srt")]), 2)

    def test_invalid_and_empty_segments_do_not_create_subtitle(self):
        with self.assertRaises(ValueError):
            transcribe_local.render_srt([types.SimpleNamespace(start=2, end=1, text="wrong")])
        with self.assertRaises(ValueError):
            transcribe_local.render_srt([])
        self.assertFalse(self.outputs.exists())

    def test_local_model_directory_requires_tokenizer(self):
        model = self.root / "model"
        model.mkdir()
        self.assertEqual(transcribe_local.main([str(self.media), "--model", str(model)]), 2)


if __name__ == "__main__":
    unittest.main()
