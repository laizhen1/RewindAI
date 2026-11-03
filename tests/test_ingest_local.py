from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_local.py"
spec = importlib.util.spec_from_file_location("ingest_local", SCRIPT)
assert spec is not None and spec.loader is not None
ingest_local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest_local)


class LocalIngestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video = self.root / "local lecture.mp4"
        self.config = self.root / "config.toml"
        self.video.touch()
        self.config.touch()
        self.artifacts = self.root / "runs"
        self.root_patch = patch.object(ingest_local, "ARTIFACTS_ROOT", self.artifacts)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_isolated_outputs_and_offline_defaults(self):
        with patch.object(ingest_local.subprocess, "run") as run:
            result = ingest_local.main([
                str(self.video), "--config", str(self.config), "--language", "en",
                "--device", "cpu", "--skip-captions",
            ])
        self.assertEqual(result, 0)
        self.assertEqual(run.call_count, 2)
        prepare, build = [call.args[0] for call in run.call_args_list]
        self.assertEqual(prepare[prepare.index("--input") + 1], str(self.video.resolve()))
        self.assertEqual(prepare[prepare.index("--language") + 1], "en")
        self.assertIn("--skip-captions", prepare)
        self.assertEqual(build[build.index("--device") + 1], "cpu")
        self.assertEqual(prepare[prepare.index("--output") + 1],
                         build[build.index("--segments") + 1])
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertEqual(call.kwargs["env"]["TRANSFORMERS_OFFLINE"], "1")
            self.assertTrue(call.kwargs["check"])

    def test_download_requires_explicit_opt_in(self):
        with patch.object(ingest_local.subprocess, "run") as run:
            result = ingest_local.main([
                str(self.video), "--config", str(self.config), "--allow-model-download",
            ])
        self.assertEqual(result, 0)
        self.assertTrue(all(call.kwargs["env"]["HF_HUB_OFFLINE"] == "0"
                            for call in run.call_args_list))

    def test_invalid_input_does_not_create_output_or_run(self):
        with patch.object(ingest_local.subprocess, "run") as run:
            result = ingest_local.main([str(self.root / "missing.mp4"),
                                        "--config", str(self.config)])
        self.assertEqual(result, 2)
        run.assert_not_called()
        self.assertFalse(self.artifacts.exists())

    def test_directory_input_is_passed_to_prepare(self):
        folder = self.root / "lectures"
        folder.mkdir()
        (folder / "one.webm").touch()
        with patch.object(ingest_local.subprocess, "run") as run:
            result = ingest_local.main([str(folder), "--config", str(self.config)])
        self.assertEqual(result, 0)
        prepare = run.call_args_list[0].args[0]
        self.assertEqual(prepare[prepare.index("--input") + 1], str(folder.resolve()))

    def test_directory_containing_video_named_subdirectory_is_rejected(self):
        folder = self.root / "lectures"
        folder.mkdir()
        (folder / "one.mp4").touch()
        (folder / "not-a-file.mp4").mkdir()
        with patch.object(ingest_local.subprocess, "run") as run:
            result = ingest_local.main([str(folder), "--config", str(self.config)])
        self.assertEqual(result, 2)
        run.assert_not_called()

    def test_prepare_failure_does_not_build_index(self):
        with patch.object(ingest_local.subprocess, "run",
                          side_effect=subprocess.CalledProcessError(4, "prepare")) as run:
            result = ingest_local.main([str(self.video), "--config", str(self.config)])
        self.assertEqual(result, 4)
        self.assertEqual(run.call_count, 1)

    def test_windows_command_quotes_powershell_metacharacters(self):
        command = ingest_local.format_server_command(
            [r"C:\Program Files\Python\python.exe", "quote's $HOME.mp4"], windows=True
        )
        self.assertEqual(command, "& 'C:\\Program Files\\Python\\python.exe' 'quote''s $HOME.mp4'")


if __name__ == "__main__":
    unittest.main()
