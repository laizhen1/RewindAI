"""One local, offline transcription job for the learning workspace."""
from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import uuid
import weakref
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_ACTIVE_JOBS = weakref.WeakSet()


def _cancel_jobs_on_exit() -> None:
    """Best effort cleanup for normal Python exit; hard termination may bypass it."""
    for job in list(_ACTIVE_JOBS):
        job.cancel()


atexit.register(_cancel_jobs_on_exit)


class LocalTranscriptionJob:
    """A single process-local job. Input and output are never sent to a service."""
    def __init__(self, import_subtitles: Callable[[Path, Path], bool]):
        self._lock = threading.Lock()
        self._state = {"state": "idle", "message": ""}
        self._import_subtitles = import_subtitles
        self._process = None
        self._cancel_requested = False
        _ACTIVE_JOBS.add(self)

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _set(self, state: str, message: str) -> None:
        with self._lock:
            self._state = {"state": state, "message": message}

    def start(self, video: Path, language: str | None) -> bool:
        with self._lock:
            if self._state["state"] in {"transcribing", "cancelling", "indexing"}:
                return False
            self._cancel_requested = False
            self._state = {"state": "transcribing", "message": "正在本机转录，耗时取决于视频长度。"}
        try:
            threading.Thread(target=self._run, args=(video, language), name="RewindAI-transcription", daemon=True).start()
        except RuntimeError:
            self._set("failed", "无法启动本地转录任务。")
            raise
        return True

    def cancel(self) -> bool:
        """Cancel only before indexing; the source and any output are never deleted."""
        with self._lock:
            if self._state["state"] != "transcribing":
                return False
            self._cancel_requested = True
            self._state = {"state": "cancelling", "message": "正在停止本地转录。"}
            if self._process is not None:
                try:
                    self._process.terminate()
                except OSError:
                    # It may have exited between the request and terminate().
                    pass
            return True

    def _run(self, video: Path, language: str | None) -> None:
        try:
            original = video.stat()
            output = ROOT / "artifacts" / "transcripts" / f"workspace-{uuid.uuid4().hex}.srt"
            command = [sys.executable, str(ROOT / "scripts" / "transcribe_local.py"), str(video),
                       "--output", str(output), "--model", "base"]
            if language:
                command.extend(["--language", language])
            env = os.environ.copy()
            env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
            cache = ROOT / "artifacts" / "hf-cache"
            if ((cache / "hub" / "models--Systran--faster-whisper-base" / "refs" / "main").is_file()
                    and not any(env.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"))):
                env["HF_HOME"] = str(cache)
            with self._lock:
                if self._cancel_requested:
                    self._state = {"state": "cancelled", "message": "已取消本地转录，原视频和已有资料保留。"}
                    return
                self._process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.DEVNULL)
                process = self._process
            return_code = process.wait()
            with self._lock:
                self._process = None
                if self._cancel_requested:
                    self._state = {"state": "cancelled", "message": "已取消本地转录，原视频和已有资料保留；已生成的字幕不会删除。"}
                    return
                if return_code:
                    self._state = {"state": "failed", "message": "转录失败。请检查 faster-whisper、已缓存的 base 模型和视频音轨；未下载模型。"}
                    return
                current = video.stat()
                if (original.st_size, original.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
                    self._state = {"state": "failed", "message": "视频在转录期间发生变化，未导入索引。生成的字幕仍保存在本机。"}
                    return
                self._state = {"state": "indexing", "message": "转录完成，正在本机建立字幕索引。"}
            if not self._import_subtitles(video, output):
                self._set("failed", "转录完成，但无法导入索引；请检查视频是否已存在或源文件是否变化。字幕仍保存在本机。")
                return
            self._set("complete", "本地转录和索引完成，请核对字幕内容。")
        except (OSError, RuntimeError, ValueError):
            self._set("failed", "本地转录未完成，请检查文件、依赖和磁盘空间。")
