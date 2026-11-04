import threading

from video_rag.api import create_app
from video_rag.api import transcription_job
from video_rag.transcript_baseline import build_transcript_segments


def test_local_transcription_runs_offline_once_and_imports(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    started, finish = threading.Event(), threading.Event()
    monkeypatch.setattr(transcription_job, "ROOT", tmp_path)
    def fake_popen(command, **kwargs):
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert "--allow-model-download" not in command
        assert command[command.index("--language") + 1] == "en"
        path = __import__("pathlib").Path(command[command.index("--output") + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1\n00:00:01,000 --> 00:00:04,000\napple learning\n", encoding="utf-8")
        started.set()
        class Process:
            def wait(self):
                assert finish.wait(5)
                return 0
        return Process()
    monkeypatch.setattr(transcription_job.subprocess, "Popen", fake_popen)
    app = create_app(build_transcript_segments([]), database_path=tmp_path / "workspace.sqlite3", mode="transcript")
    client = app.test_client()
    assert client.get("/api/transcription").get_json()["state"] == "idle"
    assert client.post("/api/transcription", json={"video_path": str(video), "language": "en"},
                       headers={"Origin": "https://bad.example"}).status_code == 403
    assert client.post("/api/transcription", json={"video_path": str(video), "language": "en"}).status_code == 202
    assert started.wait(5)
    assert client.get("/api/transcription").get_json()["state"] == "transcribing"
    assert client.post("/api/transcription", json={"video_path": str(video)}).status_code == 409
    # Observe the worker reaching a terminal state without polling indefinitely.
    job = next(t for t in threading.enumerate() if t.name == "RewindAI-transcription" and t is not threading.current_thread())
    finish.set()
    job.join(5)
    assert not job.is_alive()
    assert client.get("/api/transcription").get_json()["state"] == "complete"
    assert len(client.get("/api/library").get_json()["videos"]) == 1
    assert not client.post("/api/ask", json={"question": "apple"}).get_json()["abstained"]
    assert client.post("/api/transcription", json={"video_path": str(video)}).status_code == 409


def test_local_transcription_rejects_changed_video_without_import(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    started, finish = threading.Event(), threading.Event()
    monkeypatch.setattr(transcription_job, "ROOT", tmp_path)
    def fake_popen(command, **kwargs):
        started.set()
        class Process:
            def wait(self):
                assert finish.wait(5)
                return 0
        return Process()
    monkeypatch.setattr(transcription_job.subprocess, "Popen", fake_popen)
    job = transcription_job.LocalTranscriptionJob(lambda video, output: (_ for _ in ()).throw(AssertionError("must not import")))
    assert job.start(video, None)
    assert started.wait(5)
    worker = next(t for t in threading.enumerate() if t.name == "RewindAI-transcription" and t is not threading.current_thread())
    video.write_bytes(b"a different video")
    finish.set()
    worker.join(5)
    assert not worker.is_alive()
    assert job.status()["state"] == "failed"
    assert "发生变化" in job.status()["message"]


def test_failed_transcription_never_imports(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(transcription_job, "ROOT", tmp_path)
    class Process:
        def wait(self):
            return 1
    monkeypatch.setattr(transcription_job.subprocess, "Popen", lambda *args, **kwargs: Process())
    imported = []
    job = transcription_job.LocalTranscriptionJob(lambda *args: imported.append(args))
    job._run(video, None)
    assert job.status()["state"] == "failed"
    assert not imported
    assert video.read_bytes() == b"video"


def test_cancellation_stops_process_and_never_imports(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    started, terminated = threading.Event(), threading.Event()
    monkeypatch.setattr(transcription_job, "ROOT", tmp_path)
    class Process:
        def wait(self):
            assert terminated.wait(5)
            return 1
        def terminate(self):
            terminated.set()
    def fake_popen(*args, **kwargs):
        started.set()
        return Process()
    monkeypatch.setattr(transcription_job.subprocess, "Popen", fake_popen)
    imported = []
    job = transcription_job.LocalTranscriptionJob(lambda *args: imported.append(args))
    assert not job.cancel()
    assert job.start(video, None)
    assert started.wait(5)
    worker = next(t for t in threading.enumerate() if t.name == "RewindAI-transcription")
    assert job.cancel()
    worker.join(5)
    assert not worker.is_alive()
    assert terminated.is_set()
    assert job.status()["state"] == "cancelled"
    assert not imported
    assert video.read_bytes() == b"video"


def test_cancel_rejected_after_indexing_begins(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(transcription_job, "ROOT", tmp_path)
    started, finish = threading.Event(), threading.Event()
    class Process:
        def wait(self):
            return 0
    monkeypatch.setattr(transcription_job.subprocess, "Popen", lambda *args, **kwargs: Process())
    def importer(*args):
        started.set()
        assert finish.wait(5)
        return True
    job = transcription_job.LocalTranscriptionJob(importer)
    assert job.start(video, None)
    assert started.wait(5)
    worker = next(t for t in threading.enumerate() if t.name == "RewindAI-transcription")
    assert not job.cancel()
    assert not job.start(video, None)
    finish.set()
    worker.join(5)
    assert job.status()["state"] == "complete"


def test_cancel_api_checks_origin_and_reports_terminal_state(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(transcription_job, "ROOT", tmp_path)
    started, terminated = threading.Event(), threading.Event()
    class Process:
        def wait(self):
            assert terminated.wait(5)
            return 1
        def terminate(self):
            terminated.set()
    def fake_popen(*args, **kwargs):
        started.set()
        return Process()
    monkeypatch.setattr(transcription_job.subprocess, "Popen", fake_popen)
    client = create_app(build_transcript_segments([]), database_path=tmp_path / "workspace.sqlite3", mode="transcript").test_client()
    assert client.post("/api/transcription", json={"video_path": str(video)}).status_code == 202
    assert started.wait(5)
    worker = next(t for t in threading.enumerate() if t.name == "RewindAI-transcription")
    assert client.post("/api/transcription/cancel", headers={"Origin": "https://bad.example"}).status_code == 403
    assert not terminated.is_set()
    assert client.post("/api/transcription/cancel").status_code == 202
    worker.join(5)
    assert client.get("/api/transcription").get_json()["state"] == "cancelled"
    assert client.post("/api/transcription/cancel").status_code == 409
    assert client.get("/api/library").get_json()["videos"] == []
