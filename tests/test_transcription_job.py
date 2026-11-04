import threading
from types import SimpleNamespace

from video_rag.api import create_app
from video_rag.api import transcription_job
from video_rag.transcript_baseline import build_transcript_segments


def test_local_transcription_runs_offline_once_and_imports(tmp_path, monkeypatch):
    video = tmp_path / "lecture.mp4"
    video.write_bytes(b"video")
    started, finish = threading.Event(), threading.Event()
    monkeypatch.setattr(transcription_job, "ROOT", tmp_path)
    def fake_run(command, **kwargs):
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert "--allow-model-download" not in command
        assert command[command.index("--language") + 1] == "en"
        path = __import__("pathlib").Path(command[command.index("--output") + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1\n00:00:01,000 --> 00:00:04,000\napple learning\n", encoding="utf-8")
        started.set()
        assert finish.wait(5)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(transcription_job.subprocess, "run", fake_run)
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
    def fake_run(command, **kwargs):
        started.set()
        assert finish.wait(5)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(transcription_job.subprocess, "run", fake_run)
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
    monkeypatch.setattr(transcription_job.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    imported = []
    job = transcription_job.LocalTranscriptionJob(lambda *args: imported.append(args))
    job._run(video, None)
    assert job.status()["state"] == "failed"
    assert not imported
    assert video.read_bytes() == b"video"
