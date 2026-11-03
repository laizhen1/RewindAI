import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from video_rag.api import create_app
from video_rag.transcript_baseline import (
    build_transcript_library,
    build_transcript_pipeline,
    build_transcript_segments,
    find_subtitles,
    parse_subtitles,
    segments_from_subtitles,
)


class TranscriptBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.video = self.base / "lecture.mp4"
        self.video.write_bytes(b"fake video bytes for API test")
        self.srt = self.base / "lecture.srt"
        self.srt.write_text(
            "1\n00:00:02,000 --> 00:00:05,250\nAttention maps queries to keys.\n\n"
            "2\n00:00:06,000 --> 00:00:09,000\nThe model combines values.\n\n"
            "3\n00:00:30,000 --> 00:00:34,000\nTraining uses gradient descent.\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_srt_grouping_uses_actual_cue_times_and_validates_auto_sidecar(self):
        self.assertEqual(find_subtitles(self.video), self.srt)
        segments = segments_from_subtitles(self.video, self.srt)
        self.assertEqual(len(segments), 2)
        self.assertEqual((segments[0].start_time, segments[0].end_time), (2, 9))
        self.assertIn("Attention", segments[0].transcript)
        self.assertEqual((segments[1].start_time, segments[1].end_time), (30, 34))
        self.video.with_suffix(".vtt").write_text("WEBVTT\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "choose one"):
            find_subtitles(self.video)

    def test_vtt_comments_formatting_and_optional_hours(self):
        cues = parse_subtitles(
            "\ufeffWEBVTT\n\nNOTE a comment\nignore this\n\n"
            "cue-1\n00:01.500 --> 00:05.000 line:90%\n<v Lecturer>Hello &amp; welcome</v>\n\n"
            "00:01:05.000 --> 00:01:08.000\n<c.green>Second cue</c>\n"
        )
        self.assertEqual([(cue.start_time, cue.end_time) for cue in cues], [(1.5, 5), (65, 68)])
        self.assertEqual(cues[0].text, "Hello & welcome")

    def test_invalid_subtitles_fail_closed(self):
        for value in (
            "1\n00:00:70,000 --> 00:01:10,000\ninvalid",
            "1\n00:00:08,000 --> 00:00:03,000\ninvalid",
            "1\nnot-a-time --> also-invalid\ninvalid",
            "WEBVTT\n\nNOTE no cues",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_subtitles(value)
        self.srt.write_bytes(b"\xff\xfe\xff")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            segments_from_subtitles(self.video, self.srt)

    def test_local_api_search_and_video_response(self):
        pipeline = build_transcript_pipeline(self.video, self.srt)
        app = create_app(pipeline, database_path=self.base / "state.sqlite3", mode="transcript")
        client = app.test_client()
        self.assertEqual(client.get("/api/health").get_json()["mode"], "transcript")
        video = client.get("/api/library").get_json()["videos"][0]
        self.assertTrue(video["available"])
        response = client.post("/api/ask", json={"question": "What maps queries to keys?"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("Attention", data["answer"])
        self.assertEqual(data["evidence"][0]["start_time"], 2)
        self.assertNotIn("source_path", data["evidence"][0])
        self.assertEqual(client.get(data["evidence"][0]["video_url"]).data, b"fake video bytes for API test")
        missing = client.post("/api/ask", json={"question": "quantum entanglement"}).get_json()
        self.assertTrue(missing["abstained"])

    def test_empty_workspace_imports_and_restores_without_startup_video(self):
        db = self.base / "empty.sqlite3"
        client = create_app(build_transcript_segments([]), database_path=db, mode="transcript").test_client()
        self.assertEqual(client.get("/api/library").get_json()["videos"], [])
        self.assertEqual(client.post("/api/ask", json={"question": "attention"}).status_code, 503)
        self.assertEqual(client.post("/api/library/local-video", json={"video_path": str(self.video)}).status_code, 201)
        self.assertFalse(client.post("/api/ask", json={"question": "attention"}).get_json()["abstained"])
        reopened = create_app(build_transcript_segments([]), database_path=db, mode="transcript").test_client()
        self.assertEqual(len(reopened.get("/api/library").get_json()["videos"]), 1)
        self.assertFalse(reopened.post("/api/ask", json={"question": "attention"}).get_json()["abstained"])

    def test_removing_loaded_video_keeps_media_and_notes_allows_new_subtitles(self):
        db = self.base / "remove.sqlite3"
        client = create_app(build_transcript_segments([]), database_path=db, mode="transcript").test_client()
        video_id = client.post("/api/library/local-video", json={"video_path": str(self.video)}).get_json()["video_id"]
        evidence = client.post("/api/ask", json={"question": "attention"}).get_json()["evidence"][0]
        old_id = evidence["segment_id"]
        client.post("/api/bookmarks", json={"video_id": video_id, "segment_id": old_id, "note": "keep me"})
        self.assertEqual(client.delete(f"/api/library/videos/{video_id}").status_code, 204)
        self.assertTrue(self.video.exists())
        self.assertTrue(self.srt.exists())
        saved = client.get("/api/bookmarks").get_json()["bookmarks"][0]
        self.assertEqual(saved["note"], "keep me")
        self.assertFalse(saved["available"])
        self.assertEqual(client.delete(f"/api/library/videos/{video_id}").status_code, 404)
        self.srt.write_text("1\n00:00:03,000 --> 00:00:06,000\nAttention is updated.\n", encoding="utf-8")
        self.assertEqual(client.post("/api/library/local-video", json={"video_path": str(self.video)}).status_code, 201)
        new = client.post("/api/ask", json={"question": "attention"}).get_json()["evidence"][0]
        self.assertNotEqual(new["segment_id"], old_id)
        self.assertFalse(client.get("/api/bookmarks").get_json()["bookmarks"][0]["available"])
        reopened = create_app(build_transcript_segments([]), database_path=db, mode="transcript").test_client()
        self.assertEqual(len(reopened.get("/api/library").get_json()["videos"]), 1)

    def test_removing_command_line_video_and_demo_rejection(self):
        pipeline = build_transcript_pipeline(self.video, self.srt)
        video_id = pipeline.videos()[0]["id"]
        client = create_app(pipeline, database_path=self.base / "cli.sqlite3", mode="transcript").test_client()
        self.assertEqual(client.delete(f"/api/library/videos/{video_id}").status_code, 204)
        self.assertEqual(client.get("/api/library").get_json()["videos"], [])
        demo = create_app(pipeline, database_path=self.base / "demo2.sqlite3", mode="demo").test_client()
        self.assertEqual(demo.delete(f"/api/library/videos/{video_id}").status_code, 409)

    def test_failed_rebuild_does_not_forget_video(self):
        def fail_empty(segments):
            if not segments:
                raise ValueError("failed to build")
            return build_transcript_segments(segments)
        db = self.base / "fail.sqlite3"
        client = create_app(build_transcript_segments([]), database_path=db, mode="transcript",
                            transcript_builder=fail_empty).test_client()
        video_id = client.post("/api/library/local-video", json={"video_path": str(self.video)}).get_json()["video_id"]
        self.assertEqual(client.delete(f"/api/library/videos/{video_id}").status_code, 500)
        self.assertEqual(client.get("/api/library").get_json()["videos"][0]["id"], video_id)
        reopened = create_app(build_transcript_segments([]), database_path=db, mode="transcript").test_client()
        self.assertEqual(reopened.get("/api/library").get_json()["videos"][0]["id"], video_id)

    def test_multiple_videos_are_separate_and_search_can_be_scoped(self):
        other_video = self.base / "second.mp4"
        other_video.write_bytes(b"other")
        other_srt = self.base / "second.srt"
        other_srt.write_text("1\n00:00:03,000 --> 00:00:06,000\nAttention uses vectors.\n", encoding="utf-8")
        pipeline = build_transcript_library([(self.video, self.srt), (other_video, other_srt)])
        videos = pipeline.videos()
        self.assertEqual(len(videos), 2)
        target = next(video["id"] for video in videos if video["title"] == "second.mp4")
        result = pipeline.ask("attention", video_id=target)
        self.assertFalse(result.abstained)
        self.assertTrue(all(e.segment.video_id == target for e in result.evidence))
        with self.assertRaisesRegex(ValueError, "only be loaded once"):
            build_transcript_library([(self.video, self.srt), (self.video, other_srt)])

    def test_course_search_filters_evidence_and_handles_empty_or_invalid_scope(self):
        other = self.base / "other.mp4"
        other.write_bytes(b"other video")
        captions = other.with_suffix(".srt")
        captions.write_text("1\n00:00:03,000 --> 00:00:06,000\nAttention uses vectors.\n", encoding="utf-8")
        app = create_app(build_transcript_library([(self.video, self.srt), (other, captions)]),
                         database_path=self.base / "courses.sqlite3", mode="transcript")
        client = app.test_client()
        videos = client.get("/api/library").get_json()["videos"]
        first = next(video["id"] for video in videos if video["title"] == "lecture.mp4")
        second = next(video["id"] for video in videos if video["title"] == "other.mp4")
        course = client.post("/api/courses", json={"name": "Attention"}).get_json()["id"]
        self.assertEqual(client.post("/api/ask", json={"question": "attention", "course_id": course}).status_code, 400)
        self.assertEqual(client.put(f"/api/videos/{first}/course", json={"course_id": course}).status_code, 200)
        response = client.post("/api/ask", json={"question": "attention", "course_id": course})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["evidence"])
        self.assertTrue(all(item["video_id"] == first for item in response.get_json()["evidence"]))
        self.assertEqual(client.put(f"/api/videos/{second}/course", json={"course_id": course}).status_code, 200)
        response = client.post("/api/ask", json={"question": "vectors", "course_id": course})
        self.assertEqual(response.get_json()["evidence"][0]["video_id"], second)
        self.assertEqual(client.post("/api/ask", json={"question": "attention", "course_id": 999}).status_code, 404)
        self.assertEqual(client.post("/api/ask", json={"question": "attention", "course_id": True}).status_code, 400)
        self.assertEqual(client.post("/api/ask", json={"question": "attention", "course_id": course, "video_id": first}).status_code, 400)

    def test_local_import_adds_video_without_damaging_existing_index(self):
        pipeline = build_transcript_pipeline(self.video, self.srt)
        app = create_app(pipeline, database_path=self.base / "import.sqlite3", mode="transcript")
        client = app.test_client()
        other = self.base / "other.mp4"
        other.write_bytes(b"another video")
        captions = other.with_suffix(".srt")
        captions.write_text("1\n00:00:03,000 --> 00:00:06,000\nEmbeddings represent meaning.\n", encoding="utf-8")
        first_id = client.get("/api/library").get_json()["videos"][0]["id"]
        response = client.post("/api/library/local-video", json={"video_path": str(other)})
        self.assertEqual(response.status_code, 201)
        new_id = response.get_json()["video_id"]
        self.assertEqual(len(client.get("/api/library").get_json()["videos"]), 2)
        self.assertEqual(client.post("/api/library/local-video", json={"video_path": str(other)}).status_code, 409)
        bad = client.post("/api/library/local-video", json={"video_path": str(self.base / "missing.mp4")})
        self.assertEqual(bad.status_code, 400)
        result = client.post("/api/ask", json={"question": "embeddings", "video_id": new_id}).get_json()
        self.assertFalse(result["abstained"])
        self.assertEqual(result["evidence"][0]["video_id"], new_id)
        old_segment = result["evidence"][0]["segment_id"]
        saved = client.post("/api/bookmarks", json={"video_id": new_id, "segment_id": old_segment, "note": "Keep this note"})
        self.assertEqual(saved.status_code, 201)
        self.assertTrue(client.get("/api/bookmarks").get_json()["bookmarks"][0]["available"])
        self.assertEqual(client.delete(f"/api/library/unavailable-video/{new_id}").status_code, 404)
        self.assertTrue(client.get(f"/api/videos/{first_id}").status_code == 200)
        demo = create_app(pipeline, database_path=self.base / "demo.sqlite3", mode="demo").test_client()
        self.assertEqual(demo.post("/api/library/local-video", json={"video_path": str(other)}).status_code, 409)
        self.assertEqual(demo.delete(f"/api/library/unavailable-video/{new_id}").status_code, 409)

        # Recreate the app from the initial source to simulate a restart.
        resumed = create_app(build_transcript_pipeline(self.video, self.srt),
                             database_path=self.base / "import.sqlite3", mode="transcript").test_client()
        self.assertEqual(len(resumed.get("/api/library").get_json()["videos"]), 2)
        self.assertFalse(resumed.post("/api/ask", json={"question": "embeddings", "video_id": new_id}).get_json()["abstained"])
        # A replacement video at the same path must not silently reuse stale evidence.
        other.write_bytes(b"modified video content")
        changed = create_app(build_transcript_pipeline(self.video, self.srt),
                             database_path=self.base / "import.sqlite3", mode="transcript").test_client()
        data = changed.get("/api/library").get_json()
        self.assertEqual(len(data["videos"]), 1)
        self.assertEqual(data["unavailable_saved_videos"], 1)
        self.assertEqual(data["unavailable_videos"], [{"id": new_id, "title": "other.mp4"}])
        self.assertEqual(changed.post("/api/ask", json={"question": "embeddings", "video_id": new_id}).status_code, 404)
        self.assertFalse(changed.get("/api/bookmarks").get_json()["bookmarks"][0]["available"])
        self.assertEqual(changed.delete(f"/api/library/unavailable-video/{new_id}").status_code, 204)
        self.assertTrue(other.exists())
        self.assertEqual(changed.get("/api/library").get_json()["unavailable_saved_videos"], 0)
        self.assertEqual(changed.post("/api/library/local-video", json={"video_path": str(other)}).status_code, 201)
        new_result = changed.post("/api/ask", json={"question": "embeddings", "video_id": new_id}).get_json()
        self.assertNotEqual(new_result["evidence"][0]["segment_id"], old_segment)
        bookmark = changed.get("/api/bookmarks").get_json()["bookmarks"][0]
        self.assertEqual(bookmark["note"], "Keep this note")
        self.assertFalse(bookmark["available"])

    def test_generic_question_words_and_partial_content_do_not_produce_answers(self):
        self.srt.write_text(
            "1\n00:00:01,000 --> 00:00:05,000\nThis is about attention and queries.\n\n"
            "2\n00:00:10,000 --> 00:00:14,000\n这是关于机器学习和数据处理的课程。\n\n"
            "3\n00:00:30,000 --> 00:00:34,000\n接下来介绍CLIP模型的作用。\n",
            encoding="utf-8",
        )
        pipeline = build_transcript_pipeline(self.video, self.srt)
        self.assertTrue(pipeline.ask("What is quantum entanglement?").abstained)
        self.assertTrue(pipeline.ask("Why is the sky blue?").abstained)
        self.assertTrue(pipeline.ask("What is attention entanglement?").abstained)
        self.assertTrue(pipeline.ask("什么是量子纠缠？").abstained)
        self.assertFalse(pipeline.ask("What is attention?").abstained)
        self.assertFalse(pipeline.ask("机器学习").abstained)
        self.assertFalse(pipeline.ask("什么是CLIP模型？").abstained)

    def test_windows_cli_help_runs_from_outside_checkout_without_pythonpath(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-I", str(root / "scripts" / "run_transcript.py"), "--help"],
            cwd=self.base, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--subtitles", result.stdout)


if __name__ == "__main__":
    unittest.main()
