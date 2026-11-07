import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_rag.api import create_app
from video_rag.schemas import SearchHit, VideoSegment
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

    def test_plain_text_transcript_explains_timed_subtitle_requirement(self):
        plain = self.base / "lecture.txt"
        plain.write_text("An untimed transcript", encoding="utf-8")
        client = create_app(build_transcript_segments([]), database_path=self.base / "text.sqlite3", mode="transcript").test_client()
        response = client.post("/api/library/local-video", json={"video_path": str(self.video), "subtitle_path": str(plain)})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Plain text transcripts are not supported; use timed SRT/VTT or transcribe locally")
        self.assertEqual(client.get("/api/library").get_json()["videos"], [])

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
        partial = client.get(data["evidence"][0]["video_url"], headers={"Range": "bytes=5-9"})
        self.assertEqual(partial.status_code, 206)
        self.assertEqual(partial.data, b"video")
        self.assertEqual(partial.headers["Content-Range"], "bytes 5-9/29")
        partial.close()
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

    def test_restore_detects_video_change_during_index_rebuild(self):
        db = self.base / "restore-race.sqlite3"
        client = create_app(build_transcript_segments([]), database_path=db, mode="transcript").test_client()
        video_id = client.post("/api/library/local-video", json={"video_path": str(self.video)}).get_json()["video_id"]

        def builder(segments):
            self.video.write_bytes(b"changed while the saved index is rebuilding")
            return build_transcript_segments(segments)

        reopened = create_app(build_transcript_segments([]), database_path=db, mode="transcript",
                              transcript_builder=builder).test_client()
        video = reopened.get("/api/library").get_json()["videos"][0]
        self.assertEqual(video["id"], video_id)
        self.assertFalse(video["available"])
        self.assertEqual(reopened.post("/api/ask", json={"question": "attention"}).status_code, 409)
        self.assertEqual(reopened.get(f"/api/videos/{video_id}").status_code, 409)

    def test_import_errors_identify_video_and_keep_existing_index(self):
        def builder(segments):
            if len({segment.video_id for segment in segments}) > 1:
                raise RuntimeError("internal detail")
            return build_transcript_segments(segments)
        db = self.base / "import_failure.sqlite3"
        client = create_app(build_transcript_segments([]), database_path=db, mode="transcript", transcript_builder=builder).test_client()
        missing = client.post("/api/library/local-video", json={"video_path": str(self.base / "missing.mp4")})
        self.assertEqual(missing.status_code, 400)
        self.assertIn("Video must be an existing", missing.get_json()["error"])
        self.assertEqual(client.post("/api/library/local-video", json={"video_path": str(self.video)}).status_code, 201)
        other = self.base / "next.mp4"
        other.write_bytes(b"video")
        other.with_suffix(".srt").write_text("1\n00:00:01,000 --> 00:00:03,000\nexample\n", encoding="utf-8")
        response = client.post("/api/library/local-video", json={"video_path": str(other)})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("internal detail", response.get_json()["error"])
        self.assertEqual(len(client.get("/api/library").get_json()["videos"]), 1)
        reopened = create_app(build_transcript_segments([]), database_path=db, mode="transcript").test_client()
        self.assertEqual(len(reopened.get("/api/library").get_json()["videos"]), 1)

    def test_import_does_not_commit_if_video_changes_during_index_build(self):
        other = self.base / "next.mp4"
        other.write_bytes(b"next video")
        other.with_suffix(".srt").write_text("1\n00:00:01,000 --> 00:00:03,000\nexample\n", encoding="utf-8")

        def builder(segments):
            if len({segment.video_id for segment in segments}) > 1:
                other.write_bytes(b"changed video during indexing")
            return build_transcript_segments(segments)

        db = self.base / "changed-during-import.sqlite3"
        client = create_app(build_transcript_segments([]), database_path=db, mode="transcript",
                            transcript_builder=builder).test_client()
        self.assertEqual(client.post("/api/library/local-video", json={"video_path": str(self.video)}).status_code, 201)
        response = client.post("/api/library/local-video", json={"video_path": str(other)})
        self.assertEqual(response.status_code, 409)
        self.assertIn("changed while importing", response.get_json()["error"])
        self.assertEqual(len(client.get("/api/library").get_json()["videos"]), 1)
        self.assertFalse(client.post("/api/ask", json={"question": "attention"}).get_json()["abstained"])
        reopened = create_app(build_transcript_segments([]), database_path=db, mode="transcript").test_client()
        self.assertEqual(len(reopened.get("/api/library").get_json()["videos"]), 1)

    def test_import_does_not_commit_if_video_disappears_during_index_build(self):
        def builder(segments):
            self.video.unlink()
            return build_transcript_segments(segments)

        db = self.base / "disappeared-during-import.sqlite3"
        client = create_app(build_transcript_segments([]), database_path=db, mode="transcript",
                            transcript_builder=builder).test_client()
        response = client.post("/api/library/local-video", json={"video_path": str(self.video)})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(client.get("/api/library").get_json()["videos"], [])
        reopened = create_app(build_transcript_segments([]), database_path=db, mode="transcript").test_client()
        self.assertEqual(reopened.get("/api/library").get_json()["videos"], [])

    def test_changed_video_is_not_used_for_search_playback_or_frames(self):
        other = self.base / "other.mp4"
        other.write_bytes(b"other video")
        other_subs = other.with_suffix(".srt")
        other_subs.write_text("1\n00:00:01,000 --> 00:00:04,000\nunique oranges\n", encoding="utf-8")
        pipeline = build_transcript_library([(self.video, self.srt), (other, other_subs)])
        client = create_app(pipeline, database_path=self.base / "changed-live.sqlite3", mode="transcript",
                            visual_builder=lambda segments: None).test_client()
        original_id = next(v["id"] for v in client.get("/api/library").get_json()["videos"] if v["title"] == self.video.name)
        other_id = next(v["id"] for v in client.get("/api/library").get_json()["videos"] if v["title"] == other.name)
        evidence = client.post("/api/ask", json={"question": "attention", "video_id": original_id}).get_json()["evidence"][0]
        segment_id = evidence["segment_id"]
        client.post("/api/bookmarks", json={"video_id": original_id, "segment_id": segment_id, "note": "keep"})
        self.video.write_bytes(b"replacement video bytes that differ")
        videos = {v["id"]: v for v in client.get("/api/library").get_json()["videos"]}
        self.assertFalse(videos[original_id]["available"])
        self.assertTrue(videos[other_id]["available"])
        self.assertFalse(client.get("/api/bookmarks").get_json()["bookmarks"][0]["available"])
        self.assertEqual(client.post("/api/ask", json={"question": "attention"}).status_code, 409)
        self.assertEqual(client.post("/api/ask", json={"question": "attention", "video_id": original_id}).status_code, 409)
        self.assertEqual(client.get(f"/api/videos/{original_id}").status_code, 409)
        self.assertEqual(client.get(f"/api/segments/{segment_id}/context").status_code, 409)
        self.assertEqual(client.get(f"/api/segments/{segment_id}/frame").status_code, 409)
        self.assertEqual(client.post("/api/visual-search", json={"question": "image", "video_id": original_id}).status_code, 409)
        self.assertEqual(client.post("/api/compare-search", json={"question": "attention", "video_id": original_id}).status_code, 409)
        unaffected = client.post("/api/ask", json={"question": "oranges", "video_id": other_id})
        self.assertEqual(unaffected.status_code, 200)
        self.assertFalse(unaffected.get_json()["abstained"])
        self.assertEqual(client.delete(f"/api/library/videos/{original_id}").status_code, 204)
        self.assertEqual(client.post("/api/ask", json={"question": "oranges"}).status_code, 200)
        self.assertEqual(client.get("/api/bookmarks").get_json()["bookmarks"][0]["note"], "keep")

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

    def test_context_is_adjacent_and_never_crosses_video_boundary(self):
        other_video = self.base / "second.mp4"
        other_video.write_bytes(b"other")
        other_srt = self.base / "second.srt"
        other_srt.write_text("1\n00:00:03,000 --> 00:00:06,000\nOther video.\n", encoding="utf-8")
        pipeline = build_transcript_library([(self.video, self.srt), (other_video, other_srt)])
        client = create_app(pipeline, database_path=self.base / "context.sqlite3", mode="transcript").test_client()
        own = segments_from_subtitles(self.video, self.srt)
        first = client.get(f"/api/segments/{own[0].segment_id}/context").get_json()["context"]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["relation"], "next")
        self.assertEqual(first[0]["segment_id"], own[1].segment_id)
        self.assertNotIn("source_path", first[0])
        last = client.get(f"/api/segments/{own[1].segment_id}/context").get_json()["context"]
        self.assertEqual([item["relation"] for item in last], ["previous"])
        self.assertEqual(client.get("/api/segments/missing/context").status_code, 404)
        demo = create_app(pipeline, database_path=self.base / "context-demo.sqlite3", mode="demo").test_client()
        self.assertEqual(demo.get(f"/api/segments/{own[0].segment_id}/context").status_code, 409)

    @unittest.skipUnless(importlib.util.find_spec("av"), "requires optional PyAV")
    def test_frame_preview_only_uses_indexed_video_and_hides_paths(self):
        pipeline = build_transcript_pipeline(self.video, self.srt)
        client = create_app(pipeline, database_path=self.base / "frame.sqlite3", mode="transcript").test_client()
        segment = pipeline.segments()[0]
        with patch("video_rag.api.frames.extract_frame_png", return_value=(b"png-data", 5.5)) as extract:
            result = client.get(f"/api/segments/{segment.segment_id}/frame")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.mimetype, "image/png")
        self.assertEqual(result.headers["Cache-Control"], "no-store")
        extract.assert_called_once_with(segment.source_path, 5.5)
        with patch("video_rag.api.frames.extract_frame_png", side_effect=ValueError(str(self.video))):
            failed = client.get(f"/api/segments/{segment.segment_id}/frame")
        self.assertEqual(failed.status_code, 422)
        self.assertNotIn(str(self.video), failed.get_data(as_text=True))
        self.assertEqual(client.get("/api/segments/missing/frame").status_code, 404)
        demo = create_app(pipeline, database_path=self.base / "frame-demo.sqlite3", mode="demo").test_client()
        self.assertEqual(demo.get(f"/api/segments/{segment.segment_id}/frame").status_code, 409)

    def test_experimental_visual_search_is_opt_in_and_invalidated_after_index_change(self):
        builds = []
        class FakeVisual:
            def __init__(self, segments):
                self.segments = segments
            def search(self, question, limit):
                return [SearchHit(self.segments[0].segment_id, 0.2, "visual", 1)]
        def builder(segments):
            builds.append(segments)
            return FakeVisual(segments)
        db = self.base / "visual.sqlite3"
        app = create_app(build_transcript_segments([]), database_path=db, mode="transcript", visual_builder=builder)
        client = app.test_client()
        self.assertTrue(client.get("/api/health").get_json()["visual_enabled"])
        self.assertEqual(client.post("/api/visual-search", json={"question": "image"}).status_code, 400)
        added = client.post("/api/library/local-video", json={"video_path": str(self.video)}).get_json()
        response = client.post("/api/visual-search", json={"question": "image"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("source_path", response.get_json()["evidence"][0])
        self.assertEqual(len(builds), 1)
        client.post("/api/visual-search", json={"question": "another"})
        self.assertEqual(len(builds), 1)
        self.assertEqual(client.delete(f"/api/library/videos/{added['video_id']}").status_code, 204)
        client.post("/api/library/local-video", json={"video_path": str(self.video)})
        client.post("/api/visual-search", json={"question": "again"})
        self.assertEqual(len(builds), 2)
        off = create_app(build_transcript_segments([]), database_path=self.base / "off.sqlite3", mode="transcript").test_client()
        self.assertFalse(off.get("/api/health").get_json()["visual_enabled"])
        self.assertEqual(off.post("/api/visual-search", json={"question": "image"}).status_code, 409)

    def test_visual_search_scopes_and_rebuilds_after_course_assignment_changes(self):
        other = self.base / "other.mp4"
        other.write_bytes(b"other video")
        captions = other.with_suffix(".srt")
        captions.write_text("1\n00:00:03,000 --> 00:00:06,000\nDifferent video.\n", encoding="utf-8")
        pipeline = build_transcript_library([(self.video, self.srt), (other, captions)])
        builds = []
        class FakeVisual:
            def __init__(self, segments):
                self.segments = segments
            def search(self, question, limit):
                return [SearchHit(segment.segment_id, 0.2, "visual", i + 1)
                        for i, segment in enumerate(self.segments[:limit])]
        def builder(segments):
            builds.append(segments)
            return FakeVisual(segments)
        client = create_app(pipeline, database_path=self.base / "visual-scopes.sqlite3", mode="transcript",
                            visual_builder=builder).test_client()
        videos = {v["title"]: v["id"] for v in client.get("/api/library").get_json()["videos"]}
        course = client.post("/api/courses", json={"name": "Class"}).get_json()["id"]
        self.assertEqual(client.post("/api/visual-search", json={"question": "image", "course_id": course}).status_code, 400)
        client.put(f"/api/videos/{videos['lecture.mp4']}/course", json={"course_id": course})
        data = client.post("/api/visual-search", json={"question": "image", "course_id": course}).get_json()
        self.assertTrue(data["evidence"])
        self.assertTrue(all(item["video_id"] == videos["lecture.mp4"] for item in data["evidence"]))
        client.put(f"/api/videos/{videos['other.mp4']}/course", json={"course_id": course})
        data = client.post("/api/visual-search", json={"question": "image", "course_id": course}).get_json()
        self.assertEqual({item["video_id"] for item in data["evidence"]}, set(videos.values()))
        self.assertEqual(len(builds), 2)
        data = client.post("/api/visual-search", json={"question": "image", "video_id": videos["other.mp4"]}).get_json()
        self.assertEqual({item["video_id"] for item in data["evidence"]}, {videos["other.mp4"]})
        self.assertEqual(client.post("/api/visual-search", json={"question": "x", "video_id": "missing"}).status_code, 404)
        self.assertEqual(client.post("/api/visual-search", json={"question": "x", "course_id": True}).status_code, 400)
        self.assertEqual(client.post("/api/visual-search", json={"question": "x", "course_id": course,
                                                                  "video_id": videos["other.mp4"]}).status_code, 400)

    def test_visual_sampling_is_reported_and_selected_scope_remains_precise(self):
        segments = [VideoSegment(f"a{i}", "large", str(self.video), float(i), float(i + 1), transcript="a")
                    for i in range(101)]
        segments.append(VideoSegment("small-0", "small", str(self.video), 0.0, 1.0, transcript="b"))
        class FakeVisual:
            def search(self, question, limit):
                return [SearchHit("small-0", 0.2, "visual", 1)]
        client = create_app(build_transcript_segments(segments), database_path=self.base / "limit.sqlite3",
                            mode="transcript", visual_builder=lambda values: FakeVisual()).test_client()
        large_scope = client.post("/api/visual-search", json={"question": "image"}).get_json()
        self.assertTrue(large_scope["sampled"])
        self.assertEqual((large_scope["source_segments"], large_scope["indexed_segments"]), (102, 100))
        response = client.post("/api/visual-search", json={"question": "image", "video_id": "small"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["evidence"][0]["video_id"], "small")
        self.assertFalse(response.get_json()["sampled"])

    def test_compare_search_keeps_modalities_separate_and_handles_visual_failure(self):
        pipeline = build_transcript_pipeline(self.video, self.srt)
        class FakeVisual:
            def __init__(self, segments):
                self.segments = segments
            def search(self, question, limit):
                return [SearchHit(self.segments[-1].segment_id, 0.2, "visual", 1)]
        client = create_app(pipeline, database_path=self.base / "compare.sqlite3", mode="transcript",
                            visual_builder=lambda segments: FakeVisual(segments)).test_client()
        result = client.post("/api/compare-search", json={"question": "attention"})
        self.assertEqual(result.status_code, 200)
        data = result.get_json()
        self.assertFalse(data["transcript"]["abstained"])
        self.assertTrue(data["visual"]["evidence"])
        self.assertNotEqual(data["transcript"]["evidence"][0]["segment_id"],
                            data["visual"]["evidence"][0]["segment_id"])
        self.assertNotIn("source_path", data["visual"]["evidence"][0])
        self.assertEqual(client.post("/api/compare-search", json={"question": "x", "video_id": "missing"}).status_code, 404)
        def fail(_):
            raise ValueError("private local path")
        degraded = create_app(pipeline, database_path=self.base / "compare-error.sqlite3", mode="transcript",
                              visual_builder=fail).test_client()
        partial = degraded.post("/api/compare-search", json={"question": "attention"}).get_json()
        self.assertFalse(partial["transcript"]["abstained"])
        self.assertIsNone(partial["visual"])
        self.assertNotIn("private local path", partial["visual_error"])
        off = create_app(pipeline, database_path=self.base / "compare-off.sqlite3", mode="transcript").test_client()
        self.assertEqual(off.post("/api/compare-search", json={"question": "attention"}).status_code, 409)

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
