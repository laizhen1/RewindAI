import tempfile
import unittest
from pathlib import Path

from scripts.run_demo import build_demo_pipeline
from video_rag.api import create_app


class LibraryApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "library.sqlite3"
        self.app = create_app(build_demo_pipeline(), database_path=self.db, mode="demo")
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def test_courses_describe_only_loaded_videos_and_persist_assignment(self):
        self.assertEqual(self.client.get("/api/health").get_json()["mode"], "demo")
        response = self.client.get("/api/library").get_json()
        video = response["videos"][0]
        self.assertEqual(video["id"], "demo")
        self.assertEqual(video["segment_count"], 2)
        self.assertNotIn("source_path", video)
        course = self.client.post("/api/courses", json={"name": "AI 101"}).get_json()
        self.assertEqual(self.client.put("/api/videos/demo/course", json={"course_id": course["id"]}).status_code, 200)
        reopened = create_app(build_demo_pipeline(), database_path=self.db).test_client()
        self.assertEqual(reopened.get("/api/library").get_json()["videos"][0]["course_id"], course["id"])
        self.assertEqual(self.client.put("/api/videos/other/course", json={"course_id": course["id"]}).status_code, 404)
        self.assertEqual(self.client.put("/api/videos/demo/course", json={"course_id": 999}).status_code, 404)
        self.assertEqual(self.client.put("/api/videos/demo/course", json={"course_id": None}).status_code, 200)
        self.assertIsNone(reopened.get("/api/library").get_json()["videos"][0]["course_id"])

    def test_bookmark_uses_index_times_and_persists_edits(self):
        response = self.client.post("/api/bookmarks", json={
            "video_id": "demo", "segment_id": "demo_0001",
            "start_time": 999, "end_time": 1000, "note": "Review CLIP",
        })
        self.assertEqual(response.status_code, 201)
        item = response.get_json()
        self.assertEqual((item["start_time"], item["end_time"]), (15, 35))
        self.assertEqual(self.client.patch(f"/api/bookmarks/{item['id']}", json={"note": "Updated"}).status_code, 200)
        reopened = create_app(build_demo_pipeline(), database_path=self.db).test_client()
        self.assertEqual(reopened.get("/api/bookmarks").get_json()["bookmarks"][0]["note"], "Updated")
        self.assertEqual(self.client.delete(f"/api/bookmarks/{item['id']}").status_code, 204)
        self.assertEqual(reopened.get("/api/bookmarks").get_json()["bookmarks"], [])
        self.assertEqual(self.client.delete(f"/api/bookmarks/{item['id']}").status_code, 404)

    def test_rename_and_delete_course_keep_video_and_bookmark(self):
        course = self.client.post("/api/courses", json={"name": "Old"}).get_json()["id"]
        self.client.put("/api/videos/demo/course", json={"course_id": course})
        self.client.post("/api/bookmarks", json={"video_id": "demo", "segment_id": "demo_0001", "note": "keep"})
        self.assertEqual(self.client.patch(f"/api/courses/{course}", json={"name": " "}).status_code, 400)
        self.assertEqual(self.client.patch(f"/api/courses/{course}", json={"name": "  New  "}).get_json()["name"], "New")
        self.assertEqual(self.client.delete(f"/api/courses/{course}", headers={"Origin": "https://bad.example"}).status_code, 403)
        self.assertEqual(self.client.delete(f"/api/courses/{course}").status_code, 204)
        reopened = create_app(build_demo_pipeline(), database_path=self.db).test_client()
        data = reopened.get("/api/library").get_json()
        self.assertEqual(data["courses"], [])
        self.assertIsNone(data["videos"][0]["course_id"])
        self.assertEqual(reopened.get("/api/bookmarks").get_json()["bookmarks"][0]["note"], "keep")
        self.assertEqual(self.client.delete(f"/api/courses/{course}").status_code, 404)
        self.assertEqual(self.client.patch(f"/api/courses/{course}", json={"name": "New"}).status_code, 404)

    def test_invalid_evidence_and_feedback_are_rejected(self):
        self.assertEqual(self.client.post("/api/courses", json={"name": " "}).status_code, 400)
        self.assertEqual(self.client.post("/api/ask", json=[]).status_code, 400)
        self.assertEqual(self.client.post("/api/bookmarks", json={
            "video_id": "different", "segment_id": "demo_0001"
        }).status_code, 404)
        self.assertEqual(self.client.post("/api/feedback", json={
            "question": "CLIP?", "rating": "helpful", "segment_id": "missing"
        }).status_code, 404)
        self.assertEqual(self.client.post("/api/feedback", json={
            "question": "CLIP?", "rating": "maybe"
        }).status_code, 400)
        response = self.client.post("/api/feedback", json={
            "question": "CLIP?", "rating": "not_helpful", "segment_id": "demo_0001",
            "comment": "Wrong evidence",
        })
        self.assertEqual(response.status_code, 201)
        self.assertIn("id", response.get_json())
        rows = self.client.get("/api/feedback").get_json()["feedback"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question"], "CLIP?")
        self.assertEqual(rows[0]["segment_id"], "demo_0001")
        self.assertEqual(rows[0]["comment"], "Wrong evidence")
        reopened = create_app(build_demo_pipeline(), database_path=self.db).test_client()
        self.assertEqual(reopened.get("/api/feedback").get_json()["feedback"], rows)
        entry = rows[0]["id"]
        self.assertEqual(self.client.delete(f"/api/feedback/{entry}", headers={"Origin": "https://bad.example"}).status_code, 403)
        self.assertEqual(self.client.delete(f"/api/feedback/{entry}").status_code, 204)
        self.assertEqual(reopened.get("/api/feedback").get_json()["feedback"], [])
        self.assertEqual(self.client.delete(f"/api/feedback/{entry}").status_code, 404)

    def test_ask_does_not_expose_local_media_paths(self):
        response = self.client.post("/api/ask", json={"question": "CLIP有什么作用"})
        self.assertEqual(response.status_code, 200)
        evidence_items = response.get_json()["evidence"]
        self.assertTrue(evidence_items)
        for evidence in evidence_items:
            self.assertNotIn("source_path", evidence)
            for frame in evidence["keyframes"]:
                self.assertNotIn("path", frame)

    def test_workspace_rejects_untrusted_host_and_cross_origin_requests(self):
        self.assertEqual(self.client.get("/api/library", headers={"Host": "attacker.example"}).status_code, 403)
        self.assertEqual(self.client.post(
            "/api/courses", json={"name": "test"},
            headers={"Origin": "https://attacker.example"},
        ).status_code, 403)
        self.assertEqual(self.client.post(
            "/api/courses", json={"name": "test"},
            headers={"Origin": "http://localhost"},
        ).status_code, 201)
        self.assertEqual(self.client.post(
            "/api/courses", json={"name": "test"}, base_url="http://127.0.0.1:5000",
            headers={"Origin": "http://127.0.0.1:5000"},
        ).status_code, 201)
        self.assertEqual(self.client.post("/api/ask", json={"question": "x" * 4001}).status_code, 400)
        self.assertEqual(self.client.post("/api/ask", json={"question": "x" * 70000}).status_code, 413)


if __name__ == "__main__":
    unittest.main()
