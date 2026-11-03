import unittest

from scripts.run_demo import build_demo_pipeline
from video_rag.api import create_app


class ApiTests(unittest.TestCase):
    def test_api_validates_and_returns_answer(self):
        client = create_app(build_demo_pipeline()).test_client()
        self.assertEqual(client.post("/api/ask", json={}).status_code, 400)
        response = client.post("/api/ask", json={"question": "CLIP有什么作用"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("evidence", response.get_json())
        self.assertEqual(client.post("/api/ask", json={"question": "x", "video_id": "missing"}).status_code, 404)
        scoped = client.post("/api/ask", json={"question": "CLIP", "video_id": "demo"})
        self.assertEqual(scoped.status_code, 200)
        self.assertTrue(all(item["video_id"] == "demo" for item in scoped.get_json()["evidence"]))

