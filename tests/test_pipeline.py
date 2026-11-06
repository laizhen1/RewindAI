import unittest

from video_rag.adapters import EvidenceGenerator, TokenOverlapReranker
from video_rag.pipeline import VideoRAGPipeline
from video_rag.retrieval import InMemoryLexicalRetriever
from video_rag.schemas import VideoSegment


def make_pipeline(minimum_score=0.01):
    pipeline = VideoRAGPipeline(
        retrievers=[InMemoryLexicalRetriever()],
        reranker=TokenOverlapReranker(),
        generator=EvidenceGenerator(),
        minimum_rerank_score=minimum_score,
    )
    pipeline.build(
        [
            VideoSegment(
                segment_id="v1_0000",
                video_id="v1",
                source_path="v1.mp4",
                start_time=0,
                end_time=20,
                transcript="飞机在机场附近降落",
            ),
            VideoSegment(
                segment_id="v2_0000",
                video_id="v2",
                source_path="v2.mp4",
                start_time=0,
                end_time=20,
                transcript="厨师正在制作蛋糕",
            ),
        ]
    )
    return pipeline


class PipelineTests(unittest.TestCase):
    def test_scoped_search_checks_past_global_top_k(self):
        pipeline = make_pipeline()
        pipeline._recall_top_k = 1
        result = pipeline.ask("飞机蛋糕", video_id="v2")
        self.assertFalse(result.abstained)
        self.assertTrue(all(e.segment.video_id == "v2" for e in result.evidence))

    def test_pipeline_returns_structured_evidence(self):
        result = make_pipeline().ask("飞机在哪里降落")
        self.assertFalse(result.abstained)
        self.assertEqual(result.evidence[0].segment.segment_id, "v1_0000")
        self.assertIn("飞机在机场附近降落", result.answer)
        self.assertNotIn("v1_0000", result.answer)
        self.assertGreaterEqual(result.latency_ms["total"], 0)

    def test_pipeline_abstains_for_unrelated_question(self):
        result = make_pipeline().ask("量子计算机")
        self.assertTrue(result.abstained)
        self.assertEqual(result.evidence, ())

    def test_weak_candidates_are_not_returned_or_given_to_generator(self):
        pipeline = make_pipeline(minimum_score=0.5)

        class FixedReranker:
            def score(self, query, segments):
                return [0.9 if item.segment_id == "v1_0000" else 0.1 for item in segments]

        class CaptureGenerator:
            def __init__(self):
                self.seen = []

            def generate(self, query, segments):
                self.seen = segments
                return "answer"

        capture = CaptureGenerator()
        pipeline._reranker = FixedReranker()
        pipeline._generator = capture
        result = pipeline.ask("飞机蛋糕")
        self.assertEqual([item.segment.segment_id for item in result.evidence], ["v1_0000"])
        self.assertEqual([item.segment_id for item in capture.seen], ["v1_0000"])
