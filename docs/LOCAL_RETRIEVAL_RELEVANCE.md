# Local segment-relevance review and retrieval comparison

This development/validation workflow evaluates text, sampled video frames and RRF on the same questions. It does not establish answer correctness, hallucination reduction, a reranker improvement, or a final test result. Keep course videos, subtitles, labels, segment IDs, fingerprints and reports on the local computer.

1. Save a video-bound CSV of questions using the local [time-labeling workflow](REWINDAI_EVALUATION_ZH.md). Install the documented PyAV and local model dependencies. Use a separate video for each split and avoid selecting thresholds or prompts on the final test video.
2. Start the local relevance reviewer. It shows all transcript segments in video order, without retrieval rankings. Watch the original recording and select every segment that directly supports each answer. Time overlap by itself is not relevance.

   ```bash
   python scripts/review_retrieval_relevance.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --labels PATH_TO_LABELS_CSV --video-id VIDEO_ID --split validation --output-dir artifacts/evaluation/relevance
   ```

   Open `http://127.0.0.1:5098/`. Save progress and restore the latest matching draft as needed. Confirm individual relevance only after reviewing each selected segment. Declare the list exhaustive only if the entire video was checked for that question; an incomplete label set can overestimate recall. These are user declarations, not independent verification. Unanswerable questions are excluded and need a separate evaluation. Changed video, subtitles, question CSV or segmentation requires new labels.
3. Once every answerable question has a verified, exhaustively declared set, run the comparison using already downloaded local Qwen3-Embedding GGUF and CLIP ONNX files:

   ```bash
   python scripts/evaluate_local_segment_retrieval.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --labels PATH_TO_LABELS_CSV --gold PATH_TO_SAVED_RELEVANCE_JSON --video-id VIDEO_ID --split validation --semantic-model PATH_TO_QWEN3_GGUF --visual-model PATH_TO_CLIP_DIRECTORY --output artifacts/evaluation/local-segment-comparison.json --confirm-reviewed
   ```

The script runs on CPU and compares unthresholded top-five Qwen3 text retrieval, CLIP frame retrieval, and RRF with `k=60`. It expects the local Qwen3-Embedding-0.6B GGUF file; its hash is recorded, but the tool does not independently establish the model's identity. Qwen3 document input is truncated to 512 tokens, and CLIP queries can be truncated to 77 tokens. It reports macro Recall@1/5 (fraction of labeled relevant segments recovered per question), Hit@1/5 (at least one hit), MRR@1/5 and per-type counts. CLIP samples one midpoint for at most 100 segments; an unsampled relevant segment cannot be found by that route. This offline top-five experiment differs from the workspace's thresholded top-three answer pipeline. The script records local input, model and code fingerprints and selected dependency versions, but these do not verify the annotations or capture every runtime detail. Output is restricted to `artifacts/` or `local_data/` and must not be shared when derived from private lectures.

To compare a reranker, add `--reranker-model PATH_TO_LOCAL_QWEN3_RERANKER_DIRECTORY`. Install compatible local PyTorch, Transformers and Accelerate dependencies and obtain the public model separately after checking its license. The script uses CPU and local files only, scores up to 20 Qwen3 text candidates in batches of four with a 1024-token limit, and reports the top five alongside the original text ranking. Both see the same subtitle text; the reranked route considers a deeper candidate pool, which is necessary for Recall@5 to change. Model files are fingerprinted but their identity is not independently verified. CPU inference can take substantial time and memory; no reranker model is bundled or automatically downloaded.

## Declared frozen test

After choosing settings on separate videos, prepare a `test` CSV and relevance draft without viewing test retrieval rankings. Once every answerable test question and all relevant segments are reviewed, freeze the exact inputs and configuration. Provide **every** development/validation relevance file used during tuning via repeated `--other-gold` flags:

```bash
python scripts/freeze_local_segment_test.py --video PATH_TO_TEST_VIDEO --subtitles PATH_TO_TEST_SRT --labels PATH_TO_TEST_CSV --gold PATH_TO_TEST_RELEVANCE_JSON --video-id TEST_VIDEO_ID --semantic-model PATH_TO_QWEN3_GGUF --visual-model PATH_TO_CLIP_DIRECTORY --other-gold PATH_TO_DEVELOPMENT_RELEVANCE_JSON --other-gold PATH_TO_VALIDATION_RELEVANCE_JSON --output artifacts/evaluation/frozen-segment-test.json --confirm-reviewed --confirm-test-isolation
```

Include the same `--reranker-model` directory at freeze and evaluation if using a reranker. To run the test, use the comparison command above with `--split test`, the test inputs, the same repeated `--other-gold` files, and `--frozen-test-manifest artifacts/evaluation/frozen-segment-test.json`. The evaluator checks inputs, segment structure, model files, selected code, settings and dependency versions before and after scoring. It checks that the **recorded** video hashes differ, but cannot verify that all tuning datasets were supplied or detect near-duplicate videos, overlapping content, incorrect labels, non-independent review, repeated test runs, or changes later reverted. The declaration and manifest are not tamper-proof. Report these limitations and avoid tuning on the test results.

The legacy [retrieval experiment](../data/evaluation/README.md) also has a reranker route, but its text input may include visual descriptions and its model revision is not fully pinned; it must not be presented as a pure-subtitle or verified lecture result without further work.
