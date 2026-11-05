# Experimental local visual retrieval

This prototype scores an English text query against **one midpoint frame per subtitle segment** using the locally stored [Xenova CLIP ViT-B/32 ONNX model](https://huggingface.co/Xenova/clip-vit-base-patch32). It uses CPU execution, in-memory vectors, and PyAV for local decoding; it does not upload recordings, screenshots or queries. The model card declares the model licence as MIT. The downloaded model is separate from RewindAI's code licence.

The visual search is explicitly opt-in through `--visual-model`; keyword or semantic subtitle search continues to be available separately. The comparison view displays the two lists separately without generating an answer. An additional experimental workspace option fuses their candidate ranks with RRF; neither view establishes that a candidate is relevant. If visual search fails, the workspace retains the transcript result and displays the failure. Indexing occurs on the first visual query and repeats after the video library changes. The prototype indexes at most 100 segments in the selected scope and handles at most four frames per encoding batch. For larger scopes it deterministically spreads the budget across videos, retaining at least one segment per video, and samples evenly within each video; the interface reports how many were selected. Unselected segments cannot be retrieved, so coverage is incomplete. More than 100 videos in one scope requires narrowing the scope. It supports all loaded videos, the selected video, or that video’s assigned course. It keeps only the most recently used visual index in memory, so switching scope can trigger reindexing. It may take significant CPU time for long recordings. The 6 GB GPU is not used by this path.

The single sampled frame can miss slides, animations and brief events. CLIP's similarity is not a calibrated probability, and the system does not have a validated rejection threshold: even an absent object returns the highest-scoring frames. A retrieved frame is a candidate for human review, not a verified answer. The image preprocessing uses PyAV bicubic resizing and a center crop with CLIP's published RGB normalization; it is not a claim of bit-exact equivalence to every other CLIP implementation.

## Public-video smoke test

On 4 November 2026, the quantized model revision `d15189d7028b43f1d3e65039190477f6af591c2a` was tested locally on the 97-second [KI-Campus public video](PUBLIC_SAMPLE_ZH.md), with locally generated subtitles forming five segments. On this machine, first index construction took approximately 2.45 seconds and subsequent individual queries approximately 0.024 seconds; these are one-run observations, not a benchmark. The query `A drawing of a hospital` returned the 0–19 second segment first. An unrelated query about a dog also returned candidates, demonstrating the lack of abstention. The selected midpoint frames were inspected locally; this was not an independently annotated visual evaluation and does not provide precision, recall or F1. No course content was used.

The integrated browser flow was also checked using the same public video and an isolated database: the result warning was visible and a selected candidate's frame loaded. These checks establish that the pipeline runs for this input, not that the candidate is relevant.

## Reproduce

Install optional packages with `python -m pip install -e ".[semantic]" "av>=11,<17"` and obtain `tokenizer.json` and `onnx/model_quantized.onnx` from the model repository into a local directory. Model download is a separate user action; the diagnostic and app do not download weights. For example, from the project root, install `huggingface_hub` and explicitly download only these public model files (approximately 154 MB) into the ignored project cache:

```powershell
python -m pip install huggingface_hub
$env:HF_HUB_DISABLE_TELEMETRY = "1"
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Xenova/clip-vit-base-patch32', revision='d15189d7028b43f1d3e65039190477f6af591c2a', cache_dir='artifacts/hf-cache/hub', allow_patterns=['tokenizer.json','onnx/model_quantized.onnx']))"
```

This network request downloads the public model; it does not send any lecture file. Windows users can then double-click `Start-RewindAI-Visual.cmd`, which uses that pinned cache path. Set `REWINDAI_VISUAL_MODEL_DIR` if the model is stored elsewhere. Alternatively run:

```powershell
python scripts/diagnose_clip_visual.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --model-dir PATH_TO_LOCAL_CLIP_MODEL --question "A drawing of a hospital"
python scripts/run_transcript.py --visual-model PATH_TO_LOCAL_CLIP_MODEL
```

Use authorized videos only. For private lectures, keep videos, frames, transcripts, indexes, local queries and evaluation records on the computer and out of public repositories or external model services. To establish visual precision and recall, build and independently review a held-out set of visual and unanswerable queries with time-aligned video evidence before tuning a threshold.

## Local frame-label evaluation

Create a new local review directory under `artifacts/` or `local_data/` (the command refuses to overwrite existing directories):

```powershell
python scripts/prepare_clip_labels.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --output-dir artifacts/visual-review/my-video
```

This saves midpoint PNGs and `labels.json` locally. Open the local review interface to inspect and annotate the sampled frames:

```powershell
python scripts/visual_review_server.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --review-dir artifacts/visual-review/my-video
```

Visit `http://127.0.0.1:5094/` on the same computer. Enter a query, select **all** matching frames, or explicitly mark that none match, then save. Check all queries before marking the file reviewed. Editing a reviewed file makes it a draft again. The status is a reviewer declaration, not proof that the reviewer was independent or that the labels are correct. Reopening the page retains saved work; concurrent edits require reloading rather than silently overwriting each other. Close the server window when done.

Alternatively edit `labels.json` yourself: frame `frame-000.png` corresponds to index 0, and a query is `{ "query": "A hand holding a pen", "relevant_segment_indices": [3] }`. Use an empty list if none of the sampled frames match. Indices refer to **sampled frames**, not the whole video. Record reviewer identity separately in private notes. SHA-256 hashes bind labels to the video and subtitle files, and are checked when opening the review and running evaluation. Keep the whole review directory private.

```powershell
python scripts/evaluate_clip_frames.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --model-dir PATH_TO_LOCAL_CLIP_MODEL --labels artifacts/visual-review/my-video/labels.json
```

The evaluator computes micro frame-level precision, recall and F1 for the top three candidate frames per query. It also reports the number of absent queries that still returned candidates. It refuses unreviewed labels by default; `--allow-unreviewed` explicitly marks provisional diagnostics and does not turn them into ground truth. No threshold is applied. The scoring treats each query-frame pair as positive or negative, and assumes the label list is complete; it is not video-time localization, object detection accuracy or answer correctness.

## Five-frame provisional diagnostic

As a workflow check, five midpoint frames from the public ethics video were visually inspected by the assistant and five queries were drafted afterward, including one absent object. These labels were **not independently reviewed** and the queries were written with knowledge of the frames and some model behavior. Across 25 query–frame pairs, top-three ranking gave TP=9, FP=6, FN=0, TN=10: precision **0.600**, recall **1.000**, F1 **0.750**. The single absent query still returned three candidates. For the hand-and-pen query, the matching frame was third rather than first. This tiny, selectively designed diagnostic is not a benchmark or a portfolio accuracy claim; it shows both the end-to-end scoring workflow and the absence of safe abstention. Source images and draft labels were kept in ignored local artifacts.

## Bounded sampling latency smoke test

On 4 November 2026, the same public 97-second video was divided into 105 **synthetic 0.8-second segments**; the visual sampler selected 100. With the local quantized CLIP model on CPU, one run took **21.34 seconds** to load the model and build the index, followed by **0.023 seconds** for one query and three returned candidates. This checks the 100-frame path on this machine; closely spaced synthetic frames are not a representative hour-long lecture. Peak system memory, GPU use, audio transcription, visual accuracy and user-perceived browser latency were not measured. Different decoding formats, resolution and hardware can materially change the time.

```powershell
python scripts/benchmark_clip_sampling.py --video PATH_TO_VIDEO --model-dir PATH_TO_LOCAL_CLIP_MODEL --count 105 --step 0.8
```

The whole synthetic time range must fit inside the video. The benchmark reads the video locally, does not save frames, and is for timing rather than evaluation.


## Paired subtitle versus frame diagnostic

To compare both routes on the **same** locally labeled frames and questions, run:

```powershell
python scripts/compare_frame_retrieval.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --model-dir PATH_TO_LOCAL_CLIP_MODEL --labels PATH_TO_LOCAL_LABELS_JSON
```

The subtitle side defaults to the same keyword pipeline as the workspace; pass `--semantic-model PATH_TO_LOCAL_MINILM` to use the experimental semantic pipeline instead, with `--semantic-threshold` defaulting to the uncalibrated 0.4. The visual side uses local CLIP. Both are scored against exactly the same midpoint-frame labels and return at most three candidate segments. The comparison also reports an experimental fusion of the top-three ranked lists using reciprocal rank fusion with k=60. It measures whether each route finds segments whose **sampled frame** was labeled relevant; it does not measure transcript relevance, video-wide relevance, generated answers or answer accuracy. The fusion does not use raw score magnitudes or have a calibrated rejection rule. The script refuses more than 100 segments so that the compared candidate sets stay aligned. Source SHA-256 hashes are checked before and after evaluation; results contain queries and must stay local for private material. It refuses draft labels unless `--allow-unreviewed` is explicit.

For an integration diagnostic on 4 November 2026, the assistant examined five sampled frames from the public ethics sample and wrote four visual queries afterward, one with no matching frame. These draft labels were not independently reviewed, and the questions were chosen after seeing the images. The subtitle route returned no candidates: TP=0, FP=0, FN=6; precision is undefined, recall **0**, F1 **0**. The visual route returned TP=5, FP=7, FN=1; precision **0.417**, recall **0.833**, F1 **0.556**. The one absent query returned visual candidates. This hand-picked four-query smoke test cannot support a portfolio accuracy claim or a general comparison: the questions favor visible content, the labels are subjective, and only one midpoint per segment was inspected.

For these frame metrics, precision is undefined when there are no predicted positives; recall is undefined when there are no labeled positives. F1 is calculated as `2TP/(2TP+FP+FN)`: it is 0 for missed positives with no predictions, and undefined if there are no predicted or labeled positives. The labels are assumed complete only for the sampled frames.


### Experimental rank-fusion result on the same draft

The paired diagnostic now includes a third route: reciprocal rank fusion of the subtitle and visual top-three rankings (k=60, equal weight), followed by a top-three cutoff. On the four draft queries described above, subtitle retrieval returned no candidates, so the fused ranking exactly matched the visual ranking: TP=5, FP=7, FN=1; precision **0.417**, recall **0.833**, F1 **0.556**. The absent query again returned three candidates. This shows no fusion benefit on the sample and does not test whether agreement between sources improves ranking. It does not establish safety for absent queries. This uncalibrated fusion is an **opt-in workspace experiment**; independently reviewed audio, visual, multimodal and absent questions are needed before deciding whether it helps.

Using the optional MiniLM semantic transcript route at threshold 0.4 on the same four draft visual queries also returned no transcript candidates, so the fused and visual frame scores remained identical. This repeats the same tiny, visually selected diagnostic and is not a separate benchmark; it provides no evidence that fusion improves ranking.

## Comparing timestamp candidates with transcript-only search

The local `compare_temporal_multimodal.py` command evaluates transcript-only candidates and the workspace's experimental transcript-plus-frame RRF candidates against the same time-labeled questions. It uses the same top-three limit and at most 100 sampled midpoint frames as the workspace, never generates answers, and records sample coverage and visual-index build time. Keyword search is the default; add `--semantic-model PATH_TO_LOCAL_MINILM` for local English semantic search with the uncalibrated 0.4 threshold. The script groups results by question type and reports positive temporal hits and negative candidate rates separately. A top-one hit requires a returned segment to cover at least half the reference time interval by default; this does not verify that the frame, transcript, or answer actually supports the question.

`temporal_recall_at_3` is the fraction of answerable questions for which any of up to three candidates covers at least the specified fraction of the one labeled reference interval. It is a time-based proxy, not visual relevance Recall@3: other valid intervals may be unlabeled, and matching a time does not establish that a sampled frame or transcript answers the question. Negative questions are excluded from this recall denominator and reported separately.

The non-aggregate local report includes each candidate's timestamp and whether transcript search, visual search or both produced it. It also counts paired abstention changes on unanswerable questions. Sources identify retrieval routes, not verified support; question text and detailed results must remain local for private recordings.

To inspect a non-aggregate report beside the same source video on the same computer, run `python scripts/review_fusion_comparison.py --video PATH_TO_VIDEO --report PATH_TO_LOCAL_REPORT_JSON`, then open `http://127.0.0.1:5098/`. Reference and candidate buttons seek to their start times. The page initially checks the video's fingerprint against the report and rechecks the video size and modification time and report hash during use; a video replacement preserving both metadata values can evade the later check. The page does not upload course material. Older reports may lack a reference interval or route labels; re-run the comparison to include them. A time match still does not prove that the content supports an answer.

For a development-only sensitivity experiment, append `--visual-min-score 0.25` (or another value in [-1, 1]) to exclude visual candidates whose cosine similarity is below that value before fusion. This cutoff is uncalibrated and can discard relevant evidence. It applies to the evaluator only: the workspace currently uses no visual cutoff. Compare both negative candidate rates and positive temporal recall, freeze any selected setting before an independent validation, and do not choose a threshold using the final test split. Candidate scores are not probabilities.

```powershell
python scripts/compare_temporal_multimodal.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --questions PATH_TO_LOCAL_QUESTIONS_JSON --visual-model PATH_TO_LOCAL_CLIP --semantic-model PATH_TO_LOCAL_MINILM --aggregate-only --output artifacts/evaluation/paired_temporal.json
```

Only development and validation drafts should be used for iteration. A test CSV is accepted only with a separately declared frozen-test manifest as described below. Labels and source binding are not independently verified, and the model cannot reliably reject absent visual content. Reports, questions and fingerprints can contain sensitive information and must remain local. Index build latency excludes model loading; query latency is sequential and excludes indexing. The sampled frames can omit relevant visual evidence.

The same `--questions` argument also accepts a CSV produced by the local temporal labeler. Add `--video-id YOUR_VIDEO_ID --split development` (or `validation`). The adjacent `.csv.source.json` sidecar must match the selected video, ID and split; it does not bind the subtitles used while the labels were written. The report records that limitation and fingerprints the inputs before and after evaluation. For JSON input, source binding is reported separately when available. The output is always a new JSON file under `artifacts/` or `local_data/`.

### Frozen test after development

Do not use the test video or its questions to choose a threshold or prompt. After freezing the design on separate development/validation videos and actually reviewing every test question, reference and checked scope against the test video, a reviewer can explicitly create a private manifest:

```powershell
python scripts/freeze_temporal_test.py --video PATH_TO_TEST_VIDEO --subtitles PATH_TO_SRT --questions PATH_TO_TEST_CSV --video-id TEST_VIDEO_ID --visual-model PATH_TO_LOCAL_CLIP --semantic-model PATH_TO_LOCAL_MINILM --output artifacts/evaluation/frozen-test.json --confirm-labels-reviewed
```

The CSV must be labeled `test` and have the local labeler's matching video sidecar. The manifest records hashes of the video, current subtitles, labels, sidecar, model files and selected implementation files, as well as dependency versions and the chosen thresholds. Current subtitle hashes do not prove those subtitles were used when annotating. The command checks reference ranges against video duration and requires a whole-video-check declaration for each negative question. The flag is a human declaration, not proof of independent review, correct labels or video-disjoint splits; near-duplicate or shared-content videos still require manual checking. Manifest hashes and results are private.

Run the same fixed settings and inputs using `compare_temporal_multimodal.py` with `--questions PATH_TO_TEST_CSV --video-id TEST_VIDEO_ID --split test --frozen-test-manifest artifacts/evaluation/frozen-test.json`, plus the same video, subtitles, model paths and thresholds used when freezing. Provide a new output under `artifacts/` or `local_data/`. The script verifies input hashes, listed implementation files, runtime versions and settings before and after the run; changes reject the run. The checks do not cover every possible dependency, cannot detect a change reverted between checks, and do not prevent someone from making a new manifest or repeatedly tuning after seeing results. Report those limitations and avoid representing self-declared labels as independent review.

As a smoke test, the assistant ran this command on the public ethics video and nine previously explored subtitle-derived draft questions, using generated subtitles, local MiniLM and CLIP. The questions were unbound to those subtitle files and not independently reviewed; none was designed to establish visual understanding. Transcript-only yielded temporal F1 0.500 and returned candidates on 2 of 6 draft negative questions. Fusion yielded temporal F1 0.333 and returned candidates on all 6 negative questions; both methods had 2 top-one temporal hits among 3 positive questions. The sample is too small and biased for performance claims. In particular, a timestamp match does not establish visual relevance or factual answer correctness.

An exploratory rerun on these same drafts with evaluation-only `--visual-min-score 0.25` returned candidates for 2 of the 6 negatives and gave temporal F1 0.500. However, its time-based Recall@3 was 2/3, compared with 3/3 for transcript-only search on the same inputs. The unfiltered fusion also covered 2/3 at this cutoff. This tiny, previously inspected sample does not validate the 0.25 threshold or establish any general reduction in false positives. The workspace remains unfiltered.
