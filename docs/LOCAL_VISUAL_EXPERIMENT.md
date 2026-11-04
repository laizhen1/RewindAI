# Experimental local visual retrieval

This prototype scores an English text query against **one midpoint frame per subtitle segment** using the locally stored [Xenova CLIP ViT-B/32 ONNX model](https://huggingface.co/Xenova/clip-vit-base-patch32). It uses CPU execution, in-memory vectors, and PyAV for local decoding; it does not upload recordings, screenshots or queries. The model card declares the model licence as MIT. The downloaded model is separate from RewindAI's code licence.

The visual search is explicitly opt-in through `--visual-model`; keyword or semantic subtitle search continues to be available separately. The workspace also offers a comparison view: it runs the transcript and visual searches on the same query and scope, then displays the results in separate sections. It does not fuse scores, assert agreement, or generate a multimodal answer. If visual search fails, it retains the transcript result and displays the failure. Indexing occurs on the first visual query and repeats after the video library changes. The prototype indexes at most 100 segments in the selected scope and handles at most four frames per encoding batch. For larger scopes it deterministically spreads the budget across videos, retaining at least one segment per video, and samples evenly within each video; the interface reports how many were selected. Unselected segments cannot be retrieved, so coverage is incomplete. More than 100 videos in one scope requires narrowing the scope. It supports all loaded videos, the selected video, or that video’s assigned course. It keeps only the most recently used visual index in memory, so switching scope can trigger reindexing. It may take significant CPU time for long recordings. The 6 GB GPU is not used by this path.

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

The subtitle side uses the same keyword pipeline as the default workspace; the visual side uses local CLIP. Both are scored against exactly the same midpoint-frame labels and return at most three candidate segments. This measures whether each route finds segments whose **sampled frame** was labeled relevant; it does not measure transcript relevance, video-wide relevance, a fused audio-visual system, generated answers or answer accuracy. The script refuses more than 100 segments so that the compared candidate sets stay aligned. Source SHA-256 hashes are checked before and after evaluation; results contain queries and must stay local for private material. It refuses draft labels unless `--allow-unreviewed` is explicit.

For an integration diagnostic on 4 November 2026, the assistant examined five sampled frames from the public ethics sample and wrote four visual queries afterward, one with no matching frame. These draft labels were not independently reviewed, and the questions were chosen after seeing the images. The subtitle route returned no candidates: TP=0, FP=0, FN=6; precision is undefined, recall **0**, F1 **0**. The visual route returned TP=5, FP=7, FN=1; precision **0.417**, recall **0.833**, F1 **0.556**. The one absent query returned visual candidates. This hand-picked four-query smoke test cannot support a portfolio accuracy claim or a general comparison: the questions favor visible content, the labels are subjective, and only one midpoint per segment was inspected.

For these frame metrics, precision is undefined when there are no predicted positives; recall is undefined when there are no labeled positives. F1 is calculated as `2TP/(2TP+FP+FN)`: it is 0 for missed positives with no predictions, and undefined if there are no predicted or labeled positives. The labels are assumed complete only for the sampled frames.
