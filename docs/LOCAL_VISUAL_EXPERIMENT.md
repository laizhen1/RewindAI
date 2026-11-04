# Experimental local visual retrieval

This prototype scores an English text query against **one midpoint frame per subtitle segment** using the locally stored [Xenova CLIP ViT-B/32 ONNX model](https://huggingface.co/Xenova/clip-vit-base-patch32). It uses CPU execution, in-memory vectors, and PyAV for local decoding; it does not upload recordings, screenshots or queries. The model card declares the model licence as MIT. The downloaded model is separate from RewindAI's code licence.

The visual search is explicitly opt-in through `--visual-model`; keyword or semantic subtitle search continues to be available separately. Indexing occurs on the first visual query and repeats after the video library changes. The prototype accepts at most 100 segments, handles at most four frames per encoding batch, and indexes all loaded videos. It does not support video or course filtering yet. It may take significant CPU time for long recordings. The 6 GB GPU is not used by this path.

The single sampled frame can miss slides, animations and brief events. CLIP's similarity is not a calibrated probability, and the system does not have a validated rejection threshold: even an absent object returns the highest-scoring frames. A retrieved frame is a candidate for human review, not a verified answer. The image preprocessing uses PyAV bicubic resizing and a center crop with CLIP's published RGB normalization; it is not a claim of bit-exact equivalence to every other CLIP implementation.

## Public-video smoke test

On 4 November 2026, the quantized model revision `d15189d7028b43f1d3e65039190477f6af591c2a` was tested locally on the 97-second [KI-Campus public video](PUBLIC_SAMPLE_ZH.md), with locally generated subtitles forming five segments. On this machine, first index construction took approximately 2.45 seconds and subsequent individual queries approximately 0.024 seconds; these are one-run observations, not a benchmark. The query `A drawing of a hospital` returned the 0–19 second segment first. An unrelated query about a dog also returned candidates, demonstrating the lack of abstention. The selected midpoint frames were inspected locally; this was not an independently annotated visual evaluation and does not provide precision, recall or F1. No course content was used.

The integrated browser flow was also checked using the same public video and an isolated database: the result warning was visible and a selected candidate's frame loaded. These checks establish that the pipeline runs for this input, not that the candidate is relevant.

## Reproduce

Install optional packages with `python -m pip install -e ".[semantic]" "av>=11,<17"` and obtain `tokenizer.json` and `onnx/model_quantized.onnx` from the model repository into a local directory. Model download is a separate user action; the diagnostic and app do not download weights. Run:

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

This saves midpoint PNGs and `labels.json` locally. Frame `frame-000.png` corresponds to segment index 0, etc. Review every sampled frame and add objects to `queries`, for example `{ "query": "A hand holding a pen", "relevant_segment_indices": [3] }`. Use an empty list for a query that matches none of the sampled frames. The index applies to the numbered **sampled frames**, not the entire video. Record who reviewed the labels separately in your own private notes; after independent review, set `review_status` to `reviewed`. SHA-256 hashes bind the labels to the video and subtitle files. Keep the whole review directory private.

```powershell
python scripts/evaluate_clip_frames.py --video PATH_TO_VIDEO --subtitles PATH_TO_SRT --model-dir PATH_TO_LOCAL_CLIP_MODEL --labels artifacts/visual-review/my-video/labels.json
```

The evaluator computes micro frame-level precision, recall and F1 for the top three candidate frames per query. It also reports the number of absent queries that still returned candidates. It refuses unreviewed labels by default; `--allow-unreviewed` explicitly marks provisional diagnostics and does not turn them into ground truth. No threshold is applied. The scoring treats each query-frame pair as positive or negative, and assumes the label list is complete; it is not video-time localization, object detection accuracy or answer correctness.

## Five-frame provisional diagnostic

As a workflow check, five midpoint frames from the public ethics video were visually inspected by the assistant and five queries were drafted afterward, including one absent object. These labels were **not independently reviewed** and the queries were written with knowledge of the frames and some model behavior. Across 25 query–frame pairs, top-three ranking gave TP=9, FP=6, FN=0, TN=10: precision **0.600**, recall **1.000**, F1 **0.750**. The single absent query still returned three candidates. For the hand-and-pen query, the matching frame was third rather than first. This tiny, selectively designed diagnostic is not a benchmark or a portfolio accuracy claim; it shows both the end-to-end scoring workflow and the absence of safe abstention. Source images and draft labels were kept in ignored local artifacts.
