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
