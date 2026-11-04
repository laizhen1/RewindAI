# RewindAI

**English** | [中文](README.zh-CN.md)

**Find it. Verify it. Learn it.**

RewindAI is a local learning workspace for finding and reviewing moments in lecture videos. Search video transcripts, inspect timestamped evidence, return to the source, and save useful clips with notes.

> **Project status:** Under development and evaluation. The CPU demo uses preset text and simplified retrieval; it does not demonstrate real video understanding. The full multimodal pipeline requires separate model setup and independent evaluation. The workspace is intended for local use and should not be deployed directly to the public internet.

## Features

- Organize indexed videos into courses; rename or delete courses without deleting videos or notes.
- Search videos or a course and inspect timestamped evidence. Subtitle mode can display neighboring transcript segments and, with PyAV installed, a midpoint frame for manual review.
- Save clips, edit review notes, and provide separate feedback on answers and evidence. Notes and feedback stay in the local database.
- Add local videos with SRT/VTT subtitles, or transcribe an authorized video locally with a cached Whisper model. The interface displays processing and failure states.
- Try optional local English semantic search and experimental visual search. A comparison view shows their results separately.
- Explore the full pipeline's speech, keyframe, hybrid retrieval, reranking and evaluation components; the complete model workflow still needs independent end-to-end validation.

## Current limitations

The app does not automatically download Canvas recordings. Subtitle search returns source excerpts using keyword matching; it does not generate answers or understand video frames. Semantic and visual search are experimental. In particular, visual search can return candidates even when nothing is relevant.

The local library stores video paths, subtitle copies, course assignments, bookmarks and feedback. Unchanged videos are restored on restart. If a source moves or changes, affected searches and playback are blocked; restore the file or remove and rebuild its index. Original media and notes are preserved, although bookmarks for an old subtitle revision may no longer be playable. Updating an external subtitle file does not update the saved copy automatically. Video changes are detected using size and modification time, not a content hash, and already buffered media may continue playing. If an evidence start time exceeds video duration, the player warns about a possible mismatch; this check cannot validate other timestamps.

Search covers all indexed videos by default and can be scoped to the selected video or its course. Scoped search scans the index and may be slow for large collections. The demo does not include real media. Full multimodal mode requires preprocessing and indexes. The server accepts loopback Host values and same-origin browser requests, but has no accounts, multi-user isolation or production authentication; remote use requires an SSH tunnel to the local service.

## Quick start

The CPU demo needs no large model download. From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_demo.py
```

Open `http://127.0.0.1:5000`. The demo uses two preset segments, not lecture recordings; playback requires a real video. On Windows, after installing the environment, `Start-RewindAI-Demo.cmd` starts the demo. Close its command window to stop the service.

Run tests and the documented lint check with:

```bash
python -m pytest -q
python -m ruff check src/video_rag/api scripts/ingest_local.py tests/test_ingest_local.py tests/test_library_api.py
```

## Work with local video

### Subtitle search

On Windows, start `Start-RewindAI-Workspace.cmd`, or run `python scripts/run_transcript.py`, and open `http://127.0.0.1:5000`. Enter paths to an authorized local video and its subtitles in the workspace; previously saved unchanged videos are restored. Alternatively, start with a video and UTF-8 SRT or VTT file:

```bash
python scripts/run_transcript.py --video /path/to/lecture.mp4 --subtitles /path/to/lecture.srt
```

Multiple videos can be supplied in matching order:

```bash
python scripts/run_transcript.py --video lecture1.mp4 --video lecture2.mp4 --subtitles lecture1.srt --subtitles lecture2.srt
```

Omit all `--subtitles` arguments to look for same-name SRT or VTT files. This mode uses local keyword matching and transcript excerpts without downloading a model. It does not guarantee cross-language or semantic retrieval. For on-demand frame previews, install `python -m pip install -e ".[asr]"`; decoding stays local and is not a claim that the frame has been understood.

For video without subtitles, install faster-whisper and prepare a local cached model as described in [local transcription](docs/LOCAL_TRANSCRIPTION.md). Enter the video path, choose a language, and select **Transcribe locally and load** in the workspace. One job runs at a time on CPU; the interface remains in Chinese. Transcription can be cancelled before indexing starts. A completed transcript is kept under `artifacts/transcripts/` and a copy is saved in the local library. The process does not initiate model downloads or send media to a hosted speech API. Model weights must already be cached; obtaining them initially requires an explicit download. The workflow has been checked on two short public videos, but lecture performance remains unverified.

Windows users can also run `Start-RewindAI-Transcribe.cmd`. An unchanged saved video opens the workspace without retranscription; remove its index first to replace subtitles. A forced service shutdown can bypass best-effort child-process cleanup; see the transcription guide for details.

### Experimental English semantic search

Install `python -m pip install -e ".[semantic]"` and place the `tokenizer.json` and `onnx/model_quantized.onnx` files for [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) in a local model directory:

```powershell
python scripts/run_transcript.py --semantic-model PATH_TO_LOCAL_MODEL_DIRECTORY
```

On Windows, `Start-RewindAI-Semantic.cmd` looks for revision `08a308f628bc9d6774b7922f319eb1b65afa1a82` in the project cache; set `REWINDAI_SEMANTIC_MODEL_DIR` for another location. The launcher does not download weights. Inference runs locally on CPU. The English-only mode uses an uncalibrated cosine threshold of 0.4 and may return unrelated segments. It does not generate answers or understand frames. See the [evaluation report](docs/INITIAL_EVALUATION.md); the default mode remains keyword search.

### Experimental local visual search

Optional CLIP search compares an English description with one midpoint frame per subtitle segment on CPU. It requires an existing video and subtitles. Results are candidates for inspection, not verified answers. There is no calibrated no-match threshold: **unrelated queries can still return candidates**. Search can cover the library, one video or one course. At most 100 segments are sampled across the selected scope; unsampled content can be missed, and scopes over 100 videos must be narrowed. The first search and subsequent scope or library changes may require reindexing.

Install `python -m pip install -e ".[semantic]" "av>=11,<17"` and prepare the `tokenizer.json` and `onnx/model_quantized.onnx` files for [Xenova/clip-vit-base-patch32](https://huggingface.co/Xenova/clip-vit-base-patch32) in a local directory. The approximately 154 MB model has its own license.

```powershell
python scripts/run_transcript.py --visual-model PATH_TO_LOCAL_CLIP_MODEL
```

On Windows, `Start-RewindAI-Visual.cmd` looks for cached revision `d15189d7028b43f1d3e65039190477f6af591c2a`; set `REWINDAI_VISUAL_MODEL_DIR` for another location. The launcher does not download the model. In the Chinese interface, enter an English visual description and choose **experimental visual search**, or compare subtitle and visual candidates side by side. The comparison does not fuse scores or generate a multimodal answer; a visual failure does not hide subtitle results. Frames, queries and indexes stay local. See the [visual experiment](docs/LOCAL_VISUAL_EXPERIMENT.md) for setup, labeling, results and limitations.

### Full multimodal pipeline

The full workflow includes video preprocessing, index building and a real model server. The original configuration recommends Ubuntu 22.04, Python 3.11+, FFmpeg, at least 24 GB of NVIDIA VRAM, 64 GB RAM and 100 GB storage; actual requirements depend on the input and models. Review [local video import](docs/LOCAL_VIDEO_IMPORT.md) and `config.toml` before attempting it. Initial dependency or model acquisition may require network access.

```bash
python scripts/ingest_local.py /path/to/authorized-video.mp4 --language en
```

The command writes a new `artifacts/ingestions/` directory without copying the source video. Model-hub offline mode is the default; missing models require explicit download permission. It prints a server command. The full workflow has not been independently validated end to end on real models and course recordings.

Building an index is not model training: indexing generally does not change model weights. Keep course recordings, transcripts, frames, indexes, questions, notes, feedback and course-derived evaluation results on the local computer. Do not upload them to GitHub, cloud models or third parties. `.gitignore` excludes common media, subtitle and database files plus `artifacts/` and `local_data/`, but it cannot identify every private file; inspect changes before publishing.

## Architecture

```text
Video → speech transcript + keyframes / visual descriptions
      → timestamped segments → hybrid retrieval → reranking
Question ──────────────────────────────────────────┘
      → evidence-grounded answer + timestamps → review and bookmarks
```

This diagram describes the intended full pipeline. The CPU demo uses simulated segments, lexical retrieval and simple excerpt assembly; it does not validate full-model quality.

The workspace displays the latest 100 local feedback records and aggregates all feedback by source. Records can include questions, comments and segment IDs. User-submitted feedback may be repeated and is not precision, recall, F1 or ground truth; it must not be published with private course content.

## Evaluation

Candidate questions are diagnostic material, not a fully reviewed gold standard. Review answers and evidence against video, split by video, and separately assess audio, visual, combined and unanswerable questions. Do not present provisional results as verified accuracy.

In an exploratory TutorialVQA subset of 18 instructional videos and 294 questions, reference-sentence overlap gave keyword Precision / Recall / F1 of **0.394 / 0.095 / 0.153**, and experimental English semantic retrieval **0.388 / 0.129 / 0.194**. These numbers measure transcript sentence overlap, not video timestamps or answer correctness. The dataset has no unanswerable questions, and the subset was included in earlier exploratory analysis; it is not a pristine final test set.

Visual retrieval has only a small public-video frame diagnostic with assistant-drafted, independently unreviewed labels. It does not establish reliable visual F1. Private lecture recordings, labels and detailed results must stay local. See:

- [Evaluation plan (Chinese)](docs/REWINDAI_EVALUATION_ZH.md)
- [Initial experiments, TutorialVQA results and reproduction](docs/INITIAL_EVALUATION.md)
- [Annotation template](data/evaluation/rewindai_annotation_template.csv)
- [Experiment record template](data/evaluation/rewindai_experiment_template.md)
- [Product and development plan (Chinese)](docs/REWINDAI_PLAN_ZH.md)
- [Public educational samples and licenses (Chinese)](docs/PUBLIC_SAMPLE_ZH.md)

## License

RewindAI's code is available for non-commercial use, modification and redistribution under the [RewindAI Non-Commercial Use License](LICENSE). Commercial use requires separate written permission. Model weights, datasets, videos and third-party dependencies have their own terms.

## Links

- [RewindAI](https://github.com/laizhen1/RewindAI)
- [Whisper](https://github.com/openai/whisper)
- [FAISS](https://github.com/facebookresearch/faiss)
