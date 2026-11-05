# RewindAI

**English** | [中文](README.zh-CN.md)

**Find it. Verify it. Learn it.**

RewindAI is a local learning workspace for finding and reviewing moments in lecture videos. Search video transcripts, inspect timestamped evidence, return to the source, and save useful clips with notes.

> **Project status:** Under development and evaluation. The CPU demo uses preset text and simplified retrieval; it does not demonstrate real video understanding. The full multimodal pipeline requires separate model setup and independent evaluation. The workspace is intended for local use and should not be deployed directly to the public internet.

## Features

- Organize indexed videos into courses; rename or delete courses without deleting videos or notes.
- Search videos or a course and inspect timestamped evidence. Subtitle mode can display neighboring transcript segments and, with PyAV installed, a midpoint frame for manual review.
- Save clips, edit review notes, filter saved clips by video name or note, and provide separate feedback on answers and evidence. Notes and feedback stay in the local database; filtering happens in the browser.

- Add local videos with SRT/VTT subtitles, or transcribe an authorized video locally with a cached Whisper model. The interface displays processing and failure states.
- Try optional local English semantic search and experimental visual search. A comparison view shows their results separately.
- Explore the full pipeline's speech, keyframe, hybrid retrieval, reranking and evaluation components; the complete model workflow still needs independent end-to-end validation.

While the page remains open, an unsaved note is retained when the bookmark list refreshes. Select **Save note** to store it. The browser may warn before a reload or close when notes are unsaved or saving; drafts are still lost if the page closes, and browsers may suppress this warning.

## Current limitations

The app does not automatically download Canvas recordings. By default, subtitle search returns source excerpts using keyword matching and does not generate answers or understand video frames. An opt-in local answer generator is experimental. Semantic and visual search are experimental. In particular, visual search can return candidates even when nothing is relevant.

The local library stores video paths, subtitle copies, course assignments, bookmarks and feedback. Unchanged videos are restored on restart. If a source moves or changes, affected searches and playback are blocked; restore the file or remove and rebuild its index. Original media and notes are preserved, although bookmarks for an old subtitle revision may no longer be playable. Updating an external subtitle file does not update the saved copy automatically; use the workspace subtitle update action to rebuild the index. Video changes are detected using size and modification time, not a content hash, and already buffered media may continue playing. If an evidence start time exceeds video duration, the player warns about a possible mismatch; this check cannot validate other timestamps.

Subtitle and experimental visual searches recheck the source after processing; a detected change discards the results. Frame previews are similarly rechecked. These lightweight checks cannot detect changes that preserve both file size and modification time, or changes after the final check.

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

Some tests are skipped when optional dependencies such as FAISS or PyAV are not installed. Passing the lightweight test suite does not validate the full multimodal model workflow.

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

Untimed `.txt` transcripts cannot provide reliable video timestamps and are not imported directly. Use a timed SRT/VTT file or generate one with local transcription. Generating timestamps does not establish transcript accuracy; check the result against the recording.

### Review inaccurate subtitles locally

The local subtitle reviewer lets you play individual cues, correct their text, and export a new SRT draft without modifying the input. It uses no hosted model or external service:

```bash
python scripts/review_subtitles.py --video /path/to/lecture.mp4 --subtitles /path/to/lecture.srt
```

Open `http://127.0.0.1:5096/` on the same computer. Search the cues, listen to the source, make corrections, and export to `artifacts/corrected_subtitles/`. Each export is a new file. To compare an untimed UTF-8 TXT transcript, add `--alternative /path/to/transcript.txt`. The optional differences filter narrows the list to cues with aligned word changes. Word alignment is approximate, can shift or truncate text, and differences do not show which version is correct; neither version is a reference. Listen to the video before correcting. Only edited cues have been changed; all cues can still be wrong, and this tool does not adjust timestamps. To continue after closing the page, restart with the exported SRT as the input. To search the corrections, select the saved video in the workspace, enter the new SRT path and choose **Update current video subtitles**. Exact matching time ranges keep their bookmarks; other bookmarks retain their notes but may no longer be playable. Startup-only videos must first be added to the local library. Keep the exported captions private. The reviewer normalizes subtitle presentation to plain text and SRT.

For video without subtitles, install faster-whisper and prepare a local cached model as described in [local transcription](docs/LOCAL_TRANSCRIPTION.md). Enter the video path, choose a language, and select **Transcribe locally and load** in the workspace. One job runs at a time on CPU; the interface remains in Chinese. Transcription can be cancelled before indexing starts. A completed transcript is kept under `artifacts/transcripts/` and a copy is saved in the local library. The process does not initiate model downloads or send media to a hosted speech API. Model weights must already be cached; obtaining them initially requires an explicit download. The workflow has been checked on two short public videos, but lecture performance remains unverified.

Windows users can also run `Start-RewindAI-Transcribe.cmd`. An unchanged saved video opens the workspace without retranscription; remove its index first to replace subtitles. A forced service shutdown can bypass best-effort child-process cleanup; see the transcription guide for details.

### Experimental local AI answers

The optional answer mode uses a locally installed `llama-cpp-python` and an existing GGUF model to answer from retrieved transcript excerpts on CPU. It makes no network request for inference or model download. Answers can be wrong, fabricated, or affected by instructions in the subtitles; the retrieved excerpts and timestamps are candidates, not proof. A search miss does not show that the whole video lacks an answer. Check the source video. This is text-only, not visual understanding, and it has not been evaluated on human-reviewed lectures.

Install a CPU wheel using the [llama-cpp-python instructions](https://github.com/abetlen/llama-cpp-python#installation), prepare a GGUF model locally, and start:

```bash
python scripts/run_transcript.py --answer-model /path/to/model.gguf
```

On Windows, `Start-RewindAI-Local-Answers.cmd` looks for the locally cached Qwen2.5 3B Q4_K_M revision `7dabda4d13d513e3e842b20f0d435c732f172cbe`; `REWINDAI_ANSWER_MODEL` overrides the path. It does not download a model. Check the [model license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) separately. This model and the smaller 0.5B were compared on 12 hand-written synthetic cases: the 3B answered six simple supported questions and declined six unsupported questions, while the 0.5B answered the supported questions and fabricated answers on five unsupported questions. These exploratory results are not a benchmark or a course F1 score, and performance can vary. On a separate 50-example, previously explored public snippet sample, a strict refusal-string proxy had recall 0.040 and F1 0.077; this measures snippet answerability rather than answer correctness. A subsequent public-video draft test exposed an unsupported diabetes-treatment claim despite a caveat; see the [local answer evaluation](docs/LOCAL_ANSWER_EXPERIMENT.md) and local human-review page and scoring tool.

### Experimental English semantic search

Install `python -m pip install -e ".[semantic]"` and place the `tokenizer.json` and `onnx/model_quantized.onnx` files for [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) in a local model directory:

```powershell
python scripts/run_transcript.py --semantic-model PATH_TO_LOCAL_MODEL_DIRECTORY
```

On Windows, `Start-RewindAI-Semantic.cmd` looks for revision `08a308f628bc9d6774b7922f319eb1b65afa1a82` in the project cache; set `REWINDAI_SEMANTIC_MODEL_DIR` for another location. The launcher does not download weights. Inference runs locally on CPU. The English-only mode uses an uncalibrated cosine threshold of 0.4 and may return unrelated segments. Without the separately enabled local answer mode, it does not generate answers; it does not understand frames. See the [evaluation report](docs/INITIAL_EVALUATION.md); the default mode remains keyword search.

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

To collect time-labeled drafts locally without editing a CSV by hand, install PyAV (`pip install -e ".[asr]"`) and run:

```powershell
python scripts/temporal_label_server.py --video PATH_TO_LOCAL_VIDEO --subtitles PATH_TO_LOCAL_SRT --video-id lecture_dev --split development --output artifacts/evaluation/lecture-dev-drafts.csv
```

Open `http://127.0.0.1:5095/` on the same computer. Play the video, enter a question and reference answer, and mark the evidence start and end. For an unanswerable question, the page requires a declaration that the entire video was checked. Entries are **drafts**, not proof of independent review; the page does not verify that you watched the material. With subtitles, development and validation drafts can be evaluated locally using keyword retrieval. The diagnostic counts the first result as correct when it covers at least 50% of the reference interval; a wrong result on an answerable question counts as both a false positive and a false negative. Its Precision, Recall and F1 measure this timestamp rule, not answer correctness. The page reserves the test split and cannot evaluate it. Review labels and freeze settings before the separate final evaluation.

The CSV and its source-binding sidecar contain private data and must remain local. The sidecar binds the file to the video's SHA-256 at startup; during the session, changes are detected using size and modification time, which is not a full integrity guarantee. The diagnostic fingerprints the video and subtitles before and after running; this does not prove the subtitles were the ones used when labeling. Use a separate output for each video and split, and run only one labeling server per CSV. The page can add and edit its own drafts but cannot delete them; records with additional review data must be edited externally with the server stopped. Stop the server before any external CSV editing. The data can be converted with the documented local evaluation tools, subject to independent review and split checks.

To compare the same development or validation questions with experimental semantic retrieval, add `--semantic-model PATH_TO_LOCAL_MINILM` and install `pip install -e ".[semantic]"`. The model must already be present locally; inference runs on CPU without downloading inputs or weights. This English-only text comparison uses the existing, uncalibrated 0.4 similarity cutoff. Both methods search subtitles rather than video frames, and the resulting draft metrics are not answer accuracy or evidence of a visual improvement. Model files are fingerprinted before and after the run. If the model is unavailable, restart without this option to use keyword search alone.

Optionally select **Show per-question timestamps locally** before running the diagnostic. The page shows each saved question, its reference interval and the returned intervals, with buttons to seek the local video. These cases can help inspect failures, but a time match still does not prove that a result answers the question. Question details and diagnostic results are private and are not automatically exported; protect the computer and do not share screenshots of course content.

Select **Save private summary locally** before running to write a new JSON file in a `reports/` folder beside the annotation CSV. The report includes aggregate counts, diagnostic settings and input fingerprints, but not question text or individual cases. It remains an unverified draft and may still contain sensitive metadata; keep it local. The server checks the input fingerprints and annotation revision again before writing, and does not overwrite an existing report.

For the development split only, **Browse development excerpts** shows up to 12 transcript segments sampled across the segment order and lets you jump to the video. You can copy a sampled time range into the draft form, then check and adjust it against the video. These are starting points, not generated questions or verified labels; transcript errors and choosing questions from transcripts can bias the resulting development set. The option is unavailable for validation and test. Check the original audio and video when writing questions.

In an exploratory TutorialVQA subset of 18 instructional videos and 294 questions, reference-sentence overlap gave keyword Precision / Recall / F1 of **0.394 / 0.095 / 0.153**, and experimental English semantic retrieval **0.388 / 0.129 / 0.194**. These numbers measure transcript sentence overlap, not video timestamps or answer correctness. The dataset has no unanswerable questions, and the subset was included in earlier exploratory analysis; it is not a pristine final test set.

On a separate YTCommentQA sample of 284 annotated instructional-video snippets, treating any keyword retrieval candidate as a positive answerability prediction produced precision / recall / F1 of **0.939 / 0.413 / 0.573**, with candidates returned for **6 of 61** snippets labeled unanswerable. On the same questions, experimental semantic retrieval produced **0.945 / 0.614 / 0.745**, with candidates on **8 of 61** unanswerable snippets. This is a snippet-level heuristic comparison, not answer accuracy or whole-video performance; see the [method and exclusions](docs/INITIAL_EVALUATION.md#ytcommentqa-independent-snippet-level-retrieval-proxy).

A separate exploratory search over the provided full transcripts measured top-three overlap with annotated time windows on **151 script-answerable questions**: keyword **50 / 151 (0.331)** and semantic **80 / 151 (0.530)**. Any temporal overlap counts as a hit; requiring the union of candidates to cover at least half the reference window yields **16 / 151** and **20 / 151**, respectively. Reference windows can include extra context, 77 transcripts were excluded for invalid cues, and no answer or source video was checked; see the [full method and sensitivity run](docs/INITIAL_EVALUATION.md#searching-the-provided-transcript-for-annotated-time-windows).

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
