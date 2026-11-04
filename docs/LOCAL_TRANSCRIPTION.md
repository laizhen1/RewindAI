# Local transcription with faster-whisper

This optional command creates an SRT file from a video or audio file you are authorized to process. It performs inference on the local CPU; it does not send the recording to a hosted speech API and does not train or fine-tune a model. Subtitles are generated text and can contain mistakes: review them against the recording.

Install the optional dependency in the project environment:

```powershell
python -m pip install -e ".[asr]"
```

Run from the project root:

```powershell
python scripts/transcribe_local.py "C:\path\authorized-lecture.mp4" --language en
```

Choose `--language zh` for Chinese, or omit it for automatic detection. The default is the multilingual `base` model, CPU with int8 computation, and `local_files_only=True`. Required model files must already be cached; the first run with an empty cache will fail without contacting the model hub. To explicitly allow model files to download, add `--allow-model-download`. This download can contact a model provider, but the command does not upload the input media. Existing offline environment settings may still prevent downloads.

On Windows, after installing the ASR dependency and caching the model, double-click `Start-RewindAI-Transcribe.cmd` and choose a video, or drag one video onto that file. The tool automatically detects the language, transcribes on CPU and then starts the local search interface. It never opts into downloading the model. If a project cache exists at `artifacts/hf-cache` and no Hugging Face cache environment variable is set, the launcher uses that project cache; explicit cache settings take precedence. Keep the command window open while using the interface. If the model is absent, follow the explicit download instructions above first. The generated SRT remains in `artifacts/transcripts`. The launcher also saves the local video path and a transcript copy to the local library, so the regular workspace can restore them after restart. If that video is already saved and unchanged, the launcher opens the existing workspace without transcribing again. To replace its subtitles, remove the existing index in the workspace first. If the source has changed, the launcher stops and asks you to restore the original file or remove the unavailable index before re-importing. The pre-check is advisory; concurrent imports are still checked by the server. Original files and review notes are kept.

To transcribe and then open the local subtitle-search interface in one command:

```powershell
python scripts/rewind_local.py "C:\path\authorized-lecture.mp4" --language en
```

If you already have subtitles, provide `--subtitles "C:\path\lecture.srt"` to skip transcription. The launcher creates a fresh local subtitle file for each transcription, never overwrites an existing one, and stops if transcription fails. This mode is keyword search with excerpts, not AI-generated answers or visual analysis.

`--model` accepts another faster-whisper model size or a local CTranslate2-compatible model directory. Local model directories must contain `tokenizer.json`; inspect model licenses and verify compatibility before use. Restricting model-hub access is not a network sandbox for arbitrary third-party libraries.

The default output is `artifacts/transcripts/<input-name>.srt`. The command refuses to overwrite an existing file; choose another `.srt` path inside `artifacts/transcripts` with `--output`. These files may include sensitive information, so keep them out of public repositories. Review the course's copyright, participant privacy, and permissions for transcription before processing. Deleting recordings should include associated subtitles and derived files.

This is a standalone transcription tool: the original `prepare_videos.py` does not automatically consume its SRT output. A separate integration would be required before these subtitles enter the video RAG index.

## Verified local example

On Windows with Python 3.12, faster-whisper 1.2.1 and PyAV 16.1.0, the multilingual `base` model successfully transcribed the 97-second [public educational sample](PUBLIC_SAMPLE_ZH.md) locally on CPU. Its output had 21 cues; the supplied English subtitle had 15. A case-normalized word-sequence comparison produced a similarity of 0.977 for this single sample. This is not a word-error rate or a formal accuracy claim, and neither subtitle has been independently verified against the recording. PyAV 19.0.2 produced an incompatible `metadata_errors` argument error in this setup, so the installation example constrains PyAV below 17.

## Local workflow smoke test

On 4 November 2026, the public AI-medical-ethics sample was transcribed again with the locally cached base model on CPU. A temporary local server started with the resulting SRT returned the 0–19 second segment for the exact phrase `medical professionals`. An initial query, `ethical decisions`, abstained because those words did not occur together under the current keyword rule. This checks the transcription-to-HTTP-retrieval path on one sample, not the launcher file picker, full multimodal pipeline, or search accuracy. No Canvas material was used.


## Transcribe from the local workspace

In subtitle mode, enter an authorized local video path, choose automatic language detection, English or Chinese, and select **本机转录并加载**. The workspace runs the cached `base` model on CPU, then imports the resulting subtitles and refreshes the video list. The task state displays transcription, indexing, completion or failure; it is not a percentage. Only one transcription may run per workspace process. It does not initiate model downloads or use a hosted speech API. Follow the dependency and model-cache setup above; a missing cache or audio/decoder failure is reported as a general failure.

The generated SRT is kept under `artifacts/transcripts/`, and a subtitle copy is saved in the local library. Both can contain private course content. Original files and existing notes are not overwritten. A video already in the library must have its index removed before retranscribing; if its source changes during processing, the new index is not imported. Review transcript accuracy against the recording.

Keep the workspace server open while transcribing. Select **取消转录** during transcription to stop the local child process; the original video and any generated subtitle file remain intact. Cancellation is unavailable once indexing has started, to avoid interrupting the library update. Task state exists only in the current server process. On normal Python exit, the server attempts to stop an active transcription child; forced termination or a crash can bypass cleanup and leave it running. There is no automatic resumption. If a task is interrupted, check local transcript files before retrying. This protection is separate from OS-level network isolation; local third-party dependencies are not sandboxed by the offline flags.

On 4 November 2026, the browser workflow was exercised against the 97-second public ethics sample using the cached base model on CPU. The process created five subtitle segments and a search for `medical professionals` returned the 0–19 second segment. This is a workflow check, not a transcription accuracy benchmark, and no Canvas material was used.


## Public-sample restart and review check

On 4 November 2026, the isolated browser workspace used the public 97-second ethics video after a real local Whisper base transcription. A search for `medical professionals` returned a 0–19 second excerpt. Browser automation created a course, assigned the video, saved the excerpt and added a note. After stopping and restarting the local server with the same isolated database, the course assignment, five indexed segments and saved note were still visible; selecting the saved clip started the public video at the beginning. No Canvas material was used.

This is one functional smoke test, not a study of transcription accuracy, timestamp accuracy, performance on long lectures, or real-user task success. The public example is not a substitute for independently reviewed questions and answers. For a private test, use authorized local inputs and keep the database, subtitles, notes and results on the same computer.


### Subtitle revision check

A separate local API smoke test copied the public ethics video and its generated subtitle to a temporary directory, imported the copy, saved a note on a retrieved segment, removed the index, changed the subtitle text and re-imported it. The new segment ID differed, the previous bookmark was unavailable for playback, and its note remained. After recreating the application with the same isolated database, the updated transcript was searchable. The copied video remained present throughout. This check used one public sample and does not verify all source-replacement or filesystem failure cases.


### Repository path check

A recursive GitHub tree listing at revision `e9cd7698a16d5f16dddd199c7c2743c759bc61ce` returned no tracked files under `artifacts/`, `local_data/`, `raw/` or `processed/`, and no tracked files with common media, subtitle or database extensions. This was a path and extension check only: it did not inspect every text file or prove that all sensitive content was absent. Future commits require their own review.
