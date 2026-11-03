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

On Windows, after installing the ASR dependency and caching the model, double-click `Start-RewindAI-Transcribe.cmd` and choose a video, or drag one video onto that file. The tool automatically detects the language, transcribes on CPU and then starts the local search interface. It never opts into downloading the model. Keep the command window open while using the interface. If the model is absent, follow the explicit download instructions above first. The generated SRT remains in `artifacts/transcripts`; the video loaded by this launcher is available for that session, and can be added through the regular workspace using that SRT for later restoration.

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
