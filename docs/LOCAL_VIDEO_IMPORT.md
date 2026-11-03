# Importing authorized local video

From the project root, after installing the original project's video/model dependencies and FFmpeg:

```powershell
python scripts/ingest_local.py "C:\path\to\authorized-lecture.mp4" --language en
```

The input may be one video or a folder, which is searched recursively for supported video extensions. All matching videos in the folder must be authorized for the intended processing. The command runs the existing `prepare_videos.py` and `build_indexes.py` sequentially and writes to a new `artifacts/ingestions/<run-id>/` directory. It does not copy source video; segment metadata refers to its original path, so keep the video in place. Generated frames, transcripts and indexes can contain private course material: keep them local and out of version control. Use recordings only where your intended processing is authorized.

The two child processes run local inference and default to Hugging Face offline mode (`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`); required models must already be cached. To permit missing model files to download, add `--allow-model-download`. This setting controls model-hub access rather than providing a network sandbox; review dependencies and provider terms for sensitive material. The separate server process may need cached model files or its own offline environment settings.

For example, before running the printed server command in PowerShell, request the same model-hub offline behavior with:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

The real server also needs the configured reranker and vision-language model cached; successful ingestion alone does not prove those additional models are available.

Other options:

- `--device cpu` selects the indexing device only; video transcription and captioning use their own model/device behavior. CPU may be very slow or unsuitable for large models.
- `--skip-captions` skips the captioning stage but **does not** disable CLIP image indexing or make the full pipeline text-only.
- `--config path/to/config.toml` selects the existing configuration.

On success, the command prints the generated paths and an example `run_server.py` command. On failure it stops and reports the partial-output directory; no existing index is overwritten. Full model execution and quality depend on hardware, installed model libraries, model availability and the input video. The command is not model training.
