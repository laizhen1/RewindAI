# React workspace setup

I built the React workspace as the primary interface for local video study. It uses the existing Flask API and runs on the same computer as the models and files. The original interface remains available at `/classic`.

## Install and build

Use Python 3.11 or later and a current Node.js LTS release. From the repository root, create and activate a virtual environment, then install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --only-binary=:all: "llama-cpp-python>=0.3.36,<0.4" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m pip install -e ".[dev,asr,qwen-semantic]" huggingface-hub
```

The first command requests an upstream prebuilt CPU wheel and refuses to compile if no compatible binary is available. If it fails, consult the upstream [installation instructions](https://github.com/abetlen/llama-cpp-python#installation) for supported binaries or a C++ compiler. If `python` is not found, install Python and add it to PATH before reopening the terminal. Then install the locked frontend dependencies and build the React assets:

```powershell
cd frontend
npm ci
npm run build
cd ..
```

## Prepare the public models

These explicit setup commands download **public model files**, not lecture recordings. Run them before importing private material:

```powershell
python scripts/setup_qwen_embedding.py
python scripts/setup_answer_model.py
python scripts/setup_whisper_model.py
```

The same unmodified Qwen3 model is available in the [GitHub release](https://github.com/laizhen1/RewindAI/releases/tag/qwen3-embedding-0.6b-q8), alongside its Apache-2.0 license and SHA-256 checksum. The setup command above downloads the pinned upstream file directly.

Qwen3-Embedding provides semantic retrieval; the Qwen2.5 model generates text answers; Whisper base transcribes videos that do not have SRT/VTT subtitles. The Qwen3 and answer setup scripts check pinned model hashes. The Whisper setup uses the public upstream `main` revision and checks that required files are present; it is not a pinned checksum verification. Each setup script supports `--check-only` for an offline cache check. Model licenses are separate: see the [Qwen3-Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF), [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF) and [faster-whisper model card](https://huggingface.co/Systran/faster-whisper-base).

Qwen3 is already pre-trained. Importing a video transcribes or reads its captions and builds a local retrieval index; it **does not train or fine-tune model weights**. Fine-tuning is not part of this workflow. The 639 MB Qwen3 weights are obtained from the upstream model repository rather than stored in regular Git history, which has a 100 MiB per-file limit.

## Start and use

On Windows, double-click `Start-RewindAI-React.cmd` and open [http://127.0.0.1:5000/](http://127.0.0.1:5000/). Keep the command window open. The launcher requires the built frontend and the local Qwen3 and answer-model files. On other platforms, run `scripts/run_transcript.py` with the `--semantic-model` and `--answer-model` paths described in the main README.

For manual startup on any platform, use the model paths printed by the two setup commands. For example, after activating the virtual environment:

```bash
python scripts/run_transcript.py --semantic-model /path/to/Qwen3-Embedding-0.6B-Q8_0.gguf --answer-model /path/to/qwen2.5-3b-instruct-q4_k_m.gguf --port 5001
```

Replace both paths with your own and open `http://127.0.0.1:5001/`. Choose an unused port. On Windows, quote paths containing spaces. The GUI launcher uses port 5000; do not run two instances on the same port. If `HF_HOME`, `HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE` is set, Whisper may use that cache instead of the project cache; make sure its base model is available there before transcription.

Select **Add lecture**, choose a video and optionally an SRT/VTT file, read the privacy notice, and confirm both permission and local storage. With subtitles the app builds the index directly; otherwise it starts a CPU transcription job using the cached Whisper base model. The video limit is 2 GiB and the subtitle limit is 10 MiB. Processing can take time and transcripts can contain errors. Ask a question, review the returned excerpts and their lecture names, and use the timestamp to return to the recording. Search results are candidates, not a guarantee that the answer is supported. Save useful moments and notes locally.

To remove a lecture from the searchable library, select it and choose **Remove from library**. This removes its index entry, not its video or subtitle files; saved moments and notes are retained but may become unavailable until compatible material is re-imported. The confirmation explains this before removal.

If the React assets have not been built, `/` serves the original workspace. Rebuild after frontend changes. The React workspace does not yet expose every advanced function of the original interface; use `/classic` for visual search, subtitle replacement, feedback history and other established tools.

## Privacy and responsibility

The browser sends the selected file to the Flask server on `127.0.0.1` on the same computer. Successful imports retain a copy under `artifacts/uploads/`; generated subtitles, local indexes, notes, feedback and reports are also stored on the computer. The operating system or browser may create temporary copies. Removing a video index does **not** delete the uploaded video, subtitles or bookmarks. Failed or cancelled transcription may leave local copies. Manage and delete copies and derived files yourself when appropriate, including any backups or cloud-synced folders.

The local processing paths described here do not send lecture contents to a hosted AI service. Downloading public dependencies and model weights requires network access, but those setup commands do not need the user's lecture file. The server is loopback-only and does not provide accounts, multi-user isolation or production authentication. Other software or users with access to the computer may be able to access the files; do not expose the server to the internet.

Before importing any lecture, confirm that its owner, institution and applicable policies permit recording, transcription and use in this tool. The fact that processing uses AI does not itself grant permission. Users must not upload or share restricted recordings or their derivatives with others unless permitted.
