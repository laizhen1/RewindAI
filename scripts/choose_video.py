"""Choose a local video and launch offline transcription and subtitle search."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1:
        print("Choose one video at a time.", file=sys.stderr)
        return 2
    if args:
        video = args[0]
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as exc:
            print(f"File picker is unavailable: {exc}", file=sys.stderr)
            return 1
        try:
            root = tk.Tk()
            root.withdraw()
            try:
                video = filedialog.askopenfilename(
                    title="Choose an authorized local video",
                    filetypes=[("Video files", "*.mp4 *.webm *.ogv *.mov *.m4v"), ("All files", "*.*")],
                )
            finally:
                root.destroy()
        except (RuntimeError, tk.TclError) as exc:
            print(f"Could not open the file picker: {exc}", file=sys.stderr)
            return 1
        if not video:
            print("No video selected.")
            return 0
    print("Transcribing locally with cached Whisper base on CPU. Model downloads are disabled.", flush=True)
    print("When ready, open http://127.0.0.1:5000/; close this window to stop.", flush=True)
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "rewind_local.py"), video], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
