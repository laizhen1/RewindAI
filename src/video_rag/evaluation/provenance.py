"""Local source fingerprints for time-labeled evaluation; no network access."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def fingerprint(path: Path) -> str:
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("A source changed while it was being fingerprinted")
    return digest.hexdigest()


def bind_sources(data: dict, video: Path, subtitles: Path) -> None:
    data["source_sha256"] = {"video": fingerprint(video), "subtitles": fingerprint(subtitles)}


def verify_sources(data: dict, video: Path, subtitles: Path) -> str:
    if not isinstance(data, dict):
        raise TypeError("Question file must contain a JSON object")
    binding = data.get("source_sha256")
    if binding is None:
        return "unbound"
    if (not isinstance(binding, dict) or set(binding) != {"video", "subtitles"} or
            any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in binding.values())):
        raise ValueError("Invalid source fingerprint metadata")
    if fingerprint(video) != binding["video"] or fingerprint(subtitles) != binding["subtitles"]:
        raise ValueError("Question file does not match the supplied video and subtitles")
    return "matched"
