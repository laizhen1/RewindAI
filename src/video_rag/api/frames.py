"""Extract a single local video frame for evidence inspection, without a model."""
from __future__ import annotations

import math
from pathlib import Path


def extract_frame_png(path: str | Path, seconds: float) -> tuple[bytes, float]:
    """Return a resized PNG and its approximate playback time.

    PyAV is optional. Frames are decoded locally; seek failures or sparse streams
    fail instead of silently showing a frame far from the requested position.
    """
    import av

    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid frame time")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
        target = max(0.0, seconds + origin)
        container.seek(int(target * av.time_base), backward=True)
        selected = None
        for index, frame in enumerate(container.decode(stream)):
            if index >= 1500:
                break
            if frame.time is None:
                continue
            if frame.time >= target:
                selected = frame
                break
            selected = frame
        if selected is None or selected.time is None or abs(selected.time - target) > 2:
            raise ValueError("No nearby frame available")
        actual = max(0.0, float(selected.time) - origin)
        width = min(selected.width, 960)
        height = max(1, round(selected.height * width / selected.width))
        image = selected.reformat(width=width, height=height, format="rgb24")
        codec = av.CodecContext.create("png", "w")
        codec.width, codec.height, codec.pix_fmt = width, height, "rgb24"
        packets = [*codec.encode(image), *codec.encode(None)]
        data = b"".join(bytes(packet) for packet in packets)
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Frame encoding failed")
        return data, actual
