"""Experimental local CLIP visual retrieval using a quantized ONNX model."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

from video_rag.api.frames import extract_frame_png
from video_rag.schemas import SearchHit, VideoSegment

_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(vectors)):
        raise ValueError("CLIP returned invalid embeddings")
    return np.ascontiguousarray(vectors / norms)


def preprocess_png(data: bytes) -> np.ndarray:
    """Resize the short side to 224, center crop, and normalize RGB locally."""
    import av

    with av.open(BytesIO(data), format="png_pipe") as container:
        frame = next(container.decode(video=0))
        scale = 224 / min(frame.width, frame.height)
        width, height = max(224, round(frame.width * scale)), max(224, round(frame.height * scale))
        image = frame.reformat(width=width, height=height, format="rgb24", interpolation="BICUBIC")
        pixels = image.to_ndarray()
    left, top = (width - 224) // 2, (height - 224) // 2
    cropped = pixels[top:top + 224, left:left + 224].astype(np.float32) / 255.0
    return np.ascontiguousarray(((cropped - _MEAN) / _STD).transpose(2, 0, 1))


class OnnxClipEncoder:
    """Run a local CLIP model on CPU; never download or transmit input data."""

    def __init__(self, directory: str | Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        directory = Path(directory)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=77)
        self.tokenizer.enable_padding(pad_id=49407, pad_token="<|endoftext|>")
        self.session = ort.InferenceSession(str(directory / "onnx" / "model_quantized.onnx"),
                                            providers=["CPUExecutionProvider"])

    def _run(self, texts: list[str], pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer.encode_batch(texts)
        ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        mask = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        text, images = self.session.run(["text_embeds", "image_embeds"],
                                        {"input_ids": ids, "attention_mask": mask, "pixel_values": pixels})
        return _normalize(text), _normalize(images)

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("At least one image is required")
        return self._run(["a video frame"], np.stack(images).astype(np.float32))[1]

    def encode_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("A query is required")
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        return self._run([query], dummy)[0][0]


class ClipFrameRetriever:
    """Index one midpoint frame per segment, separately from transcript search."""

    name = "local_clip_frames"

    def __init__(self, encoder: OnnxClipEncoder) -> None:
        self.encoder = encoder
        self.ids: list[str] = []
        self.frame_times: list[float] = []
        self.vectors: np.ndarray | None = None

    def build(self, segments: list[VideoSegment]) -> None:
        if not segments:
            raise ValueError("At least one video segment is required")
        if len({segment.segment_id for segment in segments}) != len(segments):
            raise ValueError("Visual segment IDs must be unique")
        ids: list[str] = []
        frame_times: list[float] = []
        vectors = []
        import av

        for offset in range(0, len(segments), 4):
            batch = segments[offset:offset + 4]
            images = []
            for segment in batch:
                try:
                    png, actual = extract_frame_png(segment.source_path, (segment.start_time + segment.end_time) / 2)
                except av.error.FFmpegError as exc:
                    raise ValueError("Could not decode a video frame") from exc
                if not np.isfinite(actual) or actual < 0:
                    raise ValueError("Could not determine the sampled frame time")
                images.append(preprocess_png(png))
                frame_times.append(float(actual))
            encoded = np.asarray(self.encoder.encode_images(images), dtype=np.float32)
            if (encoded.ndim != 2 or encoded.shape[0] != len(batch) or encoded.shape[1] == 0
                    or not np.all(np.isfinite(encoded)) or np.any(np.linalg.norm(encoded, axis=1) == 0)):
                raise ValueError("CLIP returned invalid image embeddings")
            vectors.append(encoded)
            ids.extend(segment.segment_id for segment in batch)
        result = np.concatenate(vectors)
        self.ids, self.frame_times, self.vectors = ids, frame_times, result

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        if self.vectors is None:
            raise RuntimeError("Build the visual index first")
        if not query.strip() or top_k <= 0:
            return []
        scores = self.vectors @ self.encoder.encode_query(query)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.ids[i]))[:top_k]
        return [SearchHit(self.ids[index], float(scores[index]), self.name, rank,
                          {"frame_time": self.frame_times[index]})
                for rank, index in enumerate(order, 1)]
