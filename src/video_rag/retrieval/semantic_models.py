"""Resolve local semantic models without downloading or transmitting data."""
from pathlib import Path


def semantic_model_files(path: str | Path) -> dict[str, Path]:
    path = Path(path).expanduser().absolute()
    if path.suffix.lower() == '.gguf':
        return {'semantic_model': path}
    return {'semantic_tokenizer': path / 'tokenizer.json',
            'semantic_model': path / 'onnx' / 'model_quantized.onnx'}


def load_semantic_encoder(path: str | Path):
    path = Path(path).expanduser().absolute()
    files = semantic_model_files(path)
    if not all(file.is_file() for file in files.values()):
        raise ValueError('Local semantic model files not found')
    if path.suffix.lower() == '.gguf':
        from video_rag.retrieval.qwen_semantic import Qwen3EmbeddingEncoder

        return Qwen3EmbeddingEncoder(path)
    from video_rag.retrieval.onnx_semantic import OnnxSentenceEncoder

    return OnnxSentenceEncoder(path)
