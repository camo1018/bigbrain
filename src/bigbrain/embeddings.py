"""Local text embeddings via fastembed (ONNX runtime, no API key).

The model is downloaded once and cached under the fastembed cache directory;
after that, embedding is fully offline.

Vectors are L2-normalized to unit length so that inner product equals cosine
similarity. This lets the store use Milvus's IP metric and read the returned
value directly as a similarity in [-1, 1] (typically [0, 1] for text), which
behaves identically on Milvus Lite and standalone Milvus.

BGE-style models are asymmetric: passages (stored topics) use a plain encoding
while search queries use a query-instruction encoding, so we expose both.
"""

from __future__ import annotations

from functools import cached_property


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def _model(self):
        # Imported lazily so importing this module (e.g. for --help) is cheap.
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode documents/topics (the stored side of the index)."""
        return [_normalize(vec.tolist()) for vec in self._model.embed(texts)]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_query(self, text: str) -> list[float]:
        """Encode a search query (asymmetric retrieval)."""
        vec = next(iter(self._model.query_embed([text])))
        return _normalize(vec.tolist())
