"""Embedder interface, the sentence-transformers implementation and a deterministic stub.

All embedders return L2-normalised float32 matrices so inner product equals cosine similarity.
The stub is a hashed bag-of-words: it exists so tests, the smoke run and the offline app path
work without downloading weights. It is a lexical baseline, not a neural model, and its numbers
must never be reported as dense-retrieval results.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_WORD_RE = re.compile(r"\w+")


class Embedder(Protocol):
    """Maps texts to unit vectors. Queries and documents are separate to support prefixes."""

    name: str

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model; weights come from the local Hugging Face cache."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(model_name, device=device)
        except OSError as exc:
            raise RuntimeError(
                f"Could not load embedding model '{model_name}'. Run once with network access "
                "so it is cached, or set embedder.model_name to a local path in the config."
            ) from exc
        self.name = model_name
        self.query_prefix, self.document_prefix = query_prefix, document_prefix
        self.batch_size = batch_size

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        vecs = self._model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.document_prefix)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)


class HashingEmbedder:
    """Signed feature hashing of lowercase words (blake2b, so stable across processes)."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = f"hashing-bow-{dim}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD_RE.findall(text.lower()):
                h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)
