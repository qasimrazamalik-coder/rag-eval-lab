"""Dense (FAISS exact inner product), BM25 and reciprocal-rank-fusion hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk
from .embedders import Embedder

MODES = ("dense", "bm25", "hybrid")
RRF_K = 60


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with its mode-specific score (cosine, BM25 or RRF)."""

    chunk: Chunk
    score: float
    rank: int


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best scores; ties break toward the lower index so runs are repeatable."""
    k = min(k, scores.shape[0])
    return np.argsort(-scores, kind="stable")[:k]


class Retriever:
    """Index over a fixed list of chunks. Dense and BM25 indexes are built lazily and cached."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, rrf_candidates: int = 50) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever over zero chunks")
        self.chunks = chunks
        self.embedder = embedder
        self.rrf_candidates = rrf_candidates
        self._dense: faiss.IndexFlatIP | None = None
        self._bm25: BM25Index | None = None

    def _dense_index(self) -> faiss.IndexFlatIP:
        if self._dense is None:
            vecs = np.ascontiguousarray(
                self.embedder.encode_documents([c.text for c in self.chunks]), dtype=np.float32
            )
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            self._dense = index
        return self._dense

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index([c.text for c in self.chunks])
        return self._bm25

    def _dense_scores(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        q = np.ascontiguousarray(self.embedder.encode_queries(queries), dtype=np.float32)
        sims, ids = self._dense_index().search(q, min(k, len(self.chunks)))
        return [
            [(int(i), float(s)) for i, s in zip(row_i, row_s, strict=True) if i >= 0]
            for row_i, row_s in zip(ids, sims, strict=True)
        ]

    def _bm25_scores(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        index = self._bm25_index()
        out = []
        for query in queries:
            s = index.scores(query)
            out.append([(int(i), float(s[i])) for i in _top_k(s, k)])
        return out

    def search_many(self, queries: list[str], k: int, mode: str = "dense") -> list[list[Hit]]:
        """Top-k hits per query. Hybrid fuses the two rankings with reciprocal rank fusion."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "dense":
            ranked = self._dense_scores(queries, k)
        elif mode == "bm25":
            ranked = self._bm25_scores(queries, k)
        else:
            depth = max(k, self.rrf_candidates)
            ranked = [
                _rrf(d, b)[:k]
                for d, b in zip(
                    self._dense_scores(queries, depth),
                    self._bm25_scores(queries, depth),
                    strict=True,
                )
            ]
        return [
            [Hit(self.chunks[i], s, r) for r, (i, s) in enumerate(row, start=1)] for row in ranked
        ]

    def search(self, query: str, k: int, mode: str = "dense") -> list[Hit]:
        return self.search_many([query], k, mode)[0]


def _rrf(*rankings: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score = sum over rankings of 1 / (RRF_K + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
