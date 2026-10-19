"""Okapi BM25 implemented directly (about 40 lines) to avoid another dependency.

Scoring uses the non-negative Lucene idf, ``log(1 + (N - n + 0.5) / (n + 0.5))``.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"\w+")


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word tokens; no stemming or stopword removal, so the baseline stays simple."""
    return _WORD_RE.findall(text.lower())


class BM25Index:
    """In-memory inverted index over a list of passages."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(texts)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(self.n_docs, dtype=np.float64)
        for i, text in enumerate(texts):
            counts = Counter(bm25_tokenize(text))
            lengths[i] = sum(counts.values())
            for term, tf in counts.items():
                self._postings.setdefault(term, []).append((i, tf))
        self._len_norm = (
            k1 * (1 - b + b * lengths / max(lengths.mean(), 1e-9)) if self.n_docs else 0
        )

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every passage for ``query``."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        for term in set(bm25_tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            idx = np.fromiter((d for d, _ in posting), dtype=np.int64, count=len(posting))
            tf = np.fromiter((t for _, t in posting), dtype=np.float64, count=len(posting))
            out[idx] += idf * tf * (self.k1 + 1) / (tf + self._len_norm[idx])
        return out
