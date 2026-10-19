"""Retrieval metrics against gold spans.

Relevance rule (binary). A chunk is relevant to a question when it is in the gold document and

    overlap_chars / min(len(gold_span), len(chunk)) >= tau        (default tau = 0.5)

i.e. either the chunk is mostly inside the gold span, or the gold span is mostly inside the chunk.
Dividing by the smaller of the two keeps the rule symmetric, so a short chunk wholly inside a long
gold passage and a long chunk wholly containing a short gold answer both count. tau is a config
value. This rule favours neither large nor small chunks by construction, but it does not remove
the effect of chunk size on the metrics; see the README.

Metric definitions (per question, then averaged):
* recall@k  = 1 if any relevant chunk is in the top k, else 0 (a hit rate: there is one gold span)
* MRR@10    = 1 / rank of the first relevant chunk in the top 10, else 0
* nDCG@10   = DCG / IDCG with binary gains and log2(rank + 1) discount; IDCG assumes
              min(10, number of relevant chunks in the index) relevant results at the top
"""

from __future__ import annotations

import math

import numpy as np

from .chunking import Chunk
from .evalset import GoldItem

KS = (1, 3, 5, 10)


def is_relevant(chunk: Chunk, gold: GoldItem, tau: float = 0.5) -> bool:
    """Apply the overlap rule to one chunk."""
    if chunk.doc_id != gold.doc_id:
        return False
    overlap = min(chunk.end, gold.end) - max(chunk.start, gold.start)
    if overlap <= 0:
        return False
    return overlap / min(gold.end - gold.start, chunk.end - chunk.start) >= tau


def recall_at_k(flags: list[bool], k: int) -> float:
    return 1.0 if any(flags[:k]) else 0.0


def reciprocal_rank(flags: list[bool], cutoff: int = 10) -> float:
    for rank, hit in enumerate(flags[:cutoff], start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(flags: list[bool], k: int, n_relevant: int) -> float:
    """Binary-gain nDCG. Returns 0.0 when the index holds no relevant chunk at all."""
    if n_relevant <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(flags[:k]) if hit)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_relevant)))
    return dcg / idcg


def bootstrap_ci(
    values: list[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling questions with replacement."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
