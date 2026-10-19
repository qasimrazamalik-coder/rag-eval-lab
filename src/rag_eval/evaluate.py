"""Retrieval evaluation: every chunker x retriever mode on one fixed question set.

Fairness: all chunkers see the same documents, the same questions, the same embedder, the same
gold spans and the same relevance rule. What differs is only how the text is cut. Chunk counts
and mean chunk lengths are recorded next to the scores because they differ between strategies
and influence them.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Any

import numpy as np
from tqdm import tqdm

from .chunking import Chunk
from .embedders import Embedder
from .evalset import GoldItem, load_corpus, load_eval
from .factory import build_chunker, build_embedder
from .fixtures import generate_corpus
from .ingest import Document
from .metrics import KS, bootstrap_ci, is_relevant, ndcg_at_k, recall_at_k, reciprocal_rank
from .retrieval import Retriever
from .tokens import RegexTokenizer
from .utils import load_config, logger, package_versions, seed_everything, write_json

DEPTH = max(KS)


def _summary(values: list[float], n_boot: int, seed: int) -> dict[str, Any]:
    lo, hi = bootstrap_ci(values, n_boot, seed)
    return {"mean": float(np.mean(values)), "ci95": [lo, hi]}


def evaluate_chunker(
    docs: list[Document],
    items: list[GoldItem],
    chunker_spec: dict[str, Any],
    embedder: Embedder,
    modes: list[str],
    tau: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Score one chunking strategy with each retriever mode."""
    chunker = build_chunker(chunker_spec)
    chunks: list[Chunk] = [c for d in docs for c in chunker.chunk(d)]
    tok = RegexTokenizer()
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    n_relevant = [sum(is_relevant(c, g, tau) for c in by_doc[g.doc_id]) for g in items]
    retriever = Retriever(chunks, embedder)
    questions = [g.question for g in items]

    out: dict[str, Any] = {
        "spec": chunker_spec,
        "n_chunks": len(chunks),
        "mean_chunk_tokens": float(np.mean([tok.count(c.text) for c in chunks])),
        "mean_chunk_chars": float(np.mean([len(c.text) for c in chunks])),
        "questions_without_relevant_chunk": int(sum(n == 0 for n in n_relevant)),
        "retrievers": {},
    }
    for mode in modes:
        hits = retriever.search_many(questions, DEPTH, mode)
        flags = [
            [is_relevant(h.chunk, g, tau) for h in row] for g, row in zip(items, hits, strict=True)
        ]
        res = {
            f"recall@{k}": _summary([recall_at_k(f, k) for f in flags], n_boot, seed) for k in KS
        }
        res["mrr@10"] = _summary([reciprocal_rank(f, DEPTH) for f in flags], n_boot, seed)
        res["ndcg@10"] = _summary(
            [ndcg_at_k(f, DEPTH, n) for f, n in zip(flags, n_relevant, strict=True)], n_boot, seed
        )
        out["retrievers"][mode] = res
    return out


def load_data(cfg: dict[str, Any]) -> tuple[list[Document], list[GoldItem], bool]:
    """Return documents, gold items and whether they are the synthetic fixture."""
    data = cfg["data"]
    if data["source"] == "synthetic":
        syn = data.get("synthetic", {})
        docs, items = generate_corpus(syn.get("n_docs", 6), syn.get("paragraphs", 5), cfg["seed"])
        return docs, items, True
    docs = load_corpus(data["corpus"])
    return docs, load_eval(data["eval"], docs), False


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    seed_everything(cfg["seed"])
    docs, items, synthetic = load_data(cfg)
    embedder = build_embedder(cfg["embedder"])
    r = cfg["retrieval"]
    results = {}
    for spec in tqdm(cfg["chunkers"], desc="chunkers"):
        name = spec.get("name", spec["type"])
        results[name] = evaluate_chunker(
            docs, items, spec, embedder, r["modes"], r["tau"], r.get("bootstrap", 1000), cfg["seed"]
        )
    return {
        "synthetic": synthetic,
        "seed": cfg["seed"],
        "versions": package_versions(),
        "embedder": embedder.name,
        "tau": r["tau"],
        "n_documents": len(docs),
        "n_questions": len(items),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    payload = run(cfg)
    path = write_json(cfg["output"]["metrics"], payload)
    logger.info("wrote %s", path)
    if payload["synthetic"]:
        logger.info("SYNTHETIC fixture: pipeline check only, not a result.")


if __name__ == "__main__":
    main()
