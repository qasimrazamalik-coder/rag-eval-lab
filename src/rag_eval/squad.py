"""Convert a SQuAD-format JSON file into a corpus (docs.jsonl) and an eval set (eval.jsonl).

Answerable questions only: SQuAD v2 ``is_impossible`` questions have no gold span, so they cannot
score retrieval. With ``granularity="article"`` all paragraphs of one article form a single
document, which gives the chunkers real multi-paragraph structure to work with. With
``"context"`` every paragraph is its own document, which makes retrieval easier because the
candidate pool is made of short, self-contained texts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

PARA_SEP = "\n\n"


def convert(
    squad_path: str | Path,
    granularity: str = "article",
    max_questions: int | None = 1000,
    max_articles: int | None = None,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Return ``(docs, eval_rows)``. Sampling is seeded, so the same inputs give the same files."""
    if granularity not in {"article", "context"}:
        raise ValueError("granularity must be 'article' or 'context'")
    with Path(squad_path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "data" not in payload:
        raise ValueError("Not a SQuAD file: missing top-level 'data'")
    articles = payload["data"][:max_articles] if max_articles else payload["data"]

    docs: list[dict] = []
    rows: list[dict] = []
    for a_idx, article in enumerate(articles):
        offset = 0
        for p_idx, para in enumerate(article["paragraphs"]):
            context = para["context"]
            if granularity == "article":
                doc_id = f"a{a_idx:03d}_{_slug(article.get('title', 'untitled'))}"
                base = offset
                offset += len(context) + len(PARA_SEP)
            else:
                doc_id = f"a{a_idx:03d}_p{p_idx:03d}"
                base = 0
                docs.append({"doc_id": doc_id, "text": context})
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa.get("answers"):
                    continue
                ans = qa["answers"][0]
                start = base + int(ans["answer_start"])
                rows.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "answer": ans["text"],
                        "gold_doc": doc_id,
                        "gold_start": start,
                        "gold_end": start + len(ans["text"]),
                    }
                )
        if granularity == "article":
            text = PARA_SEP.join(p["context"] for p in article["paragraphs"])
            docs.append(
                {"doc_id": f"a{a_idx:03d}_{_slug(article.get('title', 'untitled'))}", "text": text}
            )

    if max_questions and len(rows) > max_questions:
        rows = sorted(random.Random(seed).sample(rows, max_questions), key=lambda r: r["id"])
    return docs, rows


def _slug(title: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in title)[:40]
