"""Render retrieval results as a markdown table and write it between README marker comments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
COLUMNS = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10")
HEADERS = ("Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "nDCG@10")


def render_table(payload: dict[str, Any]) -> str:
    """Markdown for one results payload. Refuses synthetic payloads so fixtures cannot leak."""
    if payload.get("synthetic"):
        raise ValueError("Refusing to render synthetic fixture results into the README.")
    head = (
        f"Embedder `{payload['embedder']}`, {payload['n_questions']} questions over "
        f"{payload['n_documents']} documents, relevance threshold tau = {payload['tau']}, "
        f"seed {payload['seed']}. 95% bootstrap intervals are in the results JSON.\n\n"
    )
    lines = [
        "| Chunker | Retriever | Chunks | Mean tokens/chunk | " + " | ".join(HEADERS) + " |",
        "| --- | --- | ---: | ---: | " + " | ".join(["---:"] * len(HEADERS)) + " |",
    ]
    for chunker, block in payload["results"].items():
        for mode, metrics in block["retrievers"].items():
            cells = " | ".join(f"{metrics[c]['mean']:.3f}" for c in COLUMNS)
            lines.append(
                f"| {chunker} | {mode} | {block['n_chunks']} | "
                f"{block['mean_chunk_tokens']:.1f} | {cells} |"
            )
    return head + "\n".join(lines)


def update_readme(readme: str | Path, table: str) -> None:
    """Replace the text between the marker comments; fail if either marker is missing."""
    p = Path(readme)
    text = p.read_text(encoding="utf-8")
    if START not in text or END not in text or text.index(START) > text.index(END):
        raise ValueError(f"{p} must contain {START} before {END}")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    p.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--metrics", default="results/retrieval_metrics.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    update_readme(args.readme, render_table(payload))
    print(f"updated {args.readme} from {args.metrics}")


if __name__ == "__main__":
    main()
