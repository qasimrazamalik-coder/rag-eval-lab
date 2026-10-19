"""Build docs.jsonl and eval.jsonl from a SQuAD-format JSON file (v1.1 or v2.0).

Get SQuAD v2 dev with `python scripts/download_data.py` (see that script for the URL caveat).
Unanswerable questions are dropped because they have no gold span.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_eval.squad import convert
from rag_eval.utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--squad", default="data/squad/dev-v2.0.json")
    parser.add_argument("--out-dir", default="data/squad_dev")
    parser.add_argument("--granularity", choices=["article", "context"], default="article")
    parser.add_argument("--max-questions", type=int, default=1000)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    docs, rows = convert(
        args.squad, args.granularity, args.max_questions, args.max_articles, args.seed
    )
    out = Path(args.out_dir)
    write_jsonl(out / "docs.jsonl", docs)
    write_jsonl(out / "eval.jsonl", rows)
    print(f"wrote {len(docs)} documents and {len(rows)} questions to {out}")


if __name__ == "__main__":
    main()
