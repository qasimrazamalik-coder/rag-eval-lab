"""Corpus and evaluation-set formats.

Corpus: a directory of ``.pdf`` / ``.txt`` files, or a ``docs.jsonl`` with ``{doc_id, text}`` rows.

Eval set, one JSON object per line. Required: ``question``. Optional: ``id``, ``answer``.
The gold passage is given in one of two ways:

* ``gold_doc`` + ``gold_start`` + ``gold_end``: character offsets into the cleaned document text
* ``gold_passage_text`` (optionally with ``gold_doc``): located by whitespace-tolerant exact
  match; an ambiguous or missing passage is an error, not a silent guess
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .ingest import Document, IngestError, extract_pdf
from .utils import read_jsonl


@dataclass(frozen=True)
class GoldItem:
    """One question with its gold span and reference answer."""

    id: str
    question: str
    answer: str
    doc_id: str
    start: int
    end: int


def load_corpus(path: str | Path) -> list[Document]:
    """Load documents from a docs.jsonl file or a directory of PDFs and text files."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus not found: {p}. Did you run `make data`?")
    if p.is_file():
        docs = [_doc_from(r) for r in read_jsonl(p)]
    else:
        docs = []
        for f in sorted(p.iterdir()):
            if f.suffix.lower() == ".pdf":
                docs.append(extract_pdf(f))
            elif f.suffix.lower() == ".txt":
                docs.append(_doc_from({"doc_id": f.stem, "text": f.read_text(encoding="utf-8")}))
    if not docs:
        raise IngestError(f"No documents found in {p}")
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate doc_id values in corpus")
    return docs


def _doc_from(row: dict) -> Document:
    """Text rows are used verbatim: offsets in the eval set refer to exactly this text."""
    return Document(row["doc_id"], row["text"])


def _find_passage(passage: str, docs: list[Document], only: str | None) -> tuple[str, int, int]:
    pattern = re.compile(r"\s+".join(re.escape(w) for w in passage.split()))
    found = []
    for d in docs:
        if only is not None and d.doc_id != only:
            continue
        for m in pattern.finditer(d.text):
            found.append((d.doc_id, m.start(), m.end()))
    if not found:
        raise ValueError(f"gold_passage_text not found in corpus: {passage[:60]!r}")
    if len(found) > 1:
        raise ValueError(
            f"gold_passage_text is ambiguous ({len(found)} matches); add gold_doc: {passage[:60]!r}"
        )
    return found[0]


def load_eval(path: str | Path, docs: list[Document]) -> list[GoldItem]:
    """Resolve every row of an eval JSONL to a verified gold span."""
    by_id = {d.doc_id: d for d in docs}
    items = []
    for n, row in enumerate(read_jsonl(path), start=1):
        if not str(row.get("question", "")).strip():
            raise ValueError(f"{path} row {n}: missing question")
        if "gold_passage_text" in row:
            doc_id, a, b = _find_passage(row["gold_passage_text"], docs, row.get("gold_doc"))
        elif {"gold_doc", "gold_start", "gold_end"} <= row.keys():
            doc_id, a, b = row["gold_doc"], int(row["gold_start"]), int(row["gold_end"])
            if doc_id not in by_id:
                raise ValueError(f"{path} row {n}: unknown gold_doc {doc_id!r}")
            if not 0 <= a < b <= len(by_id[doc_id].text):
                raise ValueError(f"{path} row {n}: gold span ({a}, {b}) outside document")
        else:
            raise ValueError(f"{path} row {n}: need gold_passage_text or gold_doc/start/end")
        items.append(
            GoldItem(
                str(row.get("id", f"q{n}")), row["question"], row.get("answer", ""), doc_id, a, b
            )
        )
    if not items:
        raise ValueError(f"{path} contains no questions")
    return items
