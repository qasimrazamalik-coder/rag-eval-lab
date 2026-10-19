"""Synthetic corpus, gold questions and a tiny PDF, all generated from a seed.

SYNTHETIC DATA. Entities are invented syllable strings, so no pretrained model can answer from
memory. Results on this fixture prove the pipeline runs end to end; they say nothing about real
retrieval quality and are never written into the README table.
"""

from __future__ import annotations

import random
from pathlib import Path

from .evalset import GoldItem
from .ingest import PARA_SEP, Document

_SYLLABLES = ["vel", "tra", "kor", "mi", "nu", "zan", "dro", "pel", "ish", "qua", "bor", "lin"]
_CITIES = ["Marrowdale", "Ostrenna", "Kellwick", "Brannoch", "Tavistoke", "Lunmere", "Harrowgate"]
_FILLER = [
    "The committee reviewed the budget for the coming quarter.",
    "Several visitors arrived early to look at the exhibition.",
    "Records from that period were kept in a damp cellar.",
    "The annual report described steady progress on all fronts.",
    "A short report was circulated among the staff in March.",
    "Weather delayed the delivery of the main equipment twice.",
    "Local newspapers covered the opening in some detail.",
    "The building was repainted after the inspection.",
    "Engineers and clerks shared a single large office.",
    "Funding discussions continued through the winter months.",
]
_FACTS = [
    ("founded", "The {e} was founded in {n} by {p}.", "Who founded the {e}?", "{p}"),
    ("city", "The headquarters of the {e} stand in {c}.", "Where is the {e} headquartered?", "{c}"),
    (
        "staff",
        "The {e} employs {n} people at its main site.",
        "How many people does the {e} employ?",
        "{n}",
    ),
]


def _name(rng: random.Random, taken: set[str]) -> str:
    while True:
        word = "".join(rng.choice(_SYLLABLES) for _ in range(3)).capitalize()
        if word not in taken:
            taken.add(word)
            return word


def generate_corpus(
    n_docs: int = 6, paragraphs: int = 5, seed: int = 0
) -> tuple[list[Document], list[GoldItem]]:
    """Documents with one unique fact sentence per paragraph, and a question for each fact."""
    rng = random.Random(seed)
    taken: set[str] = set()
    docs, items = [], []
    for d in range(n_docs):
        doc_id = f"syn{d:02d}"
        parts: list[str] = []
        pending: list[tuple[str, str, str, int]] = []
        pos = 0
        for _ in range(paragraphs):
            kind, template, question, answer = rng.choice(_FACTS)
            fields = {
                "e": f"{_name(rng, taken)} {rng.choice(['Institute', 'Works', 'Archive'])}",
                "p": f"{_name(rng, taken)} {_name(rng, taken)}",
                "c": rng.choice(_CITIES),
                "n": str(rng.randint(1100, 1990) if kind == "founded" else rng.randint(40, 900)),
            }
            fact = template.format(**fields)
            sentences = rng.sample(_FILLER, 3)
            sentences.insert(rng.randint(0, 3), fact)
            paragraph = " ".join(sentences)
            start = pos + paragraph.index(fact)
            pending.append((question.format(**fields), answer.format(**fields), fact, start))
            parts.append(paragraph)
            pos += len(paragraph) + len(PARA_SEP)
        docs.append(Document(doc_id, PARA_SEP.join(parts)))
        for k, (q, a, fact, start) in enumerate(pending):
            items.append(GoldItem(f"{doc_id}-{k}", q, a, doc_id, start, start + len(fact)))
    return docs, items


def make_pdf(path: str | Path, pages: list[list[str]]) -> Path:
    """Write a text PDF (one list of paragraphs per page) with reportlab, for ingestion tests."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import simpleSplit
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("make_pdf needs reportlab: pip install -e '.[dev]'") from exc
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(p), pagesize=A4)
    width, height = A4
    for paragraphs in pages:
        y = height - 72
        for para in paragraphs:
            for line in simpleSplit(para, "Helvetica", 11, width - 144):
                c.setFont("Helvetica", 11)
                c.drawString(72, y, line)
                y -= 15
            y -= 18
        c.showPage()
    c.save()
    return p
