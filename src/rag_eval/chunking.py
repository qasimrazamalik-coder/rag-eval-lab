"""Three deterministic chunkers that all return character spans into the cleaned document.

Every chunk keeps ``start``/``end`` offsets so retrieval can be scored against gold spans by plain
interval overlap, and so the app can report the page a chunk came from. Budgets are expressed in
``tokens`` (regex tokens, see ``tokens.py``) or ``chars``.

* fixed: sliding window of ``size`` units advancing by ``size - overlap``. Ignores structure.
* sentence_window: ``window`` consecutive sentences advancing by ``stride`` sentences, shrunk to fit
  the budget. Ignores paragraphs beyond treating them as sentence boundaries.
* recursive: split on paragraphs, then sentences, then words, then raw units, and merge adjacent
  pieces back up to the budget. No overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .ingest import Document
from .tokens import RegexTokenizer

Span = tuple[int, int]

_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "vs",
        "fig",
        "no",
        "inc",
        "ltd",
        "etc",
        "al",
        "e.g",
        "i.e",
    }
)
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n\s*\n")
_WORD_RE = re.compile(r"\S+")
_TOKENIZER = RegexTokenizer()


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage and its provenance."""

    doc_id: str
    chunk_id: str
    start: int
    end: int
    text: str
    page: int


class Chunker(Protocol):
    """Splits a Document into chunks; must be a pure function of its inputs."""

    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def measure(text: str, unit: str) -> int:
    """Size of ``text`` in the chosen budget unit."""
    return len(text) if unit == "chars" else _TOKENIZER.count(text)


def _check_unit(unit: str) -> None:
    if unit not in {"tokens", "chars"}:
        raise ValueError(f"unit must be 'tokens' or 'chars', got {unit!r}")


def split_sentences(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Sentence spans inside ``text[start:end]``, stripped of surrounding whitespace.

    Regex based: a boundary is terminal punctuation followed by whitespace and a capital letter or
    digit, or a blank line. Common abbreviations are protected. It will mis-split some initials and
    will not split sentences that begin with a lowercase letter.
    """
    end = len(text) if end is None else end
    spans: list[Span] = []
    cursor = start
    for m in _SENT_BOUNDARY_RE.finditer(text, start, end):
        before = text[cursor : m.start()]
        last_word = before.rsplit(None, 1)[-1].rstrip(".").lower() if before.split() else ""
        if "\n" not in m.group() and last_word in _ABBREVIATIONS:
            continue
        spans.append((cursor, m.start()))
        cursor = m.end()
    spans.append((cursor, end))
    return [s for s in (_strip(text, a, b) for a, b in spans) if s[1] > s[0]]


def _strip(text: str, a: int, b: int) -> Span:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def fixed_spans(text: str, start: int, end: int, size: int, overlap: int, unit: str) -> list[Span]:
    """Sliding windows over ``text[start:end]``; the last window always reaches ``end``."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    step = size - overlap
    offsets: list[Span] | None = None
    if unit == "chars":
        n = end - start
    else:
        offsets = [(start + a, start + b) for a, b in _TOKENIZER.spans(text[start:end])]
        n = len(offsets)
    if n == 0:
        return []
    spans: list[Span] = []
    i = 0
    while True:
        j = min(i + size, n)
        if offsets is None:
            spans.append((start + i, start + j))
        else:
            spans.append((offsets[i][0], offsets[j - 1][1]))
        if j >= n:
            break
        i += step
    return spans


class FixedChunker:
    """Sliding window by token or character budget with a fixed overlap."""

    name = "fixed"

    def __init__(self, size: int = 128, overlap: int = 32, unit: str = "tokens") -> None:
        _check_unit(unit)
        if size <= 0 or not 0 <= overlap < size:
            raise ValueError("need size > 0 and 0 <= overlap < size")
        self.size, self.overlap, self.unit = size, overlap, unit

    def chunk(self, doc: Document) -> list[Chunk]:
        spans = fixed_spans(doc.text, 0, len(doc.text), self.size, self.overlap, self.unit)
        return make_chunks(doc, spans)


class SentenceWindowChunker:
    """Windows of whole sentences; ``window - stride`` sentences are shared between neighbours."""

    name = "sentence_window"

    def __init__(
        self, size: int = 128, unit: str = "tokens", window: int = 4, stride: int = 3
    ) -> None:
        _check_unit(unit)
        if size <= 0 or window <= 0 or not 0 < stride <= window:
            raise ValueError("need size > 0, window > 0 and 0 < stride <= window")
        self.size, self.unit, self.window, self.stride = size, unit, window, stride

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        sents = split_sentences(text)
        spans: list[Span] = []
        i = 0
        while i < len(sents):
            j = min(i + self.window, len(sents))
            while j > i + 1 and measure(text[sents[i][0] : sents[j - 1][1]], self.unit) > self.size:
                j -= 1
            a, b = sents[i][0], sents[j - 1][1]
            if j == i + 1 and measure(text[a:b], self.unit) > self.size:
                spans.extend(fixed_spans(text, a, b, self.size, 0, self.unit))
            else:
                spans.append((a, b))
            if j >= len(sents):
                break
            i += min(self.stride, j - i)
        return make_chunks(doc, spans)


class RecursiveChunker:
    """Split on paragraphs, then sentences, then words, then raw units; merge up to the budget."""

    name = "recursive"

    def __init__(self, size: int = 128, unit: str = "tokens") -> None:
        _check_unit(unit)
        if size <= 0:
            raise ValueError("size must be positive")
        self.size, self.unit = size, unit

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        pieces = self._split(text, 0, len(text), level=0)
        merged: list[Span] = []
        for a, b in pieces:
            if merged and measure(text[merged[-1][0] : b], self.unit) <= self.size:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        return make_chunks(doc, merged)

    def _split(self, text: str, start: int, end: int, level: int) -> list[Span]:
        if measure(text[start:end], self.unit) <= self.size:
            return [(start, end)] if end > start else []
        if level == 0:
            parts = _paragraph_spans(text, start, end)
        elif level == 1:
            parts = split_sentences(text, start, end)
        elif level == 2:
            parts = [
                (start + m.start(), start + m.end()) for m in _WORD_RE.finditer(text[start:end])
            ]
        else:
            return fixed_spans(text, start, end, self.size, 0, self.unit)
        if len(parts) == 1 and parts[0] == _strip(text, start, end):
            return self._split(text, parts[0][0], parts[0][1], level + 1)
        out: list[Span] = []
        for a, b in parts:
            out.extend(self._split(text, a, b, level + 1))
        return out


def _paragraph_spans(text: str, start: int, end: int) -> list[Span]:
    spans, cursor = [], start
    for m in re.finditer(r"\n\s*\n", text[start:end]):
        spans.append((cursor, start + m.start()))
        cursor = start + m.end()
    spans.append((cursor, end))
    return [s for s in (_strip(text, a, b) for a, b in spans) if s[1] > s[0]]


def make_chunks(doc: Document, spans: list[Span]) -> list[Chunk]:
    """Turn spans into Chunks, trimming whitespace and dropping empties."""
    chunks: list[Chunk] = []
    for a, b in spans:
        a, b = _strip(doc.text, a, b)
        if b > a:
            chunks.append(
                Chunk(
                    doc.doc_id, f"{doc.doc_id}#{len(chunks)}", a, b, doc.text[a:b], doc.page_of(a)
                )
            )
    return chunks
