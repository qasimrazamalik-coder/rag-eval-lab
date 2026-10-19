"""Token counting and truncation behind one interface.

Chunk budgets use the regex tokenizer so that chunking is deterministic and needs no model
download. It counts words and punctuation marks, which is close to but not equal to the subword
count of any embedder. Keep budgets comfortably below the embedder's max sequence length.
The generator's context budget can use the real subword tokenizer via ``HFTokenizer``.
"""

from __future__ import annotations

import re
from typing import Protocol

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class Tokenizer(Protocol):
    """Anything that can count tokens and truncate text to a token budget."""

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...


class RegexTokenizer:
    """Words and single punctuation marks. Offsets are exact, so spans map back to the text."""

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in TOKEN_RE.finditer(text)]

    def count(self, text: str) -> int:
        return sum(1 for _ in TOKEN_RE.finditer(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        for i, m in enumerate(TOKEN_RE.finditer(text), start=1):
            if i == max_tokens:
                return text[: m.end()]
        return text


class HFTokenizer:
    """Subword tokenizer from ``transformers``; decoding after truncation is lossy by design."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        ids = self._tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self._tok.decode(ids[:max_tokens], skip_special_tokens=True)
