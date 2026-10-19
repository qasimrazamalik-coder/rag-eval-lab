"""Deterministic context assembly under a token budget."""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import PASSAGE_HEADER
from .retrieval import Hit
from .tokens import Tokenizer


@dataclass(frozen=True)
class ContextBundle:
    """Context text handed to the generator, the passages it contains and its token cost."""

    text: str
    passages: tuple[str, ...]
    tokens: int
    truncated: bool


def build_context(hits: list[Hit], tokenizer: Tokenizer, budget: int) -> ContextBundle:
    """Add passages in rank order; cut the first one that does not fit and stop there.

    The same hits and budget always give the same text, so token logs are reproducible.
    """
    blocks: list[str] = []
    passages: list[str] = []
    used = 0
    truncated = False
    for n, hit in enumerate(hits, start=1):
        header = PASSAGE_HEADER.format(n=n, page=hit.chunk.page)
        remaining = budget - used - tokenizer.count(header)
        if remaining <= 0:
            truncated = True
            break
        body = hit.chunk.text
        if tokenizer.count(body) > remaining:
            body = tokenizer.truncate(body, remaining)
            truncated = True
        blocks.append(f"{header}\n{body}")
        passages.append(body)
        used += tokenizer.count(header) + tokenizer.count(body)
        if truncated:
            break
    return ContextBundle("\n\n".join(blocks), tuple(passages), used, truncated)
