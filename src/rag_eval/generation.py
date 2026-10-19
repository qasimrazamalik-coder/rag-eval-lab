"""Answer generators behind one interface: a transformers model and an extractive fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .bm25 import bm25_tokenize
from .chunking import split_sentences
from .context import ContextBundle
from .prompts import ABSTAIN, SYSTEM_PROMPT, USER_TEMPLATE

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "does",
        "do",
        "did",
        "has",
        "have",
        "had",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "as",
        "into",
        "than",
    }
)


@dataclass(frozen=True)
class Generation:
    """An answer and the tokens the generator itself consumed (zero for the extractive one)."""

    answer: str
    prompt_tokens: int
    completion_tokens: int


class Generator(Protocol):
    name: str

    def generate(self, question: str, context: ContextBundle) -> Generation: ...


def content_words(text: str) -> set[str]:
    """Lowercase word set without stopwords."""
    return {w for w in bm25_tokenize(text) if w not in _STOPWORDS}


class ExtractiveGenerator:
    """Return the retrieved sentence that shares the most content words with the question.

    Ties go to the shorter sentence, then the earlier one. No overlap at all means abstain.
    It cannot answer anything that is not already one sentence in the context.
    """

    name = "extractive"

    def generate(self, question: str, context: ContextBundle) -> Generation:
        q_words = content_words(question)
        best: tuple[int, int, int] | None = None
        best_sentence = ABSTAIN
        order = 0
        for passage in context.passages:
            for a, b in split_sentences(passage):
                sentence = passage[a:b]
                overlap = len(q_words & content_words(sentence))
                key = (-overlap, len(sentence), order)
                order += 1
                if overlap > 0 and (best is None or key < best):
                    best, best_sentence = key, sentence
        return Generation(re.sub(r"\s+", " ", best_sentence), 0, 0)


class TransformersGenerator:
    """Greedy decoding with a small instruct model through its chat template."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 128,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            self._tok = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Could not load generator '{model_name}'. Run once with network access so it is "
                "cached, or set generator.model_name to a local path in the config."
            ) from exc
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()
        self.name = model_name
        self.max_new_tokens = max_new_tokens

    def generate(self, question: str, context: ContextBundle) -> Generation:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(context=context.text, question=question),
            },
        ]
        inputs = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self._device)
        n_prompt = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        new_ids = out[0, n_prompt:]
        answer = self._tok.decode(new_ids, skip_special_tokens=True).strip()
        return Generation(answer, n_prompt, int(new_ids.shape[0]))
