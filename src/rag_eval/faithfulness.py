"""Operational faithfulness: the share of answer sentences supported by a retrieved passage.

Definition used here. Split the answer into sentences. A sentence is *supported* when at least one
retrieved passage, taken alone as the premise, gets a support score >= the scorer's threshold:

* ``NLIScorer``: probability of the entailment class from an NLI cross-encoder (premise = passage,
  hypothesis = answer sentence).
* ``LexicalSupportScorer`` (fallback, no model): fraction of the sentence's content words found
  in the passage.

Faithfulness = supported sentences / answer sentences. Abstentions ("I don't know.") are excluded
and counted separately, because refusing is faithful but uninformative. Weaknesses are listed in
the README: single-passage premises miss multi-passage reasoning, NLI models are imperfect and
domain-sensitive, the threshold is a free parameter, and faithfulness to the context says nothing
about whether the context itself is correct or whether the answer is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .chunking import split_sentences
from .generation import content_words
from .prompts import ABSTAIN


class SupportScorer(Protocol):
    name: str
    threshold: float

    def score(self, premises: list[str], hypothesis: str) -> float:
        """Highest support of ``hypothesis`` by any single premise, in [0, 1]."""
        ...


class LexicalSupportScorer:
    """Content-word recall of the hypothesis against each premise; a crude, model-free proxy."""

    name = "lexical"

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def score(self, premises: list[str], hypothesis: str) -> float:
        words = content_words(hypothesis)
        if not words:
            return 1.0
        return max((len(words & content_words(p)) / len(words) for p in premises), default=0.0)


class NLIScorer:
    """Entailment probability from a sentence-transformers CrossEncoder trained on NLI data."""

    def __init__(
        self, model_name: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.5
    ) -> None:
        from sentence_transformers import CrossEncoder

        try:
            self._model = CrossEncoder(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Could not load NLI model '{model_name}'. Cache it with network access, set "
                "faithfulness.model_name to a local path, or use faithfulness.scorer: lexical."
            ) from exc
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        entail = [i for i, label in id2label.items() if label.startswith("entail")]
        if len(entail) != 1:
            raise RuntimeError(f"Cannot find a single entailment label in {id2label}")
        self._entail_idx = entail[0]
        self.name = model_name
        self.threshold = threshold

    def score(self, premises: list[str], hypothesis: str) -> float:
        import numpy as np

        if not premises:
            return 0.0
        logits = np.asarray(self._model.predict([(p, hypothesis) for p in premises]))
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(probs[:, self._entail_idx].max())


@dataclass(frozen=True)
class FaithfulnessResult:
    """``score`` is None when the answer is an abstention or has no sentences."""

    score: float | None
    n_sentences: int
    n_supported: int
    abstained: bool


def faithfulness(answer: str, premises: list[str], scorer: SupportScorer) -> FaithfulnessResult:
    """Fraction of answer sentences supported by at least one premise."""
    text = answer.strip()
    if not text or text == ABSTAIN:
        return FaithfulnessResult(None, 0, 0, abstained=text == ABSTAIN)
    sentences = [text[a:b] for a, b in split_sentences(text)]
    supported = sum(scorer.score(premises, s) >= scorer.threshold for s in sentences)
    return FaithfulnessResult(supported / len(sentences), len(sentences), supported, False)
