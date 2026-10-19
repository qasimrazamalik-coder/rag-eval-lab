"""Index one document and answer questions about it, logging tokens per query."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .chunking import Chunker
from .context import ContextBundle, build_context
from .embedders import Embedder
from .generation import Generator
from .ingest import Document
from .retrieval import Hit, Retriever
from .tokens import Tokenizer

logger = logging.getLogger("rag_eval.pipeline")


@dataclass(frozen=True)
class Answer:
    """Everything the app and the evaluator need to show or score one answered question."""

    question: str
    text: str
    hits: list[Hit]
    context: ContextBundle
    prompt_tokens: int
    completion_tokens: int


class RagPipeline:
    """Chunk, embed and index documents once, then answer any number of questions."""

    def __init__(
        self,
        docs: list[Document],
        chunker: Chunker,
        embedder: Embedder,
        generator: Generator,
        tokenizer: Tokenizer,
        mode: str = "dense",
        top_k: int = 5,
        context_budget: int = 1024,
    ) -> None:
        chunks = [c for d in docs for c in chunker.chunk(d)]
        self.retriever = Retriever(chunks, embedder)
        self.n_chunks = len(chunks)
        self.generator, self.tokenizer = generator, tokenizer
        self.mode, self.top_k, self.context_budget = mode, top_k, context_budget

    def ask(self, question: str) -> Answer:
        if not question.strip():
            raise ValueError("Question is empty.")
        hits = self.retriever.search(question, self.top_k, self.mode)
        context = build_context(hits, self.tokenizer, self.context_budget)
        gen = self.generator.generate(question, context)
        logger.info(
            "query tokens: context=%d prompt=%d completion=%d truncated=%s",
            context.tokens,
            gen.prompt_tokens,
            gen.completion_tokens,
            context.truncated,
        )
        return Answer(question, gen.answer, hits, context, gen.prompt_tokens, gen.completion_tokens)
