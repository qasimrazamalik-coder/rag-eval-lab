"""Build components from a config mapping. Heavy imports stay inside the branches that need them."""

from __future__ import annotations

from typing import Any

from .chunking import Chunker, FixedChunker, RecursiveChunker, SentenceWindowChunker
from .embedders import Embedder, HashingEmbedder
from .faithfulness import LexicalSupportScorer, SupportScorer
from .generation import ExtractiveGenerator, Generator
from .tokens import HFTokenizer, RegexTokenizer, Tokenizer


def build_chunker(spec: dict[str, Any]) -> Chunker:
    """``spec['type']`` selects the strategy; other keys are its parameters."""
    kind = spec["type"]
    unit = spec.get("unit", "tokens")
    if kind == "fixed":
        return FixedChunker(spec.get("size", 128), spec.get("overlap", 32), unit)
    if kind == "sentence_window":
        return SentenceWindowChunker(
            spec.get("size", 128), unit, spec.get("window", 4), spec.get("stride", 3)
        )
    if kind == "recursive":
        return RecursiveChunker(spec.get("size", 128), unit)
    raise ValueError(f"Unknown chunker type {kind!r}; use fixed, sentence_window or recursive")


def build_embedder(cfg: dict[str, Any]) -> Embedder:
    kind = cfg.get("type", "sentence_transformers")
    if kind == "hashing":
        return HashingEmbedder(cfg.get("dim", 512))
    if kind == "sentence_transformers":
        from .embedders import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            cfg["model_name"],
            cfg.get("query_prefix", ""),
            cfg.get("document_prefix", ""),
            cfg.get("batch_size", 64),
            cfg.get("device"),
        )
    raise ValueError(f"Unknown embedder type {kind!r}")


def build_generator(cfg: dict[str, Any]) -> Generator:
    kind = cfg.get("type", "transformers")
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "transformers":
        from .generation import TransformersGenerator

        return TransformersGenerator(
            cfg["model_name"], cfg.get("max_new_tokens", 128), cfg.get("device")
        )
    raise ValueError(f"Unknown generator type {kind!r}")


def build_tokenizer(cfg: dict[str, Any]) -> Tokenizer:
    """Context-budget tokenizer: regex for stubs, the generator's own tokenizer otherwise."""
    if cfg.get("type", "transformers") == "transformers":
        return HFTokenizer(cfg["model_name"])
    return RegexTokenizer()


def build_scorer(cfg: dict[str, Any]) -> SupportScorer:
    kind = cfg.get("scorer", "nli")
    if kind == "lexical":
        return LexicalSupportScorer(cfg.get("threshold", 0.7))
    if kind == "nli":
        from .faithfulness import NLIScorer

        return NLIScorer(cfg["model_name"], cfg.get("threshold", 0.5))
    raise ValueError(f"Unknown faithfulness scorer {kind!r}")
