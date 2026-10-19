import pytest

from rag_eval.answer_eval import token_f1
from rag_eval.chunking import Chunk
from rag_eval.context import build_context
from rag_eval.embedders import HashingEmbedder
from rag_eval.faithfulness import LexicalSupportScorer, faithfulness
from rag_eval.generation import ExtractiveGenerator
from rag_eval.pipeline import RagPipeline
from rag_eval.prompts import ABSTAIN
from rag_eval.retrieval import Hit
from rag_eval.tokens import RegexTokenizer

TOK = RegexTokenizer()


def hit(text, rank=1, page=1):
    return Hit(Chunk("d", f"d#{rank}", 0, len(text), text, page), 1.0, rank)


def test_context_truncation_is_deterministic_and_within_budget():
    hits = [hit("one two three four five six " * 5, 1), hit("alpha beta gamma delta " * 5, 2, 3)]
    a = build_context(hits, TOK, budget=40)
    b = build_context(hits, TOK, budget=40)
    assert a == b
    assert a.tokens <= 40 and a.truncated
    assert a.text.startswith("[1] (page 1)")


def test_context_keeps_everything_when_it_fits():
    hits = [hit("short passage."), hit("another one.", 2, 2)]
    ctx = build_context(hits, TOK, budget=500)
    assert not ctx.truncated and ctx.passages == ("short passage.", "another one.")
    assert "[2] (page 2)" in ctx.text


def test_extractive_returns_best_sentence_or_abstains():
    hits = [
        hit("The mill closed in 1999. The Orwell Mill employs forty people. Rain fell."),
    ]
    ctx = build_context(hits, TOK, 200)
    gen = ExtractiveGenerator()
    assert (
        gen.generate("How many people does the Orwell Mill employ?", ctx).answer
        == "The Orwell Mill employs forty people."
    )
    assert gen.generate("zzz qqq?", ctx).answer == ABSTAIN


def test_lexical_faithfulness_hand_cases():
    scorer = LexicalSupportScorer(threshold=0.7)
    premises = ["The Orwell Mill employs forty people at its main site."]
    full = faithfulness("The Orwell Mill employs forty people.", premises, scorer)
    assert full.score == 1.0 and full.n_sentences == 1
    mixed = faithfulness(
        "The Orwell Mill employs forty people. It exports tea to Brazil.", premises, scorer
    )
    assert mixed.score == 0.5 and mixed.n_supported == 1
    assert faithfulness(ABSTAIN, premises, scorer).abstained
    assert faithfulness(ABSTAIN, premises, scorer).score is None


def test_token_f1():
    assert token_f1("The Eiffel Tower", "eiffel tower") == 1.0
    assert token_f1("blue", "red") == 0.0
    assert token_f1("big red car", "red car") == pytest.approx(0.8)


def test_pipeline_with_stubs_logs_tokens(corpus, caplog):
    from rag_eval.chunking import RecursiveChunker

    docs, items = corpus
    pipe = RagPipeline(
        docs,
        RecursiveChunker(40),
        HashingEmbedder(512),
        ExtractiveGenerator(),
        TOK,
        mode="hybrid",
        top_k=3,
        context_budget=120,
    )
    with caplog.at_level("INFO", logger="rag_eval.pipeline"):
        ans = pipe.ask(items[0].question)
    assert len(ans.hits) == 3 and ans.context.tokens <= 120
    assert any("query tokens" in r.message for r in caplog.records)
    with pytest.raises(ValueError):
        pipe.ask("   ")
