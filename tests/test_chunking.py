import pytest

from rag_eval.chunking import (
    FixedChunker,
    RecursiveChunker,
    SentenceWindowChunker,
    fixed_spans,
    split_sentences,
)
from rag_eval.ingest import Document
from rag_eval.tokens import RegexTokenizer

TOK = RegexTokenizer()


def all_chunkers(unit):
    size = 40 if unit == "tokens" else 200
    return [
        FixedChunker(size, size // 4, unit),
        SentenceWindowChunker(size, unit, window=3, stride=2),
        RecursiveChunker(size, unit),
    ]


def covered_non_space(doc, chunks):
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    return {i for i, ch in enumerate(doc.text) if not ch.isspace()} - covered


@pytest.mark.parametrize("unit", ["tokens", "chars"])
def test_invariants_for_every_chunker(corpus, unit):
    docs, _ = corpus
    doc = docs[0]
    for chunker in all_chunkers(unit):
        chunks = chunker.chunk(doc)
        assert chunks, chunker.name
        for c in chunks:
            assert c.text.strip() == c.text and c.text, "no empty or unstripped chunks"
            assert doc.text[c.start : c.end] == c.text, "span matches text"
        assert covered_non_space(doc, chunks) == set(), f"{chunker.name} loses text ({unit})"
        assert [c.start for c in chunks] == sorted(c.start for c in chunks)


def test_budget_respected_unless_single_unit(corpus):
    docs, _ = corpus
    for chunker in (FixedChunker(40, 10), SentenceWindowChunker(40), RecursiveChunker(40)):
        for c in chunker.chunk(docs[1]):
            assert TOK.count(c.text) <= 40, chunker.name


def test_fixed_overlap_is_exact_in_tokens():
    text = " ".join(f"w{i}" for i in range(30))
    spans = fixed_spans(text, 0, len(text), size=10, overlap=4, unit="tokens")
    toks = [TOK.count(text[a:b]) for a, b in spans]
    assert toks == [10, 10, 10, 10, 6]
    first, second = text[spans[0][0] : spans[0][1]].split(), text[spans[1][0] : spans[1][1]].split()
    assert first[-4:] == second[:4]
    assert spans[-1][1] == len(text)


def test_fixed_overlap_is_exact_in_chars():
    spans = fixed_spans("a" * 25, 0, 25, size=10, overlap=3, unit="chars")
    assert spans == [(0, 10), (7, 17), (14, 24), (21, 25)]


def test_sentence_window_shares_sentences():
    text = "One is here. Two is here. Three is here. Four is here. Five is here."
    doc = Document("d", text)
    chunks = SentenceWindowChunker(100, "tokens", window=3, stride=2).chunk(doc)
    assert [c.text for c in chunks] == [
        "One is here. Two is here. Three is here.",
        "Three is here. Four is here. Five is here.",
    ]


def test_sentence_window_shrinks_to_budget_and_splits_giant_sentence():
    doc = Document("d", "Short one. " + "word " * 50 + "end. Tail here.")
    chunks = SentenceWindowChunker(20, "tokens", window=3, stride=1).chunk(doc)
    assert all(TOK.count(c.text) <= 20 for c in chunks)
    assert covered_non_space(doc, chunks) == set()


def test_recursive_prefers_paragraph_boundaries():
    para_a = "Alpha sentence one. Alpha sentence two."
    para_b = "Beta sentence one. Beta sentence two."
    doc = Document("d", para_a + "\n\n" + para_b)
    chunks = RecursiveChunker(size=12, unit="tokens").chunk(doc)
    assert [c.text for c in chunks] == [para_a, para_b]


def test_recursive_merges_small_paragraphs():
    doc = Document("d", "One.\n\nTwo.\n\nThree.")
    chunks = RecursiveChunker(size=50, unit="tokens").chunk(doc)
    assert len(chunks) == 1 and chunks[0].text == doc.text


def test_recursive_falls_back_to_raw_units_for_long_token():
    doc = Document("d", "x" * 100)
    chunks = RecursiveChunker(size=30, unit="chars").chunk(doc)
    assert [len(c.text) for c in chunks] == [30, 30, 30, 10]


def test_chunking_is_deterministic(corpus):
    docs, _ = corpus
    for chunker in all_chunkers("tokens"):
        assert chunker.chunk(docs[0]) == chunker.chunk(docs[0])


def test_pages_are_assigned():
    doc = Document("d", "First page text.\n\nSecond page text.", (0, 18), (1, 3))
    chunks = FixedChunker(3, 0, "tokens").chunk(doc)
    assert chunks[0].page == 1 and chunks[-1].page == 3


def test_sentence_splitter_protects_abbreviations():
    spans = split_sentences("Dr. Smith arrived. He sat down.")
    assert [("Dr. Smith arrived. He sat down.")[a:b] for a, b in spans] == [
        "Dr. Smith arrived.",
        "He sat down.",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        lambda: FixedChunker(10, 10),
        lambda: FixedChunker(0, 0),
        lambda: FixedChunker(10, 2, "words"),
        lambda: SentenceWindowChunker(10, window=2, stride=3),
        lambda: RecursiveChunker(0),
    ],
)
def test_bad_parameters_fail_loudly(bad):
    with pytest.raises(ValueError):
        bad()
