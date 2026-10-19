import numpy as np
import pytest

from rag_eval.bm25 import BM25Index
from rag_eval.chunking import RecursiveChunker
from rag_eval.embedders import HashingEmbedder
from rag_eval.evaluate import evaluate_chunker
from rag_eval.retrieval import MODES, Retriever, _rrf


def test_hashing_embedder_is_unit_norm_and_stable():
    emb = HashingEmbedder(128)
    a = emb.encode_documents(["The quick brown fox", "jumps over"])
    b = HashingEmbedder(128).encode_documents(["The quick brown fox", "jumps over"])
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32


def test_dense_search_finds_exact_text(corpus):
    docs, _ = corpus
    chunks = RecursiveChunker(40).chunk(docs[0])
    retriever = Retriever(chunks, HashingEmbedder(512))
    for target in chunks:
        hits = retriever.search(target.text, 3, "dense")
        assert hits[0].chunk.chunk_id == target.chunk_id
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)
        assert [h.rank for h in hits] == [1, 2, 3]


def test_bm25_hand_computed_ordering():
    index = BM25Index(["apple banana", "banana banana cherry", "cherry date"])
    s = index.scores("banana")
    assert s[1] > s[0] > 0 and s[2] == 0
    assert index.scores("unseen").sum() == 0


def test_all_modes_return_k_sorted_hits(corpus):
    docs, items = corpus
    chunks = [c for d in docs for c in RecursiveChunker(40).chunk(d)]
    retriever = Retriever(chunks, HashingEmbedder(512))
    for mode in MODES:
        rows = retriever.search_many([items[0].question, items[1].question], 5, mode)
        assert len(rows) == 2 and all(len(r) == 5 for r in rows)
        for row in rows:
            scores = [h.score for h in row]
            assert scores == sorted(scores, reverse=True)


def test_rrf_hand_computed():
    fused = dict(_rrf([(1, 0.0), (2, 0.0)], [(2, 0.0), (3, 0.0)]))
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1] == pytest.approx(1 / 61)
    assert max(fused, key=fused.get) == 2


def test_bad_mode_and_empty_index():
    with pytest.raises(ValueError):
        Retriever([], HashingEmbedder())
    chunks = RecursiveChunker(40).chunk(
        __import__("rag_eval.fixtures").fixtures.generate_corpus(1, 2)[0][0]
    )
    with pytest.raises(ValueError):
        Retriever(chunks, HashingEmbedder()).search("q", 1, "nope")


def test_evaluate_chunker_end_to_end_with_stub(corpus):
    docs, items = corpus
    spec = {"type": "recursive", "unit": "tokens", "size": 40}
    res = evaluate_chunker(docs, items, spec, HashingEmbedder(512), ["bm25", "hybrid"], 0.5, 50, 0)
    assert res["questions_without_relevant_chunk"] == 0
    bm = res["retrievers"]["bm25"]
    assert bm["recall@10"]["mean"] >= bm["recall@1"]["mean"]
    assert bm["recall@10"]["mean"] > 0.9  # fixture facts are lexically distinctive
    for metrics in res["retrievers"].values():
        for summary in metrics.values():
            lo, hi = summary["ci95"]
            assert lo <= summary["mean"] <= hi
