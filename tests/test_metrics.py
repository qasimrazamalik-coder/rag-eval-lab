import math

import pytest

from rag_eval.chunking import Chunk
from rag_eval.evalset import GoldItem
from rag_eval.metrics import (
    bootstrap_ci,
    is_relevant,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def chunk(start, end, doc="d"):
    return Chunk(doc, f"{doc}#{start}", start, end, "x" * (end - start), 1)


GOLD = GoldItem("q", "?", "a", "d", 100, 200)


def test_overlap_rule_hand_cases():
    assert is_relevant(chunk(100, 200), GOLD)  # identical
    assert is_relevant(chunk(150, 180), GOLD)  # small chunk inside gold
    assert is_relevant(chunk(0, 1000), GOLD)  # big chunk containing gold
    assert is_relevant(chunk(150, 250), GOLD)  # 50 / min(100, 100) = 0.5, boundary counts
    assert not is_relevant(chunk(160, 260), GOLD)  # 40 / 100
    assert not is_relevant(chunk(200, 300), GOLD)  # touching, zero overlap
    assert not is_relevant(chunk(100, 200, doc="other"), GOLD)


def test_overlap_rule_uses_smaller_of_gold_and_chunk():
    tiny_gold = GoldItem("q", "?", "a", "d", 100, 110)
    assert is_relevant(chunk(0, 1000), tiny_gold)  # gold fully inside a large chunk
    assert is_relevant(chunk(105, 400), tiny_gold)  # 5 / min(10, 295) = 0.5 is exactly tau
    assert not is_relevant(chunk(106, 400), tiny_gold)  # 4 / 10 = 0.4


def test_overlap_threshold_is_configurable():
    assert not is_relevant(chunk(160, 260), GOLD, tau=0.5)
    assert is_relevant(chunk(160, 260), GOLD, tau=0.4)


def test_recall_at_k():
    flags = [False, False, True, False, True]
    assert [recall_at_k(flags, k) for k in (1, 2, 3, 5)] == [0.0, 0.0, 1.0, 1.0]
    assert recall_at_k([False] * 10, 10) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([True, True]) == 1.0
    assert reciprocal_rank([False] * 10 + [True], cutoff=10) == 0.0


def test_ndcg_hand_computed():
    # one relevant chunk at rank 3, one relevant in index: DCG = 1/log2(4) = 0.5, IDCG = 1
    assert ndcg_at_k([False, False, True], 10, 1) == pytest.approx(0.5)
    # relevant at ranks 1 and 3, two relevant exist: DCG = 1 + 0.5, IDCG = 1 + 1/log2(3)
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k([True, False, True], 10, 2) == pytest.approx(expected)
    # ideal ranking scores 1
    assert ndcg_at_k([True, True, False], 10, 2) == pytest.approx(1.0)
    # IDCG is capped at k
    assert ndcg_at_k([True], 1, 5) == pytest.approx(1.0)
    assert ndcg_at_k([False, False], 10, 0) == 0.0


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    values = [1.0] * 30 + [0.0] * 70
    lo, hi = bootstrap_ci(values, n_boot=500, seed=3)
    assert lo < 0.3 < hi
    assert (lo, hi) == bootstrap_ci(values, n_boot=500, seed=3)
    assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)
