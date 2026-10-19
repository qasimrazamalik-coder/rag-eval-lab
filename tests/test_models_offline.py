import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("sentence_transformers")

from rag_eval.context import ContextBundle
from rag_eval.embedders import SentenceTransformerEmbedder
from rag_eval.faithfulness import NLIScorer, faithfulness
from rag_eval.generation import TransformersGenerator
from rag_eval.tokens import HFTokenizer
from tiny_models import tiny_causal_lm, tiny_nli


def test_nli_scorer_reads_entailment_label_from_config(tmp_path):
    scorer = NLIScorer(str(tiny_nli(tmp_path / "nli")), threshold=0.0)
    assert scorer._entail_idx == 1
    s = scorer.score(["the mill employs forty people"], "the mill employs people")
    assert 0.0 <= s <= 1.0
    assert faithfulness("the mill employs people.", ["the mill"], scorer).score == 1.0
    assert scorer.score([], "x") == 0.0


def test_sentence_transformer_wrapper_on_local_model(tmp_path):
    emb = SentenceTransformerEmbedder(str(tiny_nli(tmp_path / "bert")), query_prefix="the ")
    vecs = emb.encode_documents(["the mill", "forty people"])
    assert vecs.shape[0] == 2 and vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert emb.encode_queries(["mill"]).shape == (1, vecs.shape[1])


def test_missing_weights_message_is_actionable(tmp_path):
    with pytest.raises(RuntimeError, match="Could not load"):
        TransformersGenerator(str(tmp_path / "nope"))


def test_transformers_generator_and_hf_tokenizer_plumbing(tmp_path):
    path = str(tiny_causal_lm(tmp_path / "lm"))
    gen = TransformersGenerator(path, max_new_tokens=5, device="cpu")
    ctx = ContextBundle("the mill", ("the mill",), 2, False)
    out = gen.generate("the mill", ctx)
    assert out.prompt_tokens > 0 and 0 < out.completion_tokens <= 5
    tok = HFTokenizer(path)
    assert tok.count("the mill employs") == 3
    assert tok.truncate("the mill employs forty", 2).split() == ["the", "mill"]
