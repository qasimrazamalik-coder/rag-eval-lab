import pytest

from rag_eval.fixtures import generate_corpus


@pytest.fixture(scope="session")
def corpus():
    return generate_corpus(n_docs=3, paragraphs=4, seed=1)
