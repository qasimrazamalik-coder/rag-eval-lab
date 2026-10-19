import pytest

import app as app_module
from rag_eval.embedders import HashingEmbedder
from rag_eval.fixtures import make_pdf
from rag_eval.generation import ExtractiveGenerator
from rag_eval.utils import load_config

CFG = load_config("configs/smoke.yaml")


@pytest.fixture
def demo():
    return app_module.build_app(
        CFG, embedder_factory=lambda: HashingEmbedder(256), generator_factory=ExtractiveGenerator
    )


def test_app_builds_with_empty_state_and_collapsed_sections(demo):
    labels = {getattr(b, "label", None) for b in demo.blocks.values()}
    assert {"Question", "Answer", "Retrieved chunks", "Chunker"} <= labels
    accordions = [b for b in demo.blocks.values() if type(b).__name__ == "Accordion"]
    assert {a.label for a in accordions} == {"How it works", "Limitations"}
    assert all(a.open is False for a in accordions)


def test_upload_validation_messages(tmp_path):
    check = app_module._validate_upload
    assert "Upload a PDF" in check(None, 20)
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    assert "Only .pdf" in check(str(txt), 20)
    big = tmp_path / "b.pdf"
    big.write_bytes(b"%PDF-" + b"0" * 2 * 1024 * 1024)
    assert "limit" in check(str(big), 1)
    assert check(str(big), 20) is None


def test_index_and_ask_roundtrip(demo, tmp_path):
    pdf = make_pdf(
        tmp_path / "doc.pdf",
        [["The Quillmere Archive was founded in 1904 by Anna Fell.", "Rain fell all week."]],
    )
    fns = {f.fn.__name__: f.fn for f in demo.fns.values()}
    pipe, status = fns["index"](str(pdf), "recursive", "hybrid")
    assert pipe is not None and "Indexed" in status
    answer, rows, tokens = fns["ask"](pipe, "Who founded the Quillmere Archive?")
    assert "Anna Fell" in answer
    assert rows[0][0] == 1 and rows[0][2] == 1  # rank 1, page 1
    assert next(t[0] for t in tokens).startswith("Context")


def test_ask_without_index_shows_empty_state_and_bad_inputs_are_rejected(demo, tmp_path):
    fns = {f.fn.__name__: f.fn for f in demo.fns.values()}
    assert "No document is indexed" in fns["ask"](None, "q")[0]
    pipe, status = fns["index"](None, "recursive", "dense")
    assert pipe is None and "Upload" in status
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage")
    pipe, status = fns["index"](str(bad), "recursive", "dense")
    assert pipe is None and status
