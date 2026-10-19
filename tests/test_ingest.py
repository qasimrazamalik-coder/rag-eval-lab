import pytest

from rag_eval.chunking import RecursiveChunker
from rag_eval.fixtures import make_pdf
from rag_eval.ingest import IngestError, build_document, clean_page_text, extract_pdf

PAGES = [
    ["The Quillmere Archive was founded in 1904 by Anna Fell.", "It holds maps and letters."],
    ["A second page begins here. The reading room seats forty visitors."],
    [],
    ["The fourth page is the last one."],
]


def test_pdf_extraction_text_and_pages(tmp_path):
    pdf = make_pdf(tmp_path / "sample.pdf", PAGES)
    doc = extract_pdf(pdf)
    assert doc.doc_id == "sample"
    assert "founded in 1904 by Anna Fell." in doc.text
    assert doc.page_numbers == (1, 2, 4)  # blank page 3 skipped, numbering preserved
    assert doc.page_of(doc.text.index("reading room")) == 2
    assert doc.page_of(doc.text.index("fourth page")) == 4
    assert doc.page_of(0) == 1


def test_pdf_paragraph_gaps_survive_for_the_structure_aware_chunker(tmp_path):
    doc = extract_pdf(make_pdf(tmp_path / "p.pdf", [PAGES[0]]))
    assert "\n\n" in doc.text
    chunks = RecursiveChunker(size=14, unit="tokens").chunk(doc)
    assert chunks[0].text.endswith("Anna Fell.")


def test_pdf_validation_errors(tmp_path):
    with pytest.raises(IngestError, match=r"Only \.pdf"):
        extract_pdf(tmp_path / "x.txt")
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"not a pdf at all")
    with pytest.raises(IngestError, match="PDF header"):
        extract_pdf(fake)
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(IngestError, match="empty"):
        extract_pdf(empty)
    big = make_pdf(tmp_path / "big.pdf", PAGES)
    with pytest.raises(IngestError, match="limit"):
        extract_pdf(big, max_bytes=100)
    with pytest.raises(IngestError, match="not found"):
        extract_pdf(tmp_path / "missing.pdf")


def test_pdf_without_text_is_rejected(tmp_path):
    blank = make_pdf(tmp_path / "blank.pdf", [[]])
    with pytest.raises(IngestError, match="OCR"):
        extract_pdf(blank)


def test_clean_page_text():
    raw = (
        "Intro line one\nline two of the same para.\n\n\n"
        "Next para with exam-\nple hyphen and ﬁne ligature.\x00"
    )
    assert clean_page_text(raw) == (
        "Intro line one line two of the same para.\n\n"
        "Next para with example hyphen and fine ligature."
    )


def test_running_headers_and_page_numbers_removed():
    pages = [f"Annual Report 2024\nBody text number {i}.\n{i}" for i in range(1, 6)]
    doc = build_document("r", pages)
    assert "Annual Report" not in doc.text
    assert all(f"Body text number {i}." in doc.text for i in range(1, 6))
    assert "\n5" not in doc.text
