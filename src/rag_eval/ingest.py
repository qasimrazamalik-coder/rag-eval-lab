"""PDF text extraction and cleaning.

pypdf (BSD-3-Clause) is used instead of PyMuPDF (AGPL-3.0 or a commercial licence): this repo is
MIT and the Gradio app accepts untrusted uploads, so a permissive, pure-python dependency avoids
licence obligations. The trade-off is that pypdf's extraction is less robust on complex layouts.
No OCR is attempted; image-only PDFs are rejected with a clear message.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PARA_SEP = "\n\n"
MAX_PDF_BYTES = 20 * 1024 * 1024

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f­]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n\s*([a-z])")
_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")


class IngestError(ValueError):
    """Raised with a user-readable message when a document cannot be ingested."""


@dataclass(frozen=True)
class Document:
    """Cleaned text of one source plus where each page starts inside it.

    ``page_starts[i]`` is the character offset where the page numbered ``page_numbers[i]`` begins.
    Pages with no text are omitted, so page numbers can skip.
    """

    doc_id: str
    text: str
    page_starts: tuple[int, ...] = (0,)
    page_numbers: tuple[int, ...] = (1,)

    def page_of(self, offset: int) -> int:
        """Page number containing the character at ``offset``."""
        i = bisect_right(self.page_starts, max(offset, 0)) - 1
        return self.page_numbers[max(i, 0)]


def clean_page_text(raw: str) -> str:
    """Normalise one page: fix ligatures and hyphenation, keep paragraph breaks only.

    Paragraphs are separated by blank lines in the extracted text; lines inside a paragraph are
    joined with a single space so a sentence is never broken by a visual line wrap.
    """
    text = unicodedata.normalize("NFKC", raw).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    paragraphs = []
    for block in _BLANK_LINES_RE.split(text):
        lines = [_MULTISPACE_RE.sub(" ", ln).strip() for ln in block.split("\n")]
        joined = " ".join(ln for ln in lines if ln)
        if joined:
            paragraphs.append(joined)
    return PARA_SEP.join(paragraphs)


def _strip_running_lines(pages: list[str]) -> list[str]:
    """Drop bare page numbers and header/footer lines repeated on most pages."""
    split = [p.split("\n") for p in pages]
    edge = Counter()
    for lines in split:
        non_empty = [ln.strip() for ln in lines if ln.strip()]
        for ln in {*non_empty[:2], *non_empty[-2:]}:
            edge[ln] += 1
    repeated = (
        {ln for ln, c in edge.items() if c >= max(3, 0.6 * len(pages))}
        if len(pages) >= 4
        else set()
    )
    out = []
    for lines in split:
        kept = [ln for ln in lines if ln.strip() not in repeated and not _PAGE_NUMBER_RE.match(ln)]
        out.append("\n".join(kept))
    return out


def build_document(doc_id: str, pages: list[str]) -> Document:
    """Assemble a Document from raw per-page text, skipping pages that clean to nothing."""
    cleaned = [clean_page_text(p) for p in _strip_running_lines(pages)]
    texts: list[str] = []
    starts: list[int] = []
    numbers: list[int] = []
    pos = 0
    for number, page in enumerate(cleaned, start=1):
        if not page:
            continue
        if texts:
            pos += len(PARA_SEP)
        starts.append(pos)
        numbers.append(number)
        texts.append(page)
        pos += len(page)
    if not texts:
        raise IngestError(
            "No extractable text found. The PDF may be scanned images; OCR is not supported."
        )
    return Document(doc_id, PARA_SEP.join(texts), tuple(starts), tuple(numbers))


def extract_pdf(path: str | Path, max_bytes: int = MAX_PDF_BYTES, max_pages: int = 500) -> Document:
    """Read a PDF into a Document, validating extension, size, header and page count first."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    p = Path(path)
    if p.suffix.lower() != ".pdf":
        raise IngestError(f"Only .pdf files are supported, got '{p.suffix or 'no extension'}'.")
    if not p.is_file():
        raise IngestError(f"File not found: {p}")
    size = p.stat().st_size
    if size == 0:
        raise IngestError("The file is empty.")
    if size > max_bytes:
        raise IngestError(
            f"File is {size / 2**20:.1f} MB; the limit is {max_bytes / 2**20:.0f} MB."
        )
    with p.open("rb") as fh:
        if fh.read(5) != b"%PDF-":
            raise IngestError("The file does not start with a PDF header.")
    try:
        reader = PdfReader(str(p), strict=False)
        if reader.is_encrypted:
            raise IngestError("The PDF is encrypted; remove the password first.")
        if len(reader.pages) > max_pages:
            raise IngestError(f"PDF has {len(reader.pages)} pages; the limit is {max_pages}.")
        pages = [_page_text(page) for page in reader.pages]
    except PyPdfError as exc:
        raise IngestError(f"Could not parse the PDF: {exc}") from exc
    return build_document(p.stem, pages)


def _page_text(page) -> str:
    """Layout mode keeps vertical gaps as blank lines, which is the only paragraph signal."""
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except Exception:
        return page.extract_text() or ""
