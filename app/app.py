"""Gradio app: upload a PDF, pick a chunker, ask a question, inspect what was retrieved."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from rag_eval.factory import (
    build_chunker,
    build_embedder,
    build_generator,
    build_tokenizer,
)
from rag_eval.generation import ExtractiveGenerator
from rag_eval.ingest import IngestError, extract_pdf
from rag_eval.pipeline import RagPipeline
from rag_eval.retrieval import MODES
from rag_eval.tokens import RegexTokenizer
from rag_eval.utils import load_config
from theme import CSS, build_theme

EMPTY = "No document is indexed. Upload a PDF, choose a chunker and press **Index document**."
HOW = (
    "The PDF text is extracted with pypdf and cleaned. It is cut into chunks by the strategy you "
    "pick, embedded with the configured sentence-transformers model and stored in an exact FAISS "
    "index. A question retrieves the top chunks, which are truncated to a token budget and passed "
    "to the generator. If the generator cannot be loaded, the app returns the best matching "
    "sentence instead and says so. Nothing leaves this machine at question time."
)
LIMITATIONS = (
    "- Text PDFs only. Scanned pages have no text layer and are not OCR'd.\n"
    "- Tables, multi-column layouts and footnotes may extract in the wrong order.\n"
    "- A small model can answer wrongly while sounding sure; check the cited chunks and pages.\n"
    "- Scores are cosine similarity, BM25 or fused rank scores, not probabilities of being "
    "correct.\n"
    "- Retrieval quality depends on the chunker. The evaluation in the README compares them on one "
    "benchmark, and your document may behave differently."
)
CHUNK_HEADERS = ["Rank", "Score", "Page", "Characters", "Text"]
TOKEN_HEADERS = ["Quantity", "Tokens"]


def _validate_upload(path: str | None, max_mb: int) -> str | None:
    """Return an error message, or None if the upload is acceptable."""
    if not path:
        return "Upload a PDF first."
    p = Path(path)
    if p.suffix.lower() != ".pdf":
        return "Only .pdf files are accepted."
    if p.stat().st_size > max_mb * 1024 * 1024:
        return f"File is larger than the {max_mb} MB limit."
    return None


def build_app(
    cfg: dict[str, Any],
    embedder_factory: Callable[[], Any] | None = None,
    generator_factory: Callable[[], Any] | None = None,
) -> gr.Blocks:
    """Build the UI. Factories let tests inject stub models; models load on first use."""
    app_cfg = cfg["app"]
    cache: dict[str, Any] = {}

    def embedder():
        if "embedder" not in cache:
            cache["embedder"] = (embedder_factory or (lambda: build_embedder(cfg["embedder"])))()
        return cache["embedder"]

    def generator() -> tuple[Any, Any, str]:
        if "generator" not in cache:
            try:
                gen = (generator_factory or (lambda: build_generator(cfg["generator"])))()
                tok = RegexTokenizer() if generator_factory else build_tokenizer(cfg["generator"])
                note = ""
            except Exception as exc:
                gen, tok = ExtractiveGenerator(), RegexTokenizer()
                note = f"Generator unavailable ({exc}). Showing the best matching sentence instead."
            cache["generator"] = (gen, tok, note)
        return cache["generator"]

    def index(path, chunker_name, mode):
        error = _validate_upload(path, app_cfg["max_upload_mb"])
        if error:
            return None, error
        specs = {s.get("name", s["type"]): s for s in cfg["chunkers"]}
        try:
            doc = extract_pdf(path, max_bytes=app_cfg["max_upload_mb"] * 1024 * 1024)
            gen, tok, note = generator()
            pipe = RagPipeline(
                [doc],
                build_chunker(specs[chunker_name]),
                embedder(),
                gen,
                tok,
                mode=mode,
                top_k=app_cfg["top_k"],
                context_budget=app_cfg["context_budget"],
            )
            pipe.retriever.search("warm up", 1, mode)
        except IngestError as exc:
            return None, str(exc)
        except RuntimeError as exc:
            return None, f"Model could not be loaded: {exc}"
        pages = len(doc.page_numbers)
        status = (
            f"Indexed `{doc.doc_id}`: {pages} pages with text, {len(doc.text):,} characters, "
            f"{pipe.n_chunks} chunks ({chunker_name}, {mode})."
        )
        return pipe, status + (f"\n\n{note}" if note else "")

    def ask(pipe, question):
        if pipe is None:
            return EMPTY, [], []
        if not question or not question.strip():
            return "Type a question first.", [], []
        if len(question) > 1000:
            return "Questions are limited to 1000 characters.", [], []
        ans = pipe.ask(question)
        rows = [
            [h.rank, round(h.score, 4), h.chunk.page, len(h.chunk.text), h.chunk.text]
            for h in ans.hits
        ]
        tokens = [
            ["Context passed to generator", ans.context.tokens],
            ["Generator prompt", ans.prompt_tokens],
            ["Generator output", ans.completion_tokens],
        ]
        text = ans.text + (
            "\n\n(context was truncated to the token budget)" if ans.context.truncated else ""
        )
        return text, rows, tokens

    with gr.Blocks(title="RAG Eval Lab") as demo:
        gr.Markdown(
            "# RAG Eval Lab\nAsk questions about a PDF and inspect exactly what was retrieved."
        )
        pipe_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="PDF (max 20 MB)", file_types=[".pdf"], type="filepath")
                chunker = gr.Dropdown(
                    [s.get("name", s["type"]) for s in cfg["chunkers"]],
                    value=cfg["chunkers"][-1].get("name", cfg["chunkers"][-1]["type"]),
                    label="Chunker",
                )
                mode = gr.Dropdown(list(MODES), value="dense", label="Retriever")
                index_btn = gr.Button("Index document", variant="primary")
                status = gr.Markdown(EMPTY, elem_classes="notice")
            with gr.Column(scale=2):
                question = gr.Textbox(label="Question", lines=2, max_length=1000)
                ask_btn = gr.Button("Ask")
                answer = gr.Textbox(label="Answer", lines=4, interactive=False)
                tokens = gr.Dataframe(headers=TOKEN_HEADERS, label="Tokens used", interactive=False)
        chunks = gr.Dataframe(
            headers=CHUNK_HEADERS, label="Retrieved chunks", interactive=False, wrap=True
        )
        with gr.Accordion("How it works", open=False):
            gr.Markdown(HOW)
        with gr.Accordion("Limitations", open=False):
            gr.Markdown(LIMITATIONS)

        index_btn.click(index, [upload, chunker, mode], [pipe_state, status])
        ask_btn.click(ask, [pipe_state, question], [answer, chunks, tokens])
        question.submit(ask, [pipe_state, question], [answer, chunks, tokens])
    return demo


def main() -> None:
    cfg = load_config(os.environ.get("RAG_EVAL_CONFIG", ROOT / "configs" / "default.yaml"))
    build_app(cfg).launch(server_name="127.0.0.1", server_port=7860, theme=build_theme(), css=CSS)


if __name__ == "__main__":
    main()
