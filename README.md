# RAG Eval Lab

Question answering over PDFs, with a retrieval evaluation that compares three chunking strategies (fixed window, sentence window, recursive/structure-aware) and three retrievers (dense, BM25, hybrid) on the same questions, and an operational faithfulness check on the generated answers.

## Results

Retrieval metrics per chunker and retriever (Recall@k, MRR@10, nDCG@10). Written by `make report` from `results/retrieval_metrics.json`; nothing here is typed by hand.

<!-- RESULTS:START -->
not yet run. Fill with: `make data && make eval && make report` (GPU not required for retrieval; see "Reproducing on free Colab or Kaggle").
<!-- RESULTS:END -->

Answer-level metrics (token F1, faithfulness, abstention rate, tokens per query): not yet run. `make answers` writes `results/answer_metrics.json`.

## Demo

Screenshot added after first run: `docs/demo.png`.

## What question this answers

Most RAG demos pick a chunk size and move on. Chunking decides what the retriever can find at all, so this repo asks: for a fixed embedder, fixed budget and fixed questions, how much do recall and ranking quality change when only the splitting logic changes, and does BM25 or hybrid retrieval close the gap? The answer is measured against gold answer spans, with bootstrap confidence intervals, instead of judged by eye on a few queries.

## Quickstart

```bash
git clone https://github.com/qasimrazamalik-coder/rag-eval-lab.git
cd rag-eval-lab
python -m venv .venv && source .venv/bin/activate
make install
make test          # CPU, stub models, no downloads
make smoke         # end-to-end on a synthetic fixture (a pipeline check, not a result)
make data          # download SQuAD v2 dev and build docs.jsonl + eval.jsonl
make eval          # retrieval metrics -> results/retrieval_metrics.json
make report        # render the table above into this README
make answers       # generator + faithfulness on a sample of questions
make app           # Gradio app at http://127.0.0.1:7860
```

Models (`all-MiniLM-L6-v2`, `Qwen2.5-0.5B-Instruct`, `nli-deberta-v3-small`) are downloaded from Hugging Face on first use and then read from the local cache. The app does not touch the network when you ask a question.

## Method

**Ingestion.** PDF text is extracted with `pypdf` (BSD-3-Clause). PyMuPDF extracts better on hard layouts but is AGPL-3.0; for an MIT repo that accepts uploaded files, a permissive pure-python dependency was the safer choice. Cleaning applies NFKC normalisation (fixes ligatures), removes control characters and soft hyphens, re-joins words hyphenated across lines, drops bare page numbers and headers repeated on most pages, and keeps blank-line paragraph breaks. Extraction uses pypdf's layout mode because it is the only mode that preserves vertical gaps as blank lines, which the recursive chunker needs. Scanned PDFs are rejected; there is no OCR.

**Chunkers.** All three share one budget (default 128 regex tokens, configurable as tokens or characters) and return character spans, so each chunk maps back to its page and to gold spans.

| Chunker | Rule |
| --- | --- |
| `fixed` | Sliding window of `size` units, stride `size - overlap` (default overlap 32). |
| `sentence_window` | `window` consecutive sentences (default 4), stride 3, shrunk to fit the budget. A single over-budget sentence is hard-split. |
| `recursive` | Split on paragraphs, then sentences, then words, then raw units, then merge neighbours back up to the budget. No overlap. |

"Tokens" are words and punctuation marks matched by a regex. That is deterministic and needs no model, but it is not the embedder's subword count; keep the budget well under the embedder's maximum sequence length (256 for all-MiniLM-L6-v2).

**Retrievers.** Dense: sentence-transformers embeddings (model is a config value; `bge-small-en-v1.5` works with its query prefix) in an exact FAISS inner-product index. BM25: a small Okapi implementation in this repo. Hybrid: reciprocal rank fusion (k = 60) of the top 50 from each. Ties break toward the lower chunk index so runs repeat.

**Evaluation set.** JSONL with `question`, optional `answer`, and either `gold_doc` + `gold_start` + `gold_end` (character offsets) or `gold_passage_text` (located by whitespace-tolerant exact match; missing or ambiguous passages raise an error). `scripts/build_evalset.py` converts any SQuAD-format JSON. By default all paragraphs of an article form one document so chunkers face real multi-paragraph text; `--granularity context` makes each paragraph its own document, which is easier. Unanswerable SQuAD v2 questions are dropped because they have no gold span. The gold span is the answer span itself.

**Relevance rule.** A chunk is relevant if it is in the gold document and `overlap_chars / min(len(gold), len(chunk)) >= tau` (default `tau = 0.5`). Either the chunk is mostly inside the gold span or the gold span is mostly inside the chunk.

**Metrics.** Recall@k (k = 1, 3, 5, 10) is a hit rate: 1 if any relevant chunk is in the top k. MRR@10 uses the first relevant rank within the top 10. nDCG@10 uses binary gains, where the ideal ranking holds min(10, relevant chunks in the index) relevant chunks. The JSON also stores 95% percentile-bootstrap intervals over questions (1000 resamples), the chunk count, mean chunk length, and the number of questions with no relevant chunk (a chunker bug indicator; expected 0).

**The one thing that keeps the comparison fair.** Same documents, same questions, same embedder, same gold spans, same relevance rule and same token budget for every chunker. Only the splitting logic differs. This does not make chunk size irrelevant: the strategies still produce different chunk counts and lengths (reported next to the scores), and any chunker that happens to produce longer chunks has an easier time on a containment-based rule. Read the table with the "Mean tokens/chunk" column next to it.

**Generation.** `prompts.py` holds the prompts with a one-line purpose each. Retrieved chunks are added in rank order under a token budget (default 1024 generator tokens); the first chunk that does not fit is cut and the rest are dropped, so the same inputs always give the same context. Tokens used (context, prompt, completion) are logged per query and shown in the app. The default generator is `Qwen/Qwen2.5-0.5B-Instruct` with greedy decoding. If it cannot be loaded, the app falls back to an extractive generator that returns the sentence sharing the most content words with the question, or abstains.

**Faithfulness.** Defined operationally: split the answer into sentences; a sentence counts as supported if at least one retrieved chunk, taken alone as the premise, has an entailment probability >= 0.5 from an NLI cross-encoder (`cross-encoder/nli-deberta-v3-small` by default). Faithfulness is supported sentences divided by answer sentences. Abstentions ("I don't know.") are excluded and reported as an abstention rate. A model-free lexical fallback (share of the sentence's content words found in a chunk, threshold 0.7) is available as `faithfulness.scorer: lexical`. Weaknesses:

- Single-chunk premises miss answers that combine facts from two chunks.
- The NLI model is imperfect, especially on text unlike its training data, and the 0.5 threshold is a free parameter.
- A faithful answer can still be wrong if the retrieved context is wrong, and a short correct answer can score as unfaithful if phrased very differently from the context.
- The lexical fallback rewards copying and cannot detect negation or swapped entities.
- It measures support, not completeness or correctness. Token F1 against the reference answer is reported alongside it for that reason, and is itself blunt for sentence-length answers.

## Repository structure

```
configs/    default.yaml (real run), smoke.yaml (synthetic fixture, stub models)
src/rag_eval/
  ingest.py chunking.py tokens.py        PDF -> cleaned text -> chunks
  embedders.py bm25.py retrieval.py      dense, BM25, hybrid
  evalset.py metrics.py evaluate.py      eval format, metrics, harness CLI
  report.py                              results JSON -> README table
  prompts.py context.py generation.py    answer generation
  faithfulness.py answer_eval.py         faithfulness and answer-level metrics
  squad.py fixtures.py                   SQuAD conversion, synthetic fixture and test PDF
app/        Gradio app and theme
scripts/    download_data.py, build_evalset.py
tests/      stub-based tests (plus tiny random local models for the wrapper code)
results/    outputs from real runs
```

## Limitations

- Retrieval numbers come from SQuAD, whose questions were written by annotators looking at the passage, so they share many words with it. That flatters lexical methods (BM25) and may understate the benefit of dense retrieval on real user questions. Results on your own PDFs may rank the chunkers differently.
- The gold span is the answer span, so "relevant" means "contains or is contained in the answer", not "sufficient to answer".
- Sentence splitting is regex-based and mis-splits some abbreviations and initials; it does not split sentences that begin with a lowercase letter.
- PDF extraction with pypdf can misorder columns, tables and footnotes, and paragraph gaps are inferred from layout. The cleaned text is what the gold offsets refer to, so an eval set built from one extraction is tied to that pypdf version.
- BM25 has no stemming or stopword handling by design, and a query with no matching term gets arbitrary zero-score results.
- A 0.5B generator is weak. Expect incomplete answers and occasional wrong ones that read confidently.
- The synthetic fixture (invented entity names, templated sentences) and the stub hashing embedder exist only to test the code. Their output is never put in the table.
- The app indexes one PDF at a time, up to 20 MB, and holds it in memory.

## Reproducing on free Colab or Kaggle

Retrieval evaluation with MiniLM runs on CPU; a GPU only speeds up embedding and the generator. Runtime depends on the assigned hardware and has not been measured for this repo yet; embedding about a thousand questions' worth of chunks with MiniLM is a small job on a T4 and slower but feasible on CPU.

```bash
git clone https://github.com/qasimrazamalik-coder/rag-eval-lab.git && cd rag-eval-lab
pip install -e ".[dev]"
make test && make smoke
make data            # if the SQuAD URL is unreachable, see the docstring in scripts/download_data.py
make eval && make report
make answers         # downloads Qwen2.5-0.5B-Instruct and the NLI model; set answers.n_questions in configs/default.yaml
```

Commit `results/retrieval_metrics.json` and `results/answer_metrics.json` together with the updated README table so the numbers and their provenance (seed, package versions, embedder) stay together.

## License and citations

MIT, see `LICENSE`. Models and data keep their own licences: SQuAD 2.0 is CC BY-SA 4.0; check the model cards for all-MiniLM-L6-v2, Qwen2.5-0.5B-Instruct and nli-deberta-v3-small before redistributing weights.

- Rajpurkar, Jia, Liang. Know What You Don't Know: Unanswerable Questions for SQuAD. ACL 2018.
- Reimers, Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.
- Robertson, Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. 2009.
- Cormack, Clarke, Buettcher. Reciprocal Rank Fusion outperforms Condorlet and individual Rank Learning Methods. SIGIR 2009.
- Johnson, Douze, Jegou. Billion-scale similarity search with GPUs (FAISS). IEEE Transactions on Big Data, 2019.
