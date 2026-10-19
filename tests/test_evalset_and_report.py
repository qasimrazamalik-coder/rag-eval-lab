import json

import pytest

from rag_eval.evalset import load_corpus, load_eval
from rag_eval.report import END, START, render_table, update_readme
from rag_eval.squad import convert
from rag_eval.utils import write_jsonl

SQUAD = {
    "data": [
        {
            "title": "Alpha",
            "paragraphs": [
                {
                    "context": "Cats sleep a lot. They purr.",
                    "qas": [
                        {
                            "id": "a1",
                            "question": "What do cats do?",
                            "answers": [{"text": "purr", "answer_start": 23}],
                        },
                        {
                            "id": "a2",
                            "question": "Unanswerable?",
                            "answers": [],
                            "is_impossible": True,
                        },
                    ],
                },
                {
                    "context": "Dogs bark loudly.",
                    "qas": [
                        {
                            "id": "a3",
                            "question": "What do dogs do?",
                            "answers": [{"text": "bark", "answer_start": 5}],
                        }
                    ],
                },
            ],
        }
    ]
}


@pytest.mark.parametrize("granularity", ["article", "context"])
def test_squad_conversion_offsets_point_at_answers(tmp_path, granularity):
    path = tmp_path / "squad.json"
    path.write_text(json.dumps(SQUAD))
    docs, rows = convert(path, granularity, max_questions=None)
    assert [r["id"] for r in rows] == ["a1", "a3"]  # impossible question dropped
    by_id = {d["doc_id"]: d["text"] for d in docs}
    for r in rows:
        assert by_id[r["gold_doc"]][r["gold_start"] : r["gold_end"]] == r["answer"]
    assert len(docs) == (1 if granularity == "article" else 2)


def test_squad_sampling_is_seeded(tmp_path):
    path = tmp_path / "squad.json"
    path.write_text(json.dumps(SQUAD))
    a = convert(path, "context", max_questions=1, seed=4)[1]
    assert a == convert(path, "context", max_questions=1, seed=4)[1]


def test_eval_loading_both_gold_formats(tmp_path):
    write_jsonl(tmp_path / "docs.jsonl", [{"doc_id": "d", "text": "Alpha beta.\n\nGamma  delta."}])
    docs = load_corpus(tmp_path / "docs.jsonl")
    write_jsonl(
        tmp_path / "eval.jsonl",
        [
            {"question": "q1", "gold_doc": "d", "gold_start": 0, "gold_end": 5},
            {"question": "q2", "gold_passage_text": "Gamma delta."},
        ],
    )
    items = load_eval(tmp_path / "eval.jsonl", docs)
    assert (items[0].start, items[0].end) == (0, 5)
    assert docs[0].text[items[1].start : items[1].end] == "Gamma  delta."


def test_eval_loading_rejects_bad_rows(tmp_path):
    write_jsonl(tmp_path / "docs.jsonl", [{"doc_id": "d", "text": "same same"}])
    docs = load_corpus(tmp_path / "docs.jsonl")
    for row, msg in [
        ({"question": "q", "gold_passage_text": "absent"}, "not found"),
        ({"question": "q", "gold_passage_text": "same"}, "ambiguous"),
        ({"question": "q", "gold_doc": "d", "gold_start": 0, "gold_end": 99}, "outside"),
        ({"question": "q"}, "need gold"),
        ({"question": " ", "gold_doc": "d", "gold_start": 0, "gold_end": 1}, "missing question"),
    ]:
        write_jsonl(tmp_path / "e.jsonl", [row])
        with pytest.raises(ValueError, match=msg):
            load_eval(tmp_path / "e.jsonl", docs)


def _payload(synthetic=False):
    metric = {"mean": 0.5, "ci95": [0.4, 0.6]}
    keys = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    return {
        "synthetic": synthetic,
        "embedder": "stub",
        "n_questions": 10,
        "n_documents": 2,
        "tau": 0.5,
        "seed": 0,
        "results": {
            "fixed": {
                "n_chunks": 7,
                "mean_chunk_tokens": 41.25,
                "retrievers": {"dense": {k: metric for k in keys}},
            }
        },
    }


def test_render_table_and_readme_update(tmp_path):
    table = render_table(_payload())
    assert "| fixed | dense | 7 | 41.2 | 0.500 |" in table or "41.3" in table
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nnot yet run\n{END}\noutro\n")
    update_readme(readme, table)
    text = readme.read_text()
    assert "not yet run" not in text and text.startswith("intro") and text.endswith("outro\n")
    update_readme(readme, table)  # idempotent
    assert readme.read_text() == text


def test_report_refuses_synthetic_and_missing_markers(tmp_path):
    with pytest.raises(ValueError, match="synthetic"):
        render_table(_payload(synthetic=True))
    readme = tmp_path / "README.md"
    readme.write_text("no markers")
    with pytest.raises(ValueError, match="must contain"):
        update_readme(readme, "t")
