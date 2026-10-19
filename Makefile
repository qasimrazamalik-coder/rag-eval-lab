PY ?= python

.PHONY: install lint format test smoke data eval answers report app clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -q

# End-to-end run on the synthetic fixture with stub models. Proves the pipeline works; not a result.
smoke:
	$(PY) -m rag_eval.evaluate --config configs/smoke.yaml
	$(PY) -m rag_eval.answer_eval --config configs/smoke.yaml

# Fetch SQuAD v2 dev and turn it into docs.jsonl + eval.jsonl.
data:
	$(PY) scripts/download_data.py
	$(PY) scripts/build_evalset.py

# Retrieval metrics for all chunkers and retrievers (downloads the embedder on first use).
eval:
	$(PY) -m rag_eval.evaluate --config configs/default.yaml

# Answer-level metrics (generator + faithfulness) for one chunker.
answers:
	$(PY) -m rag_eval.answer_eval --config configs/default.yaml

# Write results/retrieval_metrics.json into the README table.
report:
	$(PY) -m rag_eval.report

app:
	$(PY) app/app.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
