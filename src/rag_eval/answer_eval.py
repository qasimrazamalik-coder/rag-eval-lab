"""Answer-level evaluation for one chunker: token F1, faithfulness, abstention rate, tokens.

Runs the full pipeline (retrieve, build context, generate) on the first N questions. Token F1 is
the SQuAD-style bag-of-words F1 against the reference answer, a blunt measure for a generator
that writes full sentences; read it together with faithfulness, not alone.
"""

from __future__ import annotations

import argparse
import logging
import re
import string
from collections import Counter
from typing import Any

import numpy as np
from tqdm import tqdm

from .evaluate import load_data
from .factory import build_chunker, build_embedder, build_generator, build_scorer, build_tokenizer
from .faithfulness import faithfulness
from .pipeline import RagPipeline
from .prompts import ABSTAIN, PROMPT_VERSION
from .utils import load_config, logger, package_versions, seed_everything, write_json


def normalize(text: str) -> list[str]:
    """SQuAD answer normalisation: lowercase, strip punctuation and articles."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def token_f1(prediction: str, reference: str) -> float:
    p, r = normalize(prediction), normalize(reference)
    overlap = sum((Counter(p) & Counter(r)).values())
    if not p or not r or overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    seed_everything(cfg["seed"])
    a = cfg["answers"]
    docs, items, synthetic = load_data(cfg)
    spec = next(s for s in cfg["chunkers"] if s.get("name", s["type"]) == a["chunker"])
    pipe = RagPipeline(
        docs,
        build_chunker(spec),
        build_embedder(cfg["embedder"]),
        build_generator(cfg["generator"]),
        build_tokenizer(cfg["generator"]),
        mode=a["mode"],
        top_k=a["top_k"],
        context_budget=a["context_budget"],
    )
    scorer = build_scorer(cfg["faithfulness"])
    f1s, faith, ctx_tokens, prompt_tokens, completion_tokens = [], [], [], [], []
    abstained = 0
    for item in tqdm(items[: a["n_questions"]], desc="questions"):
        ans = pipe.ask(item.question)
        f1s.append(token_f1(ans.text, item.answer))
        result = faithfulness(ans.text, list(ans.context.passages), scorer)
        abstained += result.abstained
        if result.score is not None:
            faith.append(result.score)
        ctx_tokens.append(ans.context.tokens)
        prompt_tokens.append(ans.prompt_tokens)
        completion_tokens.append(ans.completion_tokens)
    n = len(f1s)
    return {
        "synthetic": synthetic,
        "seed": cfg["seed"],
        "versions": package_versions(),
        "prompt_version": PROMPT_VERSION,
        "generator": pipe.generator.name,
        "faithfulness_scorer": scorer.name,
        "faithfulness_threshold": scorer.threshold,
        "chunker": a["chunker"],
        "retriever": a["mode"],
        "n_questions": n,
        "token_f1": float(np.mean(f1s)),
        "faithfulness": float(np.mean(faith)) if faith else None,
        "n_faithfulness_scored": len(faith),
        "abstention_rate": abstained / n,
        "mean_context_tokens": float(np.mean(ctx_tokens)),
        "mean_prompt_tokens": float(np.mean(prompt_tokens)),
        "mean_completion_tokens": float(np.mean(completion_tokens)),
        "abstain_string": ABSTAIN,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    path = write_json(cfg["output"]["answers"], run(cfg))
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
