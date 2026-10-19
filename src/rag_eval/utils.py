"""Seeding, config loading, environment fingerprinting and JSON output in one place."""

from __future__ import annotations

import json
import logging
import platform
import random
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("rag_eval")

_TRACKED_PACKAGES = (
    "numpy",
    "faiss-cpu",
    "pypdf",
    "sentence-transformers",
    "transformers",
    "torch",
)


def seed_everything(seed: int) -> None:
    """Seed python and numpy; torch is seeded only if it is already imported."""
    random.seed(seed)
    np.random.seed(seed)
    import sys

    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config; fail loudly if the file is missing or not a mapping."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {p} must contain a YAML mapping at the top level")
    return cfg


def package_versions() -> dict[str, str]:
    """Versions of the packages that can change results, plus the python version."""
    out = {"python": platform.python_version()}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON with sorted keys so repeated runs diff cleanly."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, reporting the line number of any malformed row."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write one JSON object per line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p
