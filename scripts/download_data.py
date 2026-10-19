"""Download the SQuAD v2.0 dev set (the source of the evaluation corpus).

Dataset: SQuAD 2.0 dev, Rajpurkar et al., "Know What You Don't Know: Unanswerable Questions for
SQuAD" (ACL 2018). Licence: CC BY-SA 4.0. Size: roughly 4 MB, 35 articles in the dev split.

URL: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
I believe this is the canonical location but could not verify it from the build sandbox, which
blocks everything except PyPI. If the download fails, open https://rajpurkar.github.io/SQuAD-explorer/
in a browser, find the "Dev Set v2.0" link, and save the file to data/squad/dev-v2.0.json.
The Hugging Face `squad_v2` dataset is an alternative source of the same questions.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="data/squad/dev-v2.0.json")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"{out} already exists; nothing to do.")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            out.write_bytes(resp.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Download failed: {exc}\nSee the docstring of this script for the manual route.")
        return 1
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
