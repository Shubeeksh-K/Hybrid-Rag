"""Embed all chunk files once and cache the vectors.

    uv run python -m src.retrieval.build --model mxbai [--strategies fixed section ...]
"""
from __future__ import annotations

import argparse
import time

from ..chunking.base import load_jsonl
from .dense import CACHE_DIR, PRESETS, DenseIndex
from .hybrid import CHUNK_DIR

ALL = ["fixed", "recursive", "section", "equation", "hierarchical"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PRESETS), required=True)
    ap.add_argument("--strategies", nargs="*", default=ALL)
    args = ap.parse_args()
    emb = PRESETS[args.model]()
    for s in args.strategies:
        t = time.time()
        chunks = load_jsonl(CHUNK_DIR / f"chunks_{s}.jsonl")
        DenseIndex(chunks, emb, s)
        print(f"{s:<13} {len(chunks):>5} chunks embedded in {time.time() - t:5.0f}s -> {CACHE_DIR}")


if __name__ == "__main__":
    main()
