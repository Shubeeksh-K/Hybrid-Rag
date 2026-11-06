"""Build chunk files for every implemented strategy and print an inspection table.

    uv run python -m src.chunking.run
"""
from __future__ import annotations

from pathlib import Path

from ..parser.schema import Document
from .base import save_jsonl, summarize
from .fixed import chunk_fixed
from .recursive import chunk_recursive
from .section import chunk_section

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "data/processed/date_qft.json"
OUT = ROOT / "data/processed"

STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "section": chunk_section,
}


def main() -> None:
    doc = Document.load(DOC)
    rows = {}
    for name, fn in STRATEGIES.items():
        chunks = fn(doc)
        save_jsonl(chunks, OUT / f"chunks_{name}.jsonl")
        rows[name] = summarize(chunks, doc)
    cols = list(next(iter(rows.values())))
    print(f"{'strategy':<10}" + "".join(f"{c:>17}" for c in cols))
    for name, r in rows.items():
        print(f"{name:<10}" + "".join(f"{r[c]:>17}" for c in cols))


if __name__ == "__main__":
    main()
