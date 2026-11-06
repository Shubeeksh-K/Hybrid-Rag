"""Build chunk files for every implemented strategy and print an inspection table.

    uv run python -m src.chunking.run
"""
from __future__ import annotations

from pathlib import Path

from ..parser.schema import Document
from .base import save_jsonl, summarize
from .equation import chunk_equation
from .fixed import chunk_fixed
from .hierarchical import chunk_hierarchical
from .recursive import chunk_recursive
from .section import chunk_section

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "data/processed/date_qft.json"
OUT = ROOT / "data/processed"

STRATEGIES = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
    "section": chunk_section,
    "equation": chunk_equation,
    "hierarchical": chunk_hierarchical,
}


def main() -> None:
    doc = Document.load(DOC)
    rows, by_type = {}, {}
    for name, fn in STRATEGIES.items():
        chunks = fn(doc)
        save_jsonl(chunks, OUT / f"chunks_{name}.jsonl")
        rows[name] = summarize(chunks, doc)
        if name in ("equation", "hierarchical"):
            for t in sorted({c.chunk_type for c in chunks}):
                tk = sorted(c.n_tokens for c in chunks if c.chunk_type == t)
                by_type[f"{name}/{t}"] = (len(tk), tk[len(tk) // 2], tk[-1])
    cols = list(next(iter(rows.values())))
    print(f"{'strategy':<12}" + "".join(f"{c:>17}" for c in cols))
    for name, r in rows.items():
        print(f"{name:<12}" + "".join(f"{r[c]:>17}" for c in cols))
    print("\nchunk types:      count   median_tokens   max_tokens")
    for k, (n, med, mx) in by_type.items():
        print(f"  {k:<22}{n:>6}{med:>14}{mx:>14}")


if __name__ == "__main__":
    main()
