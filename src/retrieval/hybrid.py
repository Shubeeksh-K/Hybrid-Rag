"""Retriever: one entry point over BM25, dense, symbol and their fusion.

    r = Retriever("equation", dense=DenseIndex(...))
    r.retrieve(r"\\Box\\phi + m^2\\phi = 0", k=5, mode="hybrid")

Modes: bm25 | dense | symbol | hybrid | rrf.

hybrid: take the top `top_n` candidates from each signal, min-max normalize each
signal over that candidate set (raw BM25 / cosine / symbol scores are not
comparable), then   score = a*dense + b*bm25 + g*symbol.
rrf:    reciprocal rank fusion over the three rankings (no weights to tune).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..chunking.base import Chunk, load_jsonl
from .bm25 import LexicalIndex
from .dense import DenseIndex
from .symbol import SymbolIndex

ROOT = Path(__file__).resolve().parents[2]
CHUNK_DIR = ROOT / "data/processed"
RRF_K = 60


@dataclass
class Hit:
    rank: int
    score: float
    chunk: Chunk
    parent: Chunk | None = None
    signals: dict[str, float] = field(default_factory=dict)  # normalized per-signal scores


def _top(scores: np.ndarray, n: int) -> np.ndarray:
    finite = np.isfinite(scores)
    idx = np.argsort(-np.where(finite, scores, -np.inf), kind="stable")[:n]
    return idx[finite[idx]]


def _minmax(s: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Min-max normalize over the candidate set; non-candidates and -inf -> 0."""
    out = np.zeros_like(s, dtype=float)
    v = s[cand]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return out
    lo, hi = v.min(), v.max()
    for i in cand:
        if np.isfinite(s[i]):
            out[i] = (s[i] - lo) / (hi - lo) if hi > lo else (1.0 if hi > 0 else 0.0)
    return out


class Retriever:
    def __init__(self, strategy: str, dense: DenseIndex | None = None,
                 chunks: list[Chunk] | None = None):
        self.strategy = strategy
        self.chunks = chunks if chunks is not None else load_jsonl(CHUNK_DIR / f"chunks_{strategy}.jsonl")
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.lexical = LexicalIndex(self.chunks)
        self.symbol = SymbolIndex(self.chunks)
        self.dense = dense

    def signal_scores(self, query: str, dense_mode: str = "funnel") -> dict[str, np.ndarray]:
        out = {"bm25": self.lexical.scores(query), "symbol": self.symbol.scores(query)}
        if self.dense is not None:
            out["dense"] = self.dense.scores(query, dense_mode)
        return out

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid", dense_mode: str = "funnel",
                 weights: tuple[float, float, float] = (1.0, 1.0, 1.0), top_n: int = 50,
                 with_parent: bool = True) -> list[Hit]:
        sig = self.signal_scores(query, dense_mode)
        if mode in sig:
            final, norm = sig[mode], {}
            order = _top(final, k)
        elif mode in ("hybrid", "rrf"):
            names = [n for n in ("dense", "bm25", "symbol") if n in sig]
            if mode == "hybrid":
                a, b, g = weights
                w = {"dense": a, "bm25": b, "symbol": g}
                cand = np.unique(np.concatenate([_top(sig[n], top_n) for n in names]))
                norm = {n: _minmax(sig[n], cand) for n in names}
                final = sum(w[n] * norm[n] for n in names)
                final = np.where(np.isin(np.arange(len(final)), cand), final, -np.inf)
            else:
                final, norm = np.zeros(len(self.chunks)), {}
                for n in names:
                    if not np.any(sig[n] > 0) and n == "symbol":
                        continue  # query has no math: this signal abstains
                    for r, i in enumerate(_top(sig[n], top_n)):
                        final[i] += 1.0 / (RRF_K + r + 1)
                final = np.where(final > 0, final, -np.inf)
            order = _top(final, k)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        hits = []
        for r, i in enumerate(order, 1):
            c = self.chunks[i]
            parent = self.by_id.get(c.parent_id) if (with_parent and c.parent_id) else None
            hits.append(Hit(r, float(final[i]), c, parent,
                            {n: float(v[i]) for n, v in norm.items()}))
        return hits
