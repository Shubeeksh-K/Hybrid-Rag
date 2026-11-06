"""Symbol-matching signal.

Each chunk is reduced to its normalized notation tokens (see parser/normalizer.py),
and the query's tokens are scored against them with BM25. That gives IDF
weighting (rare structures like PARTIAL[mu](PHI) count far more than a bare `m`)
and length normalization (long chunks do not win just by containing more symbols).
Queries with no math produce an all-zero signal.
"""
from __future__ import annotations

import numpy as np

from ..chunking.base import Chunk
from ..parser.normalizer import symbols_of_query, symbols_of_text
from .bm25 import BM25


class SymbolIndex:
    def __init__(self, chunks: list[Chunk]):
        self.docs = [symbols_of_text(c.text) for c in chunks]
        self.bm25 = BM25(self.docs)
        self.n = len(chunks)

    def scores(self, query: str) -> np.ndarray:
        q = symbols_of_query(query)
        return self.bm25.scores(q) if q else np.zeros(self.n)
