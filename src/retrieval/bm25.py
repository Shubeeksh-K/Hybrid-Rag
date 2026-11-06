"""BM25 (own implementation) + a LaTeX-aware lexical tokenizer.

One small BM25 class serves two signals: lexical search over chunk text, and
symbol search over normalized notation tokens (see symbol.py).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from ..chunking.base import Chunk, index_text

# `\partial_\mu\phi` -> partial, mu, phi ; `m^2` -> m, 2 ; `Klein-Gordon` -> klein, gordon
_TOKEN = re.compile(r"\\?[A-Za-z]+|\d+")
STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or is are was were be been it its this that "
    "these those as we our us can will so then thus has have which where when what how i".split()
)


def lexical_tokens(text: str) -> list[str]:
    out = []
    for m in _TOKEN.finditer(text):
        t = m.group().lstrip("\\").lower()
        if t not in STOPWORDS:
            out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.doclen = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = self.doclen.mean() if self.n else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in postings.items()}
        self.post = {t: (np.array([i for i, _ in p]), np.array([f for _, f in p], dtype=float))
                     for t, p in postings.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n)
        norm = self.k1 * (1 - self.b + self.b * self.doclen / (self.avgdl or 1.0))
        for term in query_tokens:  # repeated query terms count repeatedly
            if term not in self.post:
                continue
            idx, tf = self.post[term]
            s[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[idx])
        return s


class LexicalIndex:
    """BM25 over `header_path + text` of each chunk."""

    def __init__(self, chunks: list[Chunk]):
        self.bm25 = BM25([lexical_tokens(index_text(c)) for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        return self.bm25.scores(lexical_tokens(query))
