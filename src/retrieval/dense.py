"""Dense retrieval with Matryoshka (truncatable) embeddings + FAISS.

One MRL-capable model embeds every chunk once at full size. Shorter vectors
are the first `d` dimensions (re-normalized). Three search modes:

    full    exact search with all dimensions
    short   exact search with the first `short_dim` dimensions
    funnel  shortlist top `shortlist` chunks with the short vectors, then
            re-score only those with the full vectors (coarse-to-fine)

Embeddings are cached on disk per (strategy, model); the model itself is
pluggable through the `Embedder` protocol, so tests can use a fake one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import os

# Both torch and faiss ship an OpenMP runtime on macOS; allow the duplicate (see below).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401  must be imported BEFORE faiss: the reverse order segfaults on macOS (OpenMP clash)
import faiss
import numpy as np

# torch and faiss bundle separate OpenMP runtimes on macOS; multithreaded faiss search
# aborts when both are loaded. Our flat indexes are tiny, so single-threaded is fine.
faiss.omp_set_num_threads(1)

from ..chunking.base import Chunk, index_text

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data/processed/index"


class Embedder(Protocol):
    key: str                       # used in cache file names
    layer_norm: bool               # apply layer norm before truncation (nomic recipe)

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        """kind is 'document' or 'query'. Returns raw (n, D) float32."""


@dataclass
class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Presets below."""
    model_name: str
    key: str
    doc_prefix: str = ""
    query_prefix: str = ""
    layer_norm: bool = False
    max_seq_length: int = 512
    trust_remote_code: bool = False
    batch_size: int = 16

    def __post_init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = SentenceTransformer(
                self.model_name, device=device, trust_remote_code=self.trust_remote_code)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        prefix = self.doc_prefix if kind == "document" else self.query_prefix
        return self._load().encode(
            [prefix + t for t in texts], batch_size=self.batch_size,
            convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=len(texts) > 64,
        ).astype(np.float32)


PRESETS = {
    # Needs trust_remote_code=True (runs code shipped in the model repo).
    "nomic": lambda: SentenceTransformerEmbedder(
        "nomic-ai/nomic-embed-text-v1.5", "nomic768", "search_document: ", "search_query: ",
        layer_norm=True, max_seq_length=1024, trust_remote_code=True),
    # Standard BERT architecture: no remote code. 1024-d, MRL-trained, 512-token limit.
    "mxbai": lambda: SentenceTransformerEmbedder(
        "mixedbread-ai/mxbai-embed-large-v1", "mxbai1024",
        "", "Represent this sentence for searching relevant passages: ", max_seq_length=512),
}


def mrl_vectors(raw: np.ndarray, dim: int, layer_norm: bool) -> np.ndarray:
    """Truncate to the first `dim` dimensions and L2-normalize (inner product = cosine)."""
    x = raw.astype(np.float32)
    if layer_norm:
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-5)
    x = np.ascontiguousarray(x[:, :dim])
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def _fingerprint(texts: list[str], embedder_key: str) -> str:
    h = hashlib.sha256(embedder_key.encode())
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()


class DenseIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, strategy: str,
                 short_dim: int = 128, shortlist: int = 100, cache: bool = True):
        self.embedder, self.n = embedder, len(chunks)
        self.short_dim, self.shortlist = short_dim, shortlist
        texts = [index_text(c) for c in chunks]
        raw = self._load_or_embed(texts, strategy, cache)
        self.full_dim = raw.shape[1]
        self.full = mrl_vectors(raw, self.full_dim, embedder.layer_norm)
        self.short = mrl_vectors(raw, short_dim, embedder.layer_norm)
        self.full_index = faiss.IndexFlatIP(self.full_dim)
        self.full_index.add(self.full)
        self.short_index = faiss.IndexFlatIP(short_dim)
        self.short_index.add(self.short)

    def _load_or_embed(self, texts, strategy, cache) -> np.ndarray:
        path = CACHE_DIR / f"{strategy}__{self.embedder.key}.npy"
        meta = path.with_suffix(".json")
        fp = _fingerprint(texts, self.embedder.key)
        if cache and path.exists() and meta.exists() and json.loads(meta.read_text())["fp"] == fp:
            return np.load(path)
        raw = self.embedder.embed(texts, "document")
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, raw)
            meta.write_text(json.dumps({"fp": fp, "n": len(texts), "dim": int(raw.shape[1])}))
        return raw

    def _query_vecs(self, query: str):
        raw = self.embedder.embed([query], "query")
        return (mrl_vectors(raw, self.full_dim, self.embedder.layer_norm),
                mrl_vectors(raw, self.short_dim, self.embedder.layer_norm))

    @staticmethod
    def _all_scores(index: faiss.Index, q: np.ndarray, n: int) -> np.ndarray:
        d, i = index.search(q, n)
        out = np.full(n, -np.inf, dtype=np.float32)
        out[i[0]] = d[0]
        return out

    def scores(self, query: str, mode: str = "funnel") -> np.ndarray:
        """Score array over all chunks; chunks outside the funnel shortlist get -inf."""
        qf, qs = self._query_vecs(query)
        if mode == "full":
            return self._all_scores(self.full_index, qf, self.n)
        if mode == "short":
            return self._all_scores(self.short_index, qs, self.n)
        if mode == "funnel":
            k = min(self.shortlist, self.n)
            _, cand = self.short_index.search(qs, k)
            cand = cand[0]
            out = np.full(self.n, -np.inf, dtype=np.float32)
            out[cand] = self.full[cand] @ qf[0]   # stage 2: re-score shortlist, full dims
            return out
        raise ValueError(f"unknown dense mode {mode!r}")
