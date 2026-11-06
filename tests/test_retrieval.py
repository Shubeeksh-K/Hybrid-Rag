import zlib
from pathlib import Path

import numpy as np
import pytest

from src.chunking.base import CHUNK_TOKENS  # noqa: F401  (keeps import path check)
from src.parser.normalizer import symbols_of_latex, symbols_of_query, symbols_of_text
from src.retrieval.bm25 import BM25, lexical_tokens
from src.retrieval.dense import DenseIndex, mrl_vectors
from src.retrieval.hybrid import CHUNK_DIR, Retriever

HAVE_CHUNKS = (CHUNK_DIR / "chunks_equation.jsonl").exists()
needs_chunks = pytest.mark.skipif(not HAVE_CHUNKS, reason="run src.chunking.run first")


# ---- normalizer --------------------------------------------------------------
@pytest.mark.parametrize("latex", [r"\phi", "ϕ", "φ", r"\varphi"])
def test_phi_variants_same_symbol(latex):
    assert symbols_of_latex(latex) == ["PHI"]


def test_operator_and_index_notation():
    assert "PARTIAL[mu](PHI)" in symbols_of_latex(r"\partial_\mu \phi")
    assert "PARTIAL[_](PHI)" in symbols_of_latex(r"\partial_\nu \phi")  # relabel-invariant
    assert "g[mu,nu]" in symbols_of_latex(r"g_{\mu\nu}")
    assert "g^[mu,nu]" in symbols_of_latex(r"g^{\mu\nu}")  # up vs down differ
    assert "m^[2]" in symbols_of_latex("m^2")
    assert "BOX(PHI)" in symbols_of_latex(r"\Box\phi")


def test_text_and_query_extraction():
    assert symbols_of_text("the field $\\phi$ obeys prose") == ["PHI"]
    assert symbols_of_query("How is the Klein-Gordon equation derived?") == []
    assert "BOX(PHI)" in symbols_of_query(r"\Box\phi + m^2\phi = 0")
    assert "m" not in symbols_of_query("what is the field?")


# ---- lexical ------------------------------------------------------------------
def test_lexical_tokens_split_latex():
    assert lexical_tokens(r"\partial_\mu\phi, Klein-Gordon") == ["partial", "mu", "phi", "klein", "gordon"]


def test_bm25_ranks_matching_doc_first():
    bm = BM25([["scalar", "field"], ["dirac", "spinor"], ["scalar", "scalar", "mass"]])
    s = bm.scores(["dirac"])
    assert s.argmax() == 1 and s[0] == 0
    assert bm.scores(["zzz"]).sum() == 0


# ---- dense (fake embedder, no model download) ---------------------------------
class FakeEmbedder:
    key, layer_norm = "fake", False

    def embed(self, texts, kind):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for r, t in enumerate(texts):
            for w in lexical_tokens(t):
                out[r, zlib.crc32(w.encode()) % 256] += 1.0
        return out


def test_mrl_truncation_is_normalized_prefix():
    raw = np.random.default_rng(0).normal(size=(4, 32)).astype(np.float32)
    short = mrl_vectors(raw, 8, False)
    assert short.shape == (4, 8) and np.allclose(np.linalg.norm(short, axis=1), 1)
    p = raw[:, :8] / np.linalg.norm(raw[:, :8], axis=1, keepdims=True)
    assert np.allclose(short, p, atol=1e-6)


@needs_chunks
def test_funnel_modes():
    r = Retriever("section")
    d = DenseIndex(r.chunks, FakeEmbedder(), "section", short_dim=64, shortlist=10, cache=False)
    q = "scalar field Klein-Gordon equation action"
    full, short, funnel = (d.scores(q, m) for m in ("full", "short", "funnel"))
    assert np.isfinite(full).all() and np.isfinite(short).all()
    assert np.isfinite(funnel).sum() == 10
    shortlist = set(np.argsort(-short)[:10])
    assert set(np.flatnonzero(np.isfinite(funnel))) == shortlist
    d_all = DenseIndex(r.chunks, FakeEmbedder(), "section", short_dim=64, shortlist=len(r.chunks), cache=False)
    assert d_all.scores(q, "funnel").argmax() == full.argmax()  # full shortlist == full search


# ---- retriever -------------------------------------------------------------------
@pytest.fixture(scope="module")
def retr():
    r = Retriever("equation")
    r.dense = DenseIndex(r.chunks, FakeEmbedder(), "equation", short_dim=64, cache=False)
    return r


@needs_chunks
@pytest.mark.parametrize("mode", ["bm25", "dense", "symbol", "hybrid", "rrf"])
def test_all_modes_return_sorted_hits(retr, mode):
    hits = retr.retrieve(r"\Box\phi + m^2\phi = 0", k=5, mode=mode)
    assert len(hits) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


@needs_chunks
def test_symbol_query_finds_klein_gordon_variation(retr):
    top = retr.retrieve(r"\Box\phi + m^2\phi = 0", k=3, mode="symbol")
    assert any("6:eq3" in h.chunk.equation_ids for h in top)


@needs_chunks
def test_symbol_signal_abstains_without_math(retr):
    assert retr.symbol.scores("How is the Klein-Gordon equation derived?").sum() == 0


@needs_chunks
def test_hierarchical_hits_carry_parent():
    r = Retriever("hierarchical")
    hits = r.retrieve(r"\partial_\mu\phi\partial^\mu\phi", k=5, mode="bm25")
    for h in hits:
        assert (h.parent is None) == (h.chunk.parent_id is None)
        if h.parent:
            assert h.parent.chunk_id == h.chunk.parent_id
