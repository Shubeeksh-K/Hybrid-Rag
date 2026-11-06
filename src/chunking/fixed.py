"""Baseline 1: fixed-size token windows over the whole flattened document.

Structure-blind on purpose: windows ignore paragraph, section and equation
boundaries (this is what a plain RAG tutorial does).
"""
from __future__ import annotations

from ..parser.schema import Document
from .base import CHUNK_TOKENS, OVERLAP_TOKENS, Chunk, encoder, flatten, make_chunk


def chunk_fixed(doc: Document, size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[Chunk]:
    assert 0 <= overlap < size
    flat = flatten(doc)
    enc = encoder()
    toks = enc.encode(flat.text, disallowed_special=())
    text, offs = enc.decode_with_offsets(toks)
    assert text == flat.text, "tokenizer round-trip changed the text"

    chunks, step, i = [], size - overlap, 0
    while i < len(toks):
        j = min(i + size, len(toks))
        end = offs[j] if j < len(toks) else len(text)
        c = make_chunk(flat, offs[i], end, "fixed", len(chunks))
        if c:
            chunks.append(c)
        if j == len(toks):
            break
        i += step
    return chunks
