"""Baseline 2: recursive splitting (the usual RAG-library default).

Split on the coarsest separator first ("\\n\\n", then "\\n", then " "), recursing
into any piece still over the size limit, then greedily merge neighbouring
pieces up to `size` tokens with `overlap` tokens carried into the next chunk.
It respects paragraphs and sentences-ish, but knows nothing about equations.
"""
from __future__ import annotations

from ..parser.schema import Document
from .base import (CHUNK_TOKENS, OVERLAP_TOKENS, Chunk, count_tokens, encoder,
                   flatten, make_chunk)

SEPARATORS = ["\n\n", "\n", " "]


def _hard_split(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Last resort for a single huge word: cut by tokens."""
    enc = encoder()
    toks = enc.encode(text[start:end], disallowed_special=())
    sub, offs = enc.decode_with_offsets(toks)
    spans = []
    for i in range(0, len(toks), size):
        j = min(i + size, len(toks))
        spans.append((start + offs[i], start + (offs[j] if j < len(toks) else len(sub))))
    return spans


def _split(text: str, start: int, end: int, seps: list[str], size: int) -> list[tuple[int, int]]:
    """Spans (<= size tokens each) that tile text[start:end]."""
    if count_tokens(text[start:end]) <= size:
        return [(start, end)]
    if not seps:
        return _hard_split(text, start, end, size)
    sep, rest = seps[0], seps[1:]
    spans, pos = [], start
    while pos < end:
        k = text.find(sep, pos, end)
        stop = end if k == -1 else k + len(sep)  # separator stays with the left piece
        spans.append((pos, stop))
        pos = stop
    if len(spans) == 1:
        return _split(text, start, end, rest, size)
    out = []
    for a, b in spans:
        out.extend([(a, b)] if count_tokens(text[a:b]) <= size else _split(text, a, b, rest, size))
    return out


def recursive_spans(text: str, start: int, end: int, size: int = CHUNK_TOKENS,
                    overlap: int = OVERLAP_TOKENS) -> list[tuple[int, int]]:
    """Merged chunk spans for text[start:end] (reusable on any sub-range)."""
    pieces = _split(text, start, end, SEPARATORS, size)
    sizes = [count_tokens(text[a:b]) for a, b in pieces]

    spans: list[tuple[int, int]] = []
    cur: list[int] = []   # indices into pieces
    total, fresh = 0, 0   # fresh = pieces not yet emitted in any chunk

    for i, t in enumerate(sizes):
        if total + t > size and cur:
            spans.append((pieces[cur[0]][0], pieces[cur[-1]][1]))
            fresh = 0
            while cur and (total > overlap or total + t > size):
                total -= sizes[cur.pop(0)]
        cur.append(i)
        total += t
        fresh += 1
    if cur and fresh:
        spans.append((pieces[cur[0]][0], pieces[cur[-1]][1]))
    return spans


def chunk_recursive(doc: Document, size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[Chunk]:
    flat = flatten(doc)
    chunks: list[Chunk] = []
    for a, b in recursive_spans(flat.text, 0, len(flat.text), size, overlap):
        c = make_chunk(flat, a, b, "recursive", len(chunks))
        if c:
            chunks.append(c)
    return chunks
