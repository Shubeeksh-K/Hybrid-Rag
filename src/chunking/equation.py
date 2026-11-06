"""Improved method 1: equation-aware chunking.

Within a window of blocks (a section's own blocks, or one derivation for the
hierarchical method):

  * every display equation is atomic: never split, whatever its length;
  * equations separated by only a little prose are merged into one group
    (a derivation writes several equations in a row);
  * each group becomes a chunk = the last sentences of the paragraph before +
    the group + the first sentences of the paragraph after (these usually
    introduce the equation and define its symbols);
  * prose not belonging to any equation group is chunked recursively so that
    conceptual explanations stay retrievable.

Every chunk is one contiguous span of the flattened text.
"""
from __future__ import annotations

import itertools
import re
from typing import Iterator

from ..parser.schema import Document
from .base import (Chunk, Flat, Seg, count_tokens, flatten, make_chunk,
                   segs_by_section)
from .recursive import recursive_spans

MERGE_GAP_TOKENS = 48     # prose between two equations still allowed inside one group
GROUP_CAP_TOKENS = 512    # stop merging equations once the group reaches this size
CONTEXT_TOKENS = 64       # budget for the sentences taken on each side
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=\S)")


def _sentence_starts(text: str) -> list[int]:
    return [0] + [m.end() for m in _SENT_END.finditer(text)]


def _context_before(flat: Flat, seg: Seg) -> int:
    """Start offset (in flat text) of the trailing sentences of a text block."""
    text = flat.text[seg.start:seg.end]
    starts = _sentence_starts(text)
    pick = starts[-1]
    if len(starts) > 1 and count_tokens(text[starts[-2]:]) <= CONTEXT_TOKENS:
        pick = starts[-2]
    return seg.start + pick


def _context_after(flat: Flat, seg: Seg) -> int:
    """End offset (in flat text) of the leading sentences of a text block."""
    text = flat.text[seg.start:seg.end]
    ends = [m.start() for m in _SENT_END.finditer(text)] + [len(text)]
    pick = ends[0]
    if len(ends) > 1 and count_tokens(text[:ends[1]]) <= CONTEXT_TOKENS:
        pick = ends[1]
    return seg.start + pick


def window_chunks(flat: Flat, segs: list[Seg], strategy: str, ids: Iterator[int],
                  parent_id: str | None = None) -> list[Chunk]:
    """Equation chunks + prose chunks for one window of consecutive segments."""
    if not segs:
        return []
    eq = [i for i, s in enumerate(segs) if s.block.type == "equation"]

    # 1. group equations (index ranges [first, last] into segs)
    groups: list[list[int]] = []
    for i in eq:
        if groups:
            first, last = groups[-1]
            between = flat.text[segs[last].end:segs[i].start]
            merged = flat.text[segs[first].start:segs[i].end]
            if (count_tokens(between) <= MERGE_GAP_TOKENS
                    and count_tokens(merged) <= GROUP_CAP_TOKENS):
                groups[-1][1] = i
                continue
        groups.append([i, i])

    out: list[Chunk] = []

    def emit(a: int, b: int, ctype: str) -> None:
        c = make_chunk(flat, a, b, strategy, next(ids), ctype)
        if c:
            c.parent_id = parent_id
            out.append(c)

    # 2. equation chunks with surrounding sentences
    for first, last in groups:
        start, end = segs[first].start, segs[last].end
        if first > 0 and segs[first - 1].block.type == "text":
            start = _context_before(flat, segs[first - 1])
        if last + 1 < len(segs) and segs[last + 1].block.type == "text":
            end = _context_after(flat, segs[last + 1])
        emit(start, end, "equation")

    # 3. prose between/around groups -> recursive chunks
    covered = set()
    for first, last in groups:
        covered.update(range(first, last + 1))
    run: list[Seg] = []
    for i, s in enumerate(segs + [None]):
        if s is not None and i not in covered:
            run.append(s)
            continue
        if run:
            for a, b in recursive_spans(flat.text, run[0].start, run[-1].end):
                emit(a, b, "prose")
            run = []
    return out


def chunk_equation(doc: Document) -> list[Chunk]:
    flat = flatten(doc)
    ids = itertools.count()
    chunks: list[Chunk] = []
    by_sec = segs_by_section(flat)
    for sec in doc.iter_sections():
        chunks.extend(window_chunks(flat, by_sec.get(sec.section_id, []), "equation", ids))
    return chunks
