"""Improved method 2: hierarchical chunking (section > derivation > leaf).

Three linked levels over the same text, joined by `parent_id`:

    section chunk      one per section node (its own blocks), as the section baseline
      derivation chunk   a connected run of prose + equations inside that section
        equation / prose chunk   the equation-aware leaves, computed *inside* the derivation

Derivation heuristic (the source has no derivation markers):
  1. equations whose gap is <= GAP_MAX paragraphs form a cluster; the derivation
     is the cluster plus the paragraph before (introduction) and after (follow-up);
  2. the stretches of prose between clusters become their own derivation chunks;
  3. anything over DERIV_CAP tokens is split at block boundaries;
  4. fragments under DERIV_MIN tokens are merged into a neighbour.
Derivations tile the section node's blocks, so every leaf has exactly one parent.
"""
from __future__ import annotations

import itertools

from ..parser.schema import Document
from .base import Chunk, Flat, Seg, count_tokens, flatten, make_chunk, segs_by_section
from .equation import window_chunks

GAP_MAX = 2
DERIV_CAP = 1000
DERIV_MIN = 64


def split_derivations(flat: Flat, segs: list[Seg]) -> list[list[Seg]]:
    """Partition one section node's segments into derivation-sized runs."""
    n = len(segs)
    if n == 0:
        return []
    toks = [count_tokens(flat.text[s.start:s.end]) for s in segs]
    eq = [i for i, s in enumerate(segs) if s.block.type == "equation"]

    clusters: list[list[int]] = []
    for i in eq:
        if clusters and i - clusters[-1][1] - 1 <= GAP_MAX:
            clusters[-1][1] = i
        else:
            clusters.append([i, i])

    # 1-2. tile into [a, b] inclusive index ranges
    tiles: list[list[int]] = []
    cursor = 0
    for first, last in clusters:
        a = first - 1 if first - 1 >= cursor and segs[first - 1].block.type == "text" else first
        b = last + 1 if last + 1 < n and segs[last + 1].block.type == "text" else last
        if cursor < a:
            tiles.append([cursor, a - 1])
        tiles.append([a, b])
        cursor = b + 1
    if cursor < n:
        tiles.append([cursor, n - 1])

    # 3. split oversized tiles at block boundaries
    capped: list[list[int]] = []
    for a, b in tiles:
        start, total = a, 0
        for i in range(a, b + 1):
            if total + toks[i] > DERIV_CAP and i > start:
                cut = i
                # never separate an equation from the paragraph introducing it
                if segs[i].block.type == "equation" and i - 1 > start \
                        and segs[i - 1].block.type == "text":
                    cut = i - 1
                capped.append([start, cut - 1])
                start, total = cut, sum(toks[cut:i])
            total += toks[i]
        capped.append([start, b])

    # 4. merge tiny fragments into the previous tile (or the next if first)
    def size(t):
        return sum(toks[t[0]:t[1] + 1])

    merged: list[list[int]] = []
    for t in capped:
        if merged and (size(t) < DERIV_MIN or size(merged[-1]) < DERIV_MIN) \
                and size(merged[-1]) + size(t) <= DERIV_CAP:
            merged[-1][1] = t[1]
        else:
            merged.append(t)
    return [segs[a:b + 1] for a, b in merged]


def chunk_hierarchical(doc: Document) -> list[Chunk]:
    flat = flatten(doc)
    ids = itertools.count()
    by_sec = segs_by_section(flat)
    chunks: list[Chunk] = []

    for sec in doc.iter_sections():
        segs = by_sec.get(sec.section_id)
        if not segs:
            continue
        section = make_chunk(flat, segs[0].start, segs[-1].end, "hierarchical",
                             next(ids), "section")
        if not section:
            continue
        chunks.append(section)
        for run in split_derivations(flat, segs):
            deriv = make_chunk(flat, run[0].start, run[-1].end, "hierarchical",
                               next(ids), "derivation")
            if not deriv:
                continue
            deriv.parent_id = section.chunk_id
            chunks.append(deriv)
            chunks.extend(window_chunks(flat, run, "hierarchical", ids, deriv.chunk_id))
    return chunks
