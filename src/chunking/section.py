"""Baseline 3: one chunk per section node.

Each section's chunk holds only its own direct blocks (text before its first
subsection); subsections get their own chunks, so nothing is duplicated.
Chunk sizes follow the source, so some are far longer than the fixed/recursive size.
"""
from __future__ import annotations

from ..parser.schema import Document
from .base import Chunk, flatten, make_chunk


def chunk_section(doc: Document) -> list[Chunk]:
    flat = flatten(doc)
    by_section: dict[str, list] = {}
    for seg in flat.segs:
        by_section.setdefault(seg.section.section_id, []).append(seg)

    chunks = []
    for sec in doc.iter_sections():  # document order
        segs = by_section.get(sec.section_id)
        if not segs:
            continue
        c = make_chunk(flat, segs[0].start, segs[-1].end, "section", len(chunks), "section")
        if c:
            chunks.append(c)
    return chunks
