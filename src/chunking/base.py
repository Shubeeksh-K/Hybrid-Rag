"""Shared pieces for all chunkers: Chunk record, flattening, tokens, I/O, stats.

Every chunker takes a `Document` and returns `list[Chunk]`. Chunks keep
`text` as the raw body; the section heading path is stored separately in
`header_path` and prepended only when indexing (`index_text`), identically for
every strategy, so no method gets extra context.
"""
from __future__ import annotations

import json
import statistics
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import tiktoken

from ..parser.schema import Block, Document, Section

CHUNK_TOKENS = 256
OVERLAP_TOKENS = 32
ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(s: str) -> int:
    return len(encoder().encode(s, disallowed_special=()))


# --------------------------------------------------------------------------
# Chunk record
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    strategy: str                 # fixed | recursive | section | ...
    text: str                     # raw body, original LaTeX preserved
    header_path: str              # "6 Relativistic Actions > 6.1 Variational ..."
    section_id: str               # section of the chunk's first block
    block_ids: list[str]          # every block the chunk touches (even partly)
    line_start: int               # source span in Final.tex
    line_end: int
    n_tokens: int
    equation_ids: list[str] = field(default_factory=list)      # fully inside the chunk
    cut_equation_ids: list[str] = field(default_factory=list)  # only partly inside
    chunk_type: str = "window"    # window | section | (later: derivation, equation)
    parent_id: str | None = None  # used by hierarchical chunking later


def index_text(c: Chunk) -> str:
    """Text actually fed to the indexes: heading path + body."""
    return f"{c.header_path}\n{c.text}"


def save_jsonl(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l]


# --------------------------------------------------------------------------
# Flattening: Document -> one string + provenance of each block in it
# --------------------------------------------------------------------------
def block_text(b: Block) -> str:
    """Text of a block as a plain-text pipeline would see it (raw LaTeX kept)."""
    if b.type == "text":
        return b.content
    if b.env == "\\[":
        return f"\\[ {b.content} \\]"
    return f"\\begin{{{b.env}}}\n{b.content}\n\\end{{{b.env}}}"


@dataclass
class Seg:
    start: int
    end: int
    section: Section
    block: Block


@dataclass
class Flat:
    text: str
    segs: list[Seg]
    sections: dict[str, Section]

    def __post_init__(self):
        self._starts = [s.start for s in self.segs]

    def header_path(self, section_id: str) -> str:
        parts, sid = [], section_id
        while sid:
            sec = self.sections[sid]
            parts.append(f"{sec.section_id} {sec.title}")
            sid = sec.parent_id
        return " > ".join(reversed(parts))

    def segs_overlapping(self, start: int, end: int) -> list[Seg]:
        i = max(bisect_right(self._starts, start) - 1, 0)
        out = []
        while i < len(self.segs) and self.segs[i].start < end:
            if self.segs[i].end > start:
                out.append(self.segs[i])
            i += 1
        return out


def flatten(doc: Document) -> Flat:
    parts, segs, pos = [], [], 0
    sections = {s.section_id: s for s in doc.iter_sections()}
    for sec in doc.iter_sections():
        for b in sec.blocks:
            t = block_text(b)
            if parts:
                pos += 2  # "\n\n" separator
            segs.append(Seg(pos, pos + len(t), sec, b))
            parts.append(t)
            pos += len(t)
    return Flat("\n\n".join(parts), segs, sections)


def segs_by_section(flat: Flat) -> dict[str, list[Seg]]:
    out: dict[str, list[Seg]] = {}
    for seg in flat.segs:
        out.setdefault(seg.section.section_id, []).append(seg)
    return out


def make_chunk(flat: Flat, start: int, end: int, strategy: str, n: int,
               chunk_type: str = "window") -> Chunk | None:
    """Build a Chunk from the character span [start, end) of the flat text."""
    raw = flat.text[start:end]
    start += len(raw) - len(raw.lstrip())
    end -= len(raw) - len(raw.rstrip())
    if end <= start:
        return None
    covered = flat.segs_overlapping(start, end)
    if not covered:
        return None
    full, cut = [], []
    for s in covered:
        if s.block.type != "equation":
            continue
        (full if start <= s.start and s.end <= end else cut).append(s.block.equation_id)
    text = flat.text[start:end]
    return Chunk(
        chunk_id=f"{strategy}:{n:04d}",
        strategy=strategy,
        text=text,
        header_path=flat.header_path(covered[0].section.section_id),
        section_id=covered[0].section.section_id,
        block_ids=[s.block.block_id for s in covered],
        line_start=min(s.block.line_start for s in covered),
        line_end=max(s.block.line_end for s in covered),
        n_tokens=count_tokens(text),
        equation_ids=full,
        cut_equation_ids=cut,
        chunk_type=chunk_type,
    )


# --------------------------------------------------------------------------
# Stats for manual inspection
# --------------------------------------------------------------------------
def summarize(chunks: list[Chunk], doc: Document) -> dict:
    toks = sorted(c.n_tokens for c in chunks)
    all_eq = {b.equation_id for _, b in doc.iter_equations()}
    intact = {e for c in chunks for e in c.equation_ids}
    cut = {e for c in chunks for e in c.cut_equation_ids}
    return {
        "chunks": len(chunks),
        "tokens_mean": round(statistics.mean(toks)),
        "tokens_median": int(statistics.median(toks)),
        "tokens_p95": toks[int(0.95 * (len(toks) - 1))],
        "tokens_max": toks[-1],
        "equations": len(all_eq),
        "eq_cut_somewhere": len(cut),
        "eq_never_intact": len(all_eq - intact),
    }
