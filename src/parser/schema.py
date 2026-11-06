"""Shared intermediate representation for a parsed document.

Every chunker reads this structure; none parse raw LaTeX themselves.

    Document -> Section (nested) -> Block ("text" | "equation")

Original LaTeX is preserved in `Block.content` for equations. Figures and
TikZ diagrams are dropped by the parser and do not appear here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

BlockType = Literal["text", "equation"]


@dataclass
class Block:
    block_id: str                 # e.g. "3.2:b17" (stable, position-based)
    type: BlockType
    content: str                  # text: prose; equation: original LaTeX
    line_start: int               # 1-based line range in the source file
    line_end: int
    # Equation-only fields (None / empty for text blocks)
    equation_id: str | None = None    # e.g. "3.2:eq4"
    env: str | None = None            # "equation", "eqnarray", "\\[", ...
    label: str | None = None          # original \label{...}, if any
    symbols: list[str] = field(default_factory=list)  # filled by normalizer later


@dataclass
class Section:
    section_id: str               # "3", "3.2", "3.2.1"
    level: int                    # 1=section, 2=subsection, 3=subsubsection
    title: str
    parent_id: str | None = None
    label: str | None = None
    blocks: list[Block] = field(default_factory=list)      # direct content only
    children: list["Section"] = field(default_factory=list)  # nested subsections


@dataclass
class Document:
    doc_id: str
    title: str
    source: str                   # e.g. "arXiv:2207.02243"
    sections: list[Section] = field(default_factory=list)

    # --- helpers -----------------------------------------------------------
    def iter_sections(self):
        """Depth-first over all sections, in document order."""
        stack = list(reversed(self.sections))
        while stack:
            s = stack.pop()
            yield s
            stack.extend(reversed(s.children))

    def iter_equations(self):
        for s in self.iter_sections():
            for b in s.blocks:
                if b.type == "equation":
                    yield s, b

    # --- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        def sec(sd: dict) -> Section:
            return Section(
                section_id=sd["section_id"],
                level=sd["level"],
                title=sd["title"],
                parent_id=sd.get("parent_id"),
                label=sd.get("label"),
                blocks=[Block(**b) for b in sd.get("blocks", [])],
                children=[sec(c) for c in sd.get("children", [])],
            )

        return cls(
            doc_id=d["doc_id"],
            title=d["title"],
            source=d["source"],
            sections=[sec(s) for s in d.get("sections", [])],
        )

    @classmethod
    def load(cls, path: str | Path) -> "Document":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
