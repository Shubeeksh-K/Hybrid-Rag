"""Grounded answering: retrieve -> numbered sources -> LLM answer with [n] citations -> verify.

    uv run python -m src.rag.answer "How is the Klein-Gordon equation derived from an action?"

The model may only use the numbered sources. After generation we check mechanically that
every citation index exists and every sentence of the answer carries a citation (or the
answer is the explicit refusal). Failures are reported, not hidden.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

from ..retrieval.dense import PRESETS, DenseIndex
from ..retrieval.hybrid import Hit, Retriever
from .llm import LLM, OllamaLLM

REFUSAL = "Not found in the notes."
SYSTEM = f"""You answer questions about quantum field theory using ONLY the numbered sources provided.
Rules:
- Every sentence of your answer must end with a citation like [1] or [2][3] naming the source(s) it relies on.
- Use only facts present in the sources. Do not add outside knowledge.
- Keep equations in LaTeX exactly as written in the sources (use $...$ for inline math).
- If the sources do not contain the answer, reply exactly: {REFUSAL}
- Be concise: at most 6 sentences."""


@dataclass
class Source:
    n: int
    chunk_id: str
    header: str
    text: str
    lines: tuple[int, int]
    tokens: int


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source]
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    refused: bool = False

    @property
    def grounded(self) -> bool:
        return self.refused or (bool(self.cited) and not self.invalid_citations and not self.uncited_sentences)


def pack_sources(hits: list[Hit], budget: int) -> list[Source]:
    """Take ranked chunks until the token budget is used (skip a chunk that would overshoot by >50%)."""
    out, used = [], 0
    for h in hits:
        c = h.chunk
        if out and used + c.n_tokens > budget * 1.5:
            continue
        out.append(Source(len(out) + 1, c.chunk_id, c.header_path, c.text, (c.line_start, c.line_end), c.n_tokens))
        used += c.n_tokens
        if used >= budget:
            break
    return out


def build_prompt(question: str, sources: list[Source]) -> str:
    blocks = [f"[{s.n}] (section: {s.header})\n{s.text}" for s in sources]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends outside math; a citation after the period belongs to the sentence."""
    masked = re.sub(r"\$[^$]*\$", lambda m: m.group(0).replace(".", "\u0001"), text)
    parts = re.split(r"(?<=[.!?])\s*((?:\[\d+\])*)\s+(?=[A-Z$\\])", masked)
    sents, i = [], 0
    while i < len(parts):
        s = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        sents.append(s.replace("\u0001", ".").strip())
        i += 2
    return [s for s in sents if s]


def verify(text: str, n_sources: int) -> tuple[list[int], list[int], list[str], bool]:
    refused = text.strip().rstrip(".").lower() == REFUSAL.rstrip(".").lower()
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
    invalid = [c for c in cited if not 1 <= c <= n_sources]
    uncited = [] if refused else [s for s in split_sentences(text) if not re.search(r"\[\d+\]", s)]
    return cited, invalid, uncited, refused


class Answerer:
    def __init__(self, retriever: Retriever, llm: LLM, budget: int = 1500, k: int = 12):
        self.retriever, self.llm, self.budget, self.k = retriever, llm, budget, k

    def ask(self, question: str) -> Answer:
        hits = self.retriever.retrieve(question, k=self.k, mode="hybrid", dense_mode="full",
                                       with_parent=False)
        sources = pack_sources(hits, self.budget)
        text = self.llm.chat(SYSTEM, build_prompt(question, sources))
        cited, invalid, uncited, refused = verify(text, len(sources))
        return Answer(question, text, sources, cited, invalid, uncited, refused)


def render(a: Answer) -> str:
    out = [a.text, ""]
    for s in a.sources:
        mark = "*" if s.n in a.cited else " "
        out.append(f"{mark}[{s.n}] {s.header}  (Final.tex L{s.lines[0]}-{s.lines[1]}, {s.tokens} tok)")
    status = "grounded" if a.grounded else (f"NOT GROUNDED: invalid={a.invalid_citations} "
                                            f"uncited={len(a.uncited_sentences)}")
    out.append(f"\n[{status}; * = cited]")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--strategy", default="recursive")
    ap.add_argument("--model", default="granite4.2:8b")
    ap.add_argument("--budget", type=int, default=1500)
    args = ap.parse_args()
    r = Retriever(args.strategy)
    r.dense = DenseIndex(r.chunks, PRESETS["nomic"](), args.strategy)
    print(render(Answerer(r, OllamaLLM(args.model), args.budget).ask(args.question)))


if __name__ == "__main__":
    main()
