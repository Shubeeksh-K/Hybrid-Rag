"""Parse a single-file LaTeX source into the shared Document schema.

Scope: tuned for Date's QFT notes (arXiv:2207.02243) but kept generic:
  * strips comments
  * \\section / \\subsection / \\subsubsection -> nested Section tree
  * math environments and \\[ ... \\] -> equation blocks (original LaTeX kept)
  * figures / TikZ / tables / bibliography -> dropped
  * everything else -> text blocks (one per paragraph, inline $...$ kept raw)

Usage:
    uv run python -m src.parser.parser [--src PATH] [--out PATH] [--report]
"""
from __future__ import annotations

import argparse
import re
from bisect import bisect_right
from collections import Counter
from pathlib import Path

from .schema import Block, Document, Section

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "data/raw/date2207/Final.tex"
DEFAULT_OUT = ROOT / "data/processed/date_qft.json"

MATH_ENVS = {
    "equation", "equation*", "eqnarray", "eqnarray*",
    "align", "align*", "gather", "gather*", "multline", "multline*",
}
DROP_ENVS = {"figure", "figure*", "tikzpicture", "thebibliography"}
# Tables here pair a Feynman diagram with its amplitude: keep equations only.
TABLE_ENVS = {"table", "table*", "tabular"}
# Wrappers that only affect layout; blanked out so their contents stay in the flow.
TRANSPARENT_ENVS = {"center", "minipage", "itemize", "enumerate", "abstract"}

LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}

_COMMENT = re.compile(r"(?<!\\)%.*$")
_EVENT = re.compile(
    r"\\(?P<sec>section|subsection|subsubsection)\*?\s*\{"
    r"|\\begin\{(?P<env>[A-Za-z]+\*?)\}"
    r"|(?<!\\)\\\["
)
_LABEL = re.compile(r"\\label\{([^}]*)\}")


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def _blank(match: re.Match) -> str:
    """Replace a match with spaces/newlines of equal length (keeps offsets)."""
    return re.sub(r"[^\n]", " ", match.group(0))


def load_body(path: Path) -> tuple[str, list[int]]:
    """Return (body text with offsets preserved, line-start offsets).

    Comments are removed. Layout-only wrappers are blanked. The body starts at
    the first \\section and ends before the bibliography.
    """
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [_COMMENT.sub("", ln) for ln in raw_lines]
    text = "\n".join(lines)

    start = re.search(r"\\section\*?\s*\{", text)
    end = re.search(r"\\begin\{thebibliography\}", text)
    if not start:
        raise ValueError("no \\section found")
    end_pos = end.start() if end else len(text)
    # blank everything outside [start, end) so offsets (and line numbers) stay valid
    text = (
        re.sub(r"[^\n]", " ", text[: start.start()])
        + text[start.start() : end_pos]
        + re.sub(r"[^\n]", " ", text[end_pos:])
    )

    envs = "|".join(re.escape(e) for e in TRANSPARENT_ENVS)
    for pat in (
        rf"\\(?:begin|end)\{{(?:{envs})\}}(?:\[[^\]]*\])?(?:\{{[^}}]*\}})?",
        r"\\fbox\s*\{",
        r"\\scalebox\{[^}]*\}\s*\{",
        r"\\(?:newpage|noindent|bigskip|medskip|smallskip|centering)\b",
        r"\\[hv]space\*?\{[^}]*\}",
    ):
        text = re.sub(pat, _blank, text)
    # lone closing braces left behind by \fbox{ ... } wrappers
    text = re.sub(r"^[ \t]*\}[ \t]*$", _blank, text, flags=re.M)

    starts = [0]
    for m in re.finditer(r"\n", text):
        starts.append(m.end())
    return text, starts


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def match_brace(text: str, open_pos: int) -> int:
    """Index of the '}' matching the '{' at open_pos."""
    depth = 0
    for i in range(open_pos, len(text)):
        c = text[i]
        if c == "\\":
            continue
        if c == "{" and text[i - 1] != "\\":
            depth += 1
        elif c == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced brace at {open_pos}")


def find_env_end(text: str, name: str, pos: int) -> tuple[int, int]:
    """(start, end) of the \\end{name} closing an env opened before pos."""
    begin = re.compile(rf"\\begin\{{{re.escape(name)}\}}")
    end = re.compile(rf"\\end\{{{re.escape(name)}\}}")
    depth = 1
    while True:
        b, e = begin.search(text, pos), end.search(text, pos)
        if not e:
            raise ValueError(f"missing \\end{{{name}}}")
        if b and b.start() < e.start():
            depth += 1
            pos = b.end()
        else:
            depth -= 1
            pos = e.end()
            if depth == 0:
                return e.start(), e.end()


def clean_text(par: str) -> str:
    par = re.sub(r"\\label\{[^}]*\}", "", par)
    par = re.sub(r"\\cite\{[^}]*\}", "", par)
    par = re.sub(r"\\(?:vspace|hspace)\*?\{[^}]*\}", "", par)
    par = re.sub(r"\\item\b", "\n- ", par)
    par = re.sub(r"[ \t]+", " ", par)
    par = re.sub(r"\s*\n\s*(?!-)", " ", par).strip()
    return par


def clean_title(t: str) -> str:
    t = re.sub(r"\\\\", " ", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------
# Main parser
# --------------------------------------------------------------------------
def parse(path: Path = DEFAULT_SRC, doc_id: str = "date2207",
          source: str = "arXiv:2207.02243") -> Document:
    body, starts = load_body(path)
    line_of = lambda pos: bisect_right(starts, pos)  # 1-based line number

    title_m = re.search(r"\\title\{", path.read_text(encoding="utf-8", errors="replace"))
    title = doc_id
    if title_m:
        raw = path.read_text(encoding="utf-8", errors="replace")
        title = clean_title(raw[title_m.end() : match_brace(raw, title_m.end() - 1)])

    doc = Document(doc_id=doc_id, title=title, source=source)
    counters = [0, 0, 0]
    stack: list[Section] = []
    block_n: dict[str, int] = {}
    eq_n: dict[str, int] = {}

    def current() -> Section:
        return stack[-1]

    def next_block_id(sec: Section) -> str:
        block_n[sec.section_id] = block_n.get(sec.section_id, 0) + 1
        return f"{sec.section_id}:b{block_n[sec.section_id]}"

    def add_text(seg_start: int, seg_end: int) -> None:
        if not stack or seg_end <= seg_start:
            return
        seg = body[seg_start:seg_end]
        pos = 0
        for m in list(re.finditer(r"\n[ \t]*\n", seg)) + [None]:
            end = m.start() if m else len(seg)
            chunk = seg[pos:end]
            if chunk.strip():
                text = clean_text(chunk)
                if re.search(r"[A-Za-z0-9]", text.replace("$", "")):
                    lead = len(chunk) - len(chunk.lstrip())
                    ls = line_of(seg_start + pos + lead)
                    le = line_of(seg_start + pos + len(chunk.rstrip()) - 1)
                    current().blocks.append(
                        Block(next_block_id(current()), "text", text, ls, le)
                    )
            pos = m.end() if m else end

    def add_equation(start: int, end: int, inner: str, env: str) -> None:
        if not stack:
            return
        sec = current()
        eq_n[sec.section_id] = eq_n.get(sec.section_id, 0) + 1
        lab = _LABEL.search(inner)
        sec.blocks.append(
            Block(
                block_id=next_block_id(sec),
                type="equation",
                # drop blank lines left behind by stripped % comments
                content=re.sub(r"\n[ \t]*(?=\n)", "", inner.strip()),
                line_start=line_of(start),
                line_end=line_of(end - 1),
                equation_id=f"{sec.section_id}:eq{eq_n[sec.section_id]}",
                env=env,
                label=lab.group(1) if lab else None,
            )
        )

    pos = 0
    while True:
        m = _EVENT.search(body, pos)
        if not m:
            add_text(pos, len(body))
            break
        add_text(pos, m.start())

        if m.group("sec"):
            level = LEVELS[m.group("sec")]
            close = match_brace(body, m.end() - 1)
            sec_title = clean_title(body[m.end() : close])
            after = body[close + 1 :]
            lab = re.match(r"\s*\\label\{([^}]*)\}", after)
            counters[level - 1] += 1
            for i in range(level, 3):
                counters[i] = 0
            sid = ".".join(str(c) for c in counters[:level])
            while stack and stack[-1].level >= level:
                stack.pop()
            sec = Section(
                section_id=sid, level=level, title=sec_title,
                parent_id=stack[-1].section_id if stack else None,
                label=lab.group(1) if lab else None,
            )
            (stack[-1].children if stack else doc.sections).append(sec)
            stack.append(sec)
            pos = close + 1 + (lab.end() if lab else 0)

        elif m.group("env"):
            name = m.group("env")
            if name in MATH_ENVS or name == "array":
                # top-level `array` = display math written as $\begin{array}..$
                s, e = find_env_end(body, name, m.end())
                add_equation(m.start(), e, body[m.end() : s], name)
                pos = e
            elif name in TABLE_ENVS:
                # diagram | amplitude tables: keep equations, drop diagrams/cell text
                s, e = find_env_end(body, name, m.end())
                sub = m.end()
                while True:
                    sm = _EVENT.search(body, sub, s)
                    if not sm:
                        break
                    sname = sm.group("env")
                    if sname in MATH_ENVS:
                        ss, se = find_env_end(body, sname, sm.end())
                        add_equation(sm.start(), se, body[sm.end() : ss], sname)
                        sub = se
                    elif sname in DROP_ENVS:
                        sub = find_env_end(body, sname, sm.end())[1]
                    elif sname is None and not sm.group("sec"):
                        se = body.index("\\]", sm.end())
                        add_equation(sm.start(), se + 2, body[sm.end() : se], "\\[")
                        sub = se + 2
                    else:
                        sub = sm.end()
                pos = e
            elif name in DROP_ENVS:
                _, e = find_env_end(body, name, m.end())
                pos = e
            else:  # unknown env: leave its contents in the text flow
                pos = m.end()

        else:  # \[ ... \]
            e = body.index("\\]", m.end())
            add_equation(m.start(), e + 2, body[m.end() : e], "\\[")
            pos = e + 2

    return doc


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def report(doc: Document, src: Path) -> str:
    out = []
    secs = list(doc.iter_sections())
    by_level = Counter(s.level for s in secs)
    eqs = list(doc.iter_equations())
    texts = [b for s in secs for b in s.blocks if b.type == "text"]
    out.append(f"Title: {doc.title}")
    out.append(
        f"Sections: {by_level[1]}  Subsections: {by_level[2]}  Subsubsections: {by_level[3]}"
    )
    out.append(f"Text blocks: {len(texts)}  Equation blocks: {len(eqs)}")
    out.append("Equations by env: " + ", ".join(f"{k}={v}" for k, v in Counter(b.env for _, b in eqs).items()))
    out.append(f"Equations with original \\label: {sum(1 for _, b in eqs if b.label)}")

    # cross-check against raw source (non-comment lines)
    raw = "\n".join(_COMMENT.sub("", l) for l in src.read_text(errors="replace").splitlines())
    raw = raw[raw.index("\\section") : raw.index("\\begin{thebibliography}")]
    expected = len(re.findall(r"\\begin\{(?:%s)\}" % "|".join(re.escape(e) for e in MATH_ENVS), raw))
    expected += len(re.findall(r"(?<!\\)\\\[", raw))
    # top-level arrays (not nested inside a math env) also count as equations
    env_alt = "|".join(re.escape(e) for e in MATH_ENVS)
    stripped = re.sub(rf"\\begin\{{({env_alt})\}}.*?\\end\{{\1\}}", "", raw, flags=re.S)
    expected += len(re.findall(r"\\begin\{array\}", stripped))
    out.append(f"Cross-check: math envs + \\[ in raw source = {expected}, parsed = {len(eqs)}"
               f"  -> {'OK' if expected == len(eqs) else 'MISMATCH'}")
    out.append("")
    out.append("Top-level sections (equations / text blocks incl. subsections):")
    for s in doc.sections:
        sub = list(Document("", "", "", [s]).iter_sections())
        ne = sum(1 for x in sub for b in x.blocks if b.type == "equation")
        nt = sum(1 for x in sub for b in x.blocks if b.type == "text")
        out.append(f"  {s.section_id:>2}  eq={ne:<4} text={nt:<4} {s.title[:70]}")
    empty = [s.section_id for s in secs if not s.blocks and not s.children]
    out.append("")
    out.append(f"Sections with no blocks and no children: {empty or 'none'}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    doc = parse(args.src)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.out)
    print(f"Wrote {args.out}")
    if args.report:
        print(report(doc, args.src))


if __name__ == "__main__":
    main()
