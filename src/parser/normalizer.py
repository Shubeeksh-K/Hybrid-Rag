"""Math-notation normalizer: LaTeX -> canonical symbol tokens.

    \\phi, \\varphi, ϕ, φ          -> PHI
    \\partial_\\mu \\phi             -> PARTIAL[mu](PHI)
    g_{\\mu\\nu}                    -> g[mu,nu]
    g^{\\mu\\nu}                    -> g^[mu,nu]
    m^2                            -> m^[2]

Each symbol yields up to three tokens at different granularities, so partial
matches still score:
    atom      PHI, PARTIAL, m
    generic   PARTIAL[_](PHI), g[_,_]     (index letters relabelled to "_")
    exact     PARTIAL[mu](PHI), g[mu,nu]
This is a pragmatic parser for common physics notation, not a full TeX engine.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --- alphabets ---------------------------------------------------------------
_GREEK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
          "omicron pi rho sigma tau upsilon phi chi psi omega").split()
_GREEK_VARIANTS = {"varepsilon": "epsilon", "vartheta": "theta", "varpi": "pi",
                   "varrho": "rho", "varsigma": "sigma", "varphi": "phi", "varkappa": "kappa"}
_CAPITALS = {"Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"}


def greek_name(cmd: str) -> str | None:
    """'phi' / 'varphi' -> 'PHI';  'Phi' -> 'PHI_CAP';  else None."""
    cmd = _GREEK_VARIANTS.get(cmd, cmd)
    if cmd in _GREEK:
        return cmd.upper()
    if cmd in _CAPITALS:
        return cmd.upper() + "_CAP"
    return None


def _unicode_greek_to_latex(text: str) -> str:
    out = []
    for ch in text:
        if 0x370 <= ord(ch) <= 0x3FF:
            name = unicodedata.name(ch, "")
            words = [w.lower() for w in name.split()]
            letter = next((w for w in words if greek_name(w) or w in _GREEK_VARIANTS or w == "lunate"), None)
            if "lunate" in words and "epsilon" in words:
                letter = "epsilon"
            if letter and letter != "lunate":
                if "CAPITAL" in name:
                    letter = letter.capitalize()
                out.append(f"\\{letter} ")
                continue
        out.append(ch)
    return "".join(out)


# --- command tables ----------------------------------------------------------
IGNORE = set("""left right big Big bigg Bigg bigl bigr Bigl Bigr nonumber notag label displaystyle textstyle
scriptstyle limits nolimits quad qquad hline ldots cdots dots vdots ddots nonumber hspace vspace phantom
Rightarrow rightarrow Leftarrow leftarrow Longrightarrow longrightarrow Leftrightarrow leftrightarrow
to mapsto equiv approx sim simeq propto pm mp times cdot ast star circ le leq ge geq ne neq ll gg
in notin subset subseteq cup cap setminus forall exists neg lor land wedge vee oplus otimes bullet
colon comma tag nolinebreak ne""".split())
IGNORE |= set(",;:!> ") | {"\\", "{", "}", "|", "%", "&", "_", "$", "#"}

BINDERS = {"PARTIAL", "NABLA", "BOX", "DELTA"}      # operators that act on the next symbol
ACCENTS = {"vec": "VEC", "hat": "HAT", "bar": "BAR", "tilde": "TILDE", "dot": "DOT", "ddot": "DDOT",
           "overline": "BAR", "widehat": "HAT", "widetilde": "TILDE", "underline": "UNDER",
           "slashed": "SLASH", "dsl": "SLASH", "not": "NOT"}
FONTS = {"mathcal": "CAL", "cal": "CAL", "mathpzc": "CAL", "mathbb": "BB", "mathbf": "BF",
         "boldsymbol": "BF", "bm": "BF", "mathfrak": "FRAK", "mathscr": "CAL"}
TRANSPARENT_ARG = {"mathrm", "mathit", "mathsf", "mathtt", "operatorname", "frac", "dfrac", "tfrac",
                   "binom", "cancel", "boxed", "fbox", "mbox_keep", "stackrel", "overset", "underset"}
DROP_ARG = {"mbox", "text", "textrm", "textit", "textbf", "hbox", "label", "tag", "hspace", "vspace",
            "mathchar", "scalebox"}
NAMES = {"infty": "INF", "hbar": "HBAR", "partial": "PARTIAL", "nabla": "NABLA", "Box": "BOX",
         "int": "INT", "oint": "OINT", "sum": "SUM", "prod": "PROD", "dagger": "DAGGER",
         "langle": "BRA", "rangle": "KET", "sqrt": "SQRT", "ell": "ELL", "Re": "RE", "Im": "IM",
         "det": "DET", "Tr": "TR", "tr": "TR", "exp": "EXP", "ln": "LN", "log": "LOG", "sin": "SIN",
         "cos": "COS", "tan": "TAN", "sinh": "SINH", "cosh": "COSH", "lim": "LIM", "max": "MAX",
         "min": "MIN", "mathcal": "CAL"}

_LEX = re.compile(r"\\[A-Za-z]+|\\.|[{}^_'\[\]]|[A-Za-z]|\d+(?:\.\d+)?|\S")


# --- AST ---------------------------------------------------------------------
@dataclass
class Atom:
    name: str
    sub: list[str] = field(default_factory=list)
    sup: list[str] = field(default_factory=list)
    arg: "Atom | None" = None

    def render(self, generic: bool = False) -> str:
        def idx(items):
            if not generic:
                return ",".join(items)
            return ",".join("_" if _is_index(i) else i for i in items)
        s = self.name
        if self.sub:
            s += f"[{idx(self.sub)}]"
        if self.sup:
            s += f"^[{idx(self.sup)}]"
        if self.arg is not None:
            s += f"({self.arg.render(generic)})"
        return s


def _is_index(name: str) -> bool:
    """Greek-letter or single lowercase-letter subscripts act as relabelable indices."""
    return name in _GREEK or (len(name) == 1 and name.isalpha() and name.islower())


# --- preprocessing -----------------------------------------------------------
def _strip_args(s: str, cmds: set[str]) -> str:
    """Remove \\cmd{...} (balanced) for the given commands."""
    out, i = [], 0
    pat = re.compile(r"\\(" + "|".join(sorted(cmds, key=len, reverse=True)) + r")\b\s*")
    while i < len(s):
        m = pat.match(s, i)
        if not m:
            out.append(s[i]); i += 1; continue
        j = m.end()
        # skip optional [..] then a {...} group
        if j < len(s) and s[j] == "{":
            depth = 0
            while j < len(s):
                depth += (s[j] == "{") - (s[j] == "}")
                j += 1
                if depth == 0:
                    break
        i = j
        out.append(" ")
    return "".join(out)


# --- parser ------------------------------------------------------------------
class _Parser:
    def __init__(self, toks: list[str]):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def group_tokens(self) -> list[str]:
        """Consume one argument: a {...} group (returned without braces) or a single token."""
        if self.peek() == "{":
            depth, j = 0, self.i
            while j < len(self.t):
                depth += (self.t[j] == "{") - (self.t[j] == "}")
                j += 1
                if depth == 0:
                    break
            inner = self.t[self.i + 1:j - 1]
            self.i = j
            return inner
        if self.peek() is None:
            return []
        tok = self.t[self.i]
        self.i += 1
        return [tok]

    def script_names(self, toks: list[str]) -> list[str]:
        names, sign = [], ""
        for tk in toks:
            if tk in "+-" and len(tk) == 1:
                sign = tk
                continue
            if tk.startswith("\\"):
                cmd = tk[1:]
                n = greek_name(cmd) or NAMES.get(cmd) or (cmd.upper() if cmd not in IGNORE else None)
                if n:
                    names.append(sign + (n.lower() if greek_name(cmd) and not cmd[0].isupper() else n))
            elif tk.isalnum():
                names.append(sign + tk)
            elif tk == "'":
                names.append("prime")
            sign = ""
        return names

    def parse(self, stop_at_close: bool = False) -> list[Atom]:
        atoms: list[Atom] = []
        while (tk := self.peek()) is not None:
            if tk == "}" and stop_at_close:
                break
            self.i += 1
            if tk in ("^", "_"):
                names = self.script_names(self.group_tokens())
                if atoms and names:
                    (atoms[-1].sup if tk == "^" else atoms[-1].sub).extend(names)
            elif tk == "'":
                if atoms:
                    atoms[-1].sup.append("prime")
            elif tk == "{":
                self.i -= 1
                atoms.extend(_Parser(self.group_tokens()).parse())
            elif tk.startswith("\\") and len(tk) > 1 and tk[1].isalpha():
                atoms.extend(self.command(tk[1:]))
            elif len(tk) == 1 and tk.isalpha():
                atoms.append(Atom(tk))
            # digits, operators, brackets, punctuation: ignored
        return atoms

    def command(self, cmd: str) -> list[Atom]:
        if cmd in FONTS or cmd in ACCENTS:
            inner = _Parser(self.group_tokens()).parse()
            if not inner:
                return []
            if cmd in FONTS:
                a = inner[0]
                a.name = f"{a.name}_{FONTS[cmd]}" if len(a.name) == 1 else a.name
            else:
                a = inner[0]
                a.name = f"{ACCENTS[cmd]}({a.name})"
            return inner
        if cmd in TRANSPARENT_ARG:
            n_args = 2 if cmd in {"frac", "dfrac", "tfrac", "binom", "stackrel", "overset", "underset"} else 1
            out: list[Atom] = []
            for _ in range(n_args):
                out.extend(_Parser(self.group_tokens()).parse())
            return out
        if cmd == "sqrt":
            if self.peek() == "[":
                while self.peek() not in ("]", None):
                    self.i += 1
                self.i += 1
            return [Atom("SQRT")] + _Parser(self.group_tokens()).parse()
        if cmd in DROP_ARG:
            self.group_tokens()
            return []
        if cmd in IGNORE:
            return []
        g = greek_name(cmd)
        return [Atom(g or NAMES.get(cmd, cmd.upper()))]


def _bind_operators(atoms: list[Atom]) -> None:
    for i in range(len(atoms) - 2, -1, -1):          # right-to-left: nested operators
        if atoms[i].name in BINDERS and atoms[i].arg is None:
            atoms[i].arg = Atom(atoms[i + 1].name, list(atoms[i + 1].sub), list(atoms[i + 1].sup), atoms[i + 1].arg)


def _flatten(atoms: list[Atom]) -> list[str]:
    toks = []
    for a in atoms:
        toks.append(a.name)
        full, generic = a.render(), a.render(generic=True)
        if full != a.name:
            toks.append(full)
            if generic != full:
                toks.append(generic)
    return toks


# --- public API ---------------------------------------------------------------
def symbols_of_latex(latex: str) -> list[str]:
    """Symbol tokens (with repeats) of one LaTeX math string."""
    s = _unicode_greek_to_latex(latex)
    s = _strip_args(s, DROP_ARG)
    atoms = _Parser(_LEX.findall(s)).parse()
    _bind_operators(atoms)
    return _flatten(atoms)


_MATH_SPANS = re.compile(
    r"\\begin\{(?P<env>[A-Za-z]+\*?)\}(?P<a>.*?)\\end\{(?P=env)\}"
    r"|\\\[(?P<b>.*?)\\\]|\\\((?P<c>.*?)\\\)|\$\$(?P<d>.*?)\$\$|\$(?P<e>[^$]+)\$",
    re.S,
)


def math_segments(text: str) -> list[str]:
    """Math inside delimiters: $..$, \\(..\\), \\[..\\], $$..$$, \\begin{env}..\\end{env}."""
    return [next(g for g in (m.group("a"), m.group("b"), m.group("c"), m.group("d"), m.group("e"))
                 if g is not None)
            for m in _MATH_SPANS.finditer(text)]


def symbols_of_text(text: str) -> list[str]:
    """Symbols of all math in a chunk of mixed prose + LaTeX."""
    toks: list[str] = []
    for seg in math_segments(text):
        toks.extend(symbols_of_latex(seg))
    return toks


def symbols_of_query(query: str) -> list[str]:
    """Symbols of a user query: delimited math if present, else the whole query
    when it looks like raw LaTeX (has \\, ^ or _), after dropping prose words."""
    segs = math_segments(query)
    if segs:
        return [t for s in segs for t in symbols_of_latex(s)]
    if re.search(r"[\\^_]", query):
        bare = re.sub(r"(?<![\\A-Za-z])[A-Za-z]{3,}(?![A-Za-z])", " ", query)
        return symbols_of_latex(bare)
    return []


def fill_block_symbols(doc) -> None:
    """Populate Block.symbols for every equation block (deduplicated, order kept)."""
    for _, b in doc.iter_equations():
        b.symbols = list(dict.fromkeys(symbols_of_latex(b.content)))


if __name__ == "__main__":
    from pathlib import Path
    from .schema import Document
    p = Path(__file__).resolve().parents[2] / "data/processed/date_qft.json"
    d = Document.load(p)
    fill_block_symbols(d)
    d.save(p)
    n = sum(1 for _, b in d.iter_equations() if b.symbols)
    print(f"filled symbols for {n} equation blocks -> {p}")
