"""Parser sanity tests. Requires the source: `uv run python scripts/fetch_data.py`."""
import re

import pytest

from src.parser.parser import DEFAULT_SRC, parse, report
from src.parser.schema import Document

pytestmark = pytest.mark.skipif(not DEFAULT_SRC.exists(), reason="source not fetched")


@pytest.fixture(scope="module")
def doc() -> Document:
    return parse()


def test_section_hierarchy(doc):
    secs = list(doc.iter_sections())
    assert len(doc.sections) == 23
    assert sum(s.level == 2 for s in secs) == 60
    assert sum(s.level == 3 for s in secs) == 9
    for s in secs:
        for c in s.children:
            assert c.parent_id == s.section_id and c.level == s.level + 1


def test_equation_count_matches_source(doc):
    assert "OK" in report(doc, DEFAULT_SRC)  # cross-check vs raw source


def test_ids_unique_and_ordered(doc):
    ids = [b.block_id for s in doc.iter_sections() for b in s.blocks]
    assert len(ids) == len(set(ids))
    for s in doc.iter_sections():
        lines = [b.line_start for b in s.blocks]
        assert lines == sorted(lines)


def test_no_figure_or_tikz_leaks(doc):
    for s in doc.iter_sections():
        for b in s.blocks:
            if b.type == "text":
                assert not re.search(r"tikz|feynman|\\begin\{", b.content)


def test_klein_gordon_action_derivation(doc):
    sec = next(s for s in doc.iter_sections() if s.section_id == "6")
    eqs = [b for b in sec.blocks if b.type == "equation"]
    action = [b for b in eqs if r"\frac{1}{2}m^2\phi^2" in b.content]
    assert action and r"\Box\phi" in action[0].content  # S[phi] and its variation
    assert action[0].env == "eqnarray" and action[0].equation_id.startswith("6:eq")


def test_json_roundtrip(doc, tmp_path):
    p = tmp_path / "d.json"
    doc.save(p)
    assert Document.load(p) == doc
