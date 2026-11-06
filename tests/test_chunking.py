"""Chunker sanity tests. Requires the source: `uv run python scripts/fetch_data.py`."""
import pytest

from src.chunking.base import CHUNK_TOKENS, count_tokens, flatten
from src.chunking.fixed import chunk_fixed
from src.chunking.recursive import chunk_recursive
from src.chunking.section import chunk_section
from src.parser.parser import DEFAULT_SRC, parse

pytestmark = pytest.mark.skipif(not DEFAULT_SRC.exists(), reason="source not fetched")


@pytest.fixture(scope="module")
def doc():
    return parse()


@pytest.fixture(scope="module")
def flat(doc):
    return flatten(doc)


def covers_everything(chunks, flat):
    """Every character of the flattened text lies in some chunk."""
    covered = set()
    for c in chunks:
        i = flat.text.index(c.text)
        covered.update(range(i, i + len(c.text)))
    return all(flat.text[i].isspace() or i in covered for i in range(len(flat.text)))


def test_fixed_sizes_and_coverage(doc, flat):
    chunks = chunk_fixed(doc)
    assert all(c.n_tokens <= CHUNK_TOKENS + 4 for c in chunks)  # decode boundary slack
    assert sum(c.n_tokens >= CHUNK_TOKENS - 8 for c in chunks) > len(chunks) * 0.95
    assert covers_everything(chunks, flat)


def test_fixed_overlap(doc):
    a, b = chunk_fixed(doc)[:2]
    assert a.text[-60:] in a.text and b.text[:30] in a.text  # b starts inside a


def test_recursive_respects_limit_and_coverage(doc, flat):
    chunks = chunk_recursive(doc)
    assert all(c.n_tokens <= CHUNK_TOKENS for c in chunks)
    assert covers_everything(chunks, flat)


def test_recursive_keeps_small_equations_whole(doc):
    chunks = chunk_recursive(doc)
    intact = {e for c in chunks for e in c.equation_ids}
    from src.chunking.base import block_text
    for _, b in doc.iter_equations():
        if count_tokens(block_text(b)) <= CHUNK_TOKENS:
            assert b.equation_id in intact


def test_section_one_chunk_per_nonempty_section(doc):
    chunks = chunk_section(doc)
    nonempty = [s for s in doc.iter_sections() if s.blocks]
    assert [c.section_id for c in chunks] == [s.section_id for s in nonempty]
    assert all(not c.cut_equation_ids for c in chunks)


def test_fixed_cuts_klein_gordon_action(doc):
    cut = {e for c in chunk_fixed(doc) for e in c.cut_equation_ids}
    assert "6:eq3" in cut  # the baseline's known weakness


def test_header_path_and_spans(doc):
    for c in chunk_section(doc):
        assert c.header_path.split(" > ")[-1].startswith(c.section_id)
        assert c.line_start <= c.line_end


# ---- equation-aware and hierarchical ---------------------------------------
from src.chunking.equation import chunk_equation  # noqa: E402
from src.chunking.hierarchical import chunk_hierarchical  # noqa: E402


@pytest.fixture(scope="module")
def eq_chunks(doc):
    return chunk_equation(doc)


@pytest.fixture(scope="module")
def hier(doc):
    return chunk_hierarchical(doc)


def test_equation_chunks_never_cut_and_cover_all(doc, flat, eq_chunks):
    assert not any(c.cut_equation_ids for c in eq_chunks)
    assert {e for c in eq_chunks for e in c.equation_ids} == \
        {b.equation_id for _, b in doc.iter_equations()}
    assert covers_everything(eq_chunks, flat)  # prose is not lost


def test_equation_chunk_has_context_sentences(eq_chunks):
    c = next(c for c in eq_chunks if "6:eq3" in c.equation_ids)
    assert c.text.startswith("This works as long as") or "Thus, let" in c.text
    assert r"\frac{1}{2}m^2\phi^2" in c.text


def test_hierarchical_parent_links(hier):
    byid = {c.chunk_id: c for c in hier}
    assert len(byid) == len(hier)  # unique ids
    for c in hier:
        if c.chunk_type == "section":
            assert c.parent_id is None
        elif c.chunk_type == "derivation":
            assert byid[c.parent_id].chunk_type == "section"
        else:
            assert c.chunk_type in ("equation", "prose")
            parent = byid[c.parent_id]
            assert parent.chunk_type == "derivation"
            assert c.text in parent.text  # nesting: leaf lies inside its parent
            assert c.section_id == parent.section_id


def test_hierarchical_no_cut_equations_and_derivations_tile_section(hier):
    assert not any(c.cut_equation_ids for c in hier)
    byid = {c.chunk_id: c for c in hier}
    for sec in (c for c in hier if c.chunk_type == "section"):
        kids = [c for c in hier if c.parent_id == sec.chunk_id]
        assert kids and all(k.chunk_type == "derivation" for k in kids)
        starts = [sec.text.index(k.text) for k in kids]
        assert starts == sorted(starts)
        assert kids[0].text[:40] == sec.text[:40] and kids[-1].text[-40:] == sec.text[-40:]


def test_kg_action_derivation_keeps_equations_with_intro(hier):
    byid = {c.chunk_id: c for c in hier}
    leaf = next(c for c in hier if c.chunk_type == "equation" and "6:eq3" in c.equation_ids)
    deriv = byid[leaf.parent_id]
    assert "6:eq3" in deriv.equation_ids and "Thus, let" in deriv.text
