from src.rag.answer import REFUSAL, Answerer, split_sentences, verify
from src.rag.llm import FakeLLM
from src.retrieval.hybrid import Retriever
from src.chunking.base import Chunk


def chunk(i, text):
    return Chunk(chunk_id=f"c{i}", strategy="t", text=text, header_path="1 Intro", section_id="1",
                 block_ids=[f"1:b{i}"], line_start=i, line_end=i, n_tokens=20)


def test_split_keeps_math_periods_and_citations():
    s = split_sentences("We have $x = 0.5$ here [1]. Then it follows [2][3]. Last one [1].")
    assert len(s) == 3 and s[1].endswith("[2][3].") or s[1].endswith("[2][3]")


def test_verify_flags_invalid_and_uncited():
    cited, invalid, uncited, refused = verify("First [1]. Second has no citation. Third [9].", 2)
    assert cited == [1, 9] and invalid == [9] and len(uncited) == 1 and not refused


def test_verify_refusal_is_grounded():
    assert verify(REFUSAL, 3)[3] is True


def test_answerer_end_to_end_with_fake_llm():
    chunks = [chunk(1, "The Klein-Gordon equation is Box phi = m^2 phi."), chunk(2, "Wick theorem contractions.")]
    r = Retriever("t", dense=None, chunks=chunks)
    llm = FakeLLM(["The field obeys the Klein-Gordon equation [1]."])
    a = Answerer(r, llm).ask("Klein-Gordon equation")
    assert a.grounded and a.cited == [1]
    assert "Klein-Gordon" in llm.calls[0][1] and "[1]" in llm.calls[0][1]
