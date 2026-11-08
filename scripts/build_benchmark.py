"""Build the DRAFT retrieval benchmark: data/benchmark/questions.json.

Each question has a gold answer expressed as source anchors (equation ids, block ids,
or Final.tex line ranges), never chunk ids, so every chunking method is scored on the
same truth. A chunk is a hit when it touches at least one gold block.

Gold labels here are PROPOSALS written by reading the notes. They must be verified
by a human (see scripts/build_review_page.py) before being used for results.

    uv run python scripts/build_benchmark.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.parser.normalizer import symbols_of_latex
from src.parser.schema import Document

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "data/processed/date_qft.json"
OUT = ROOT / "data/benchmark/questions.json"
REVIEWERS = ["Shubeeksh", "Helen", "Sanjay", "Shreya"]

# (type, query, what-the-answer-should-show, gold anchors, confidence)
# type: concept | equation | notation | context
Q = [
    # ---------------- concept: natural-language, answer is an argument/derivation ------------
    ("concept", "How is the Klein-Gordon equation derived from an action principle?",
     "Vary S[phi] = int(-1/2 d phi d phi - 1/2 m^2 phi^2); the boundary term drops and (Box - m^2)phi = 0 follows.",
     ["6:eq3", "6:b17", "6:b18"], "high"),
    ("concept", "What does Noether's theorem say about symmetries of the action?",
     "Every infinitesimal symmetry of the action gives a current conserved on solutions of the equations of motion.",
     ["6.1:b9", "6.1:b23", "6.1:eq8"], "high"),
    ("concept", "How are the irreducible representations of the Poincare group labelled by Casimir invariants?",
     "P^2 (the mass) and the square of the Pauli-Lubanski vector are Casimirs whose values classify the representations.",
     ["2.2:b3", "2.2:b4", "2.1:b22", "2.3:eq6"], "medium"),
    ("concept", "What are the retarded, advanced and Feynman prescriptions for the Green's function of the Klein-Gordon operator?",
     "The same Green's function integral with different contour choices gives retarded, advanced or Feynman propagators.",
     ["7.3:eq4", "7.3:eq5", "7.3:eq6", "7.3:b18"], "high"),
    ("concept", "What is the Kallen-Lehmann spectral representation of the two-point function?",
     "The interacting two-point function is a superposition of free propagators weighted by a spectral density rho(sigma^2).",
     ["10.3:eq6", "10.3:b15"], "medium"),
    ("concept", "How does LSZ reduction express S-matrix elements through time-ordered correlation functions of the scalar field?",
     "Asymptotic states are replaced by field insertions acted on by (Box - m^2); S-matrix = amputated correlation function.",
     ["10.5.1:eq6", "10.5.1:eq7", "10.5.1:b17"], "high"),
    ("concept", "What does Wick's theorem state for a time-ordered product of fields?",
     "A time-ordered product equals the normal-ordered product plus all contractions (pairings by propagators).",
     ["11.1:b14", "11.1:eq2"], "high"),
    ("concept", "How is the Ward identity q_mu M^mu = 0 proved for the photon self-energy?",
     "Insert q-slash between propagators, use 1/(p+q-slash+m) q-slash 1/(p-slash+m) identity; the sum telescopes to zero.",
     ["16.2.1:eq3", "16.2.1:eq5", "16.2.1:eq8", "16.2.1:b20"], "high"),
    ("concept", "How does the anomalous magnetic moment of the electron arise from the vertex form factors?",
     "Magnetic moment is proportional to F1(0)+F2(0); g=2 is Dirac, and F2(0) != 0 (a QED effect) is the anomalous moment.",
     ["16.4:b23", "16.4:eq7", "16.4:eq8"], "high"),
    ("concept", "How does vacuum polarization lead to charge screening and a running coupling?",
     "The dressed photon propagator carries 1/(1 - Pi(q^2)); Pi(q^2) - Pi(0) makes the effective alpha grow with |q^2|.",
     ["18.2.1:eq1", "18.2.1:eq4"], "medium"),
    ("concept", "How does power counting determine the superficial degree of divergence of a Feynman diagram?",
     "Count loop momenta and propagators: D = d - (d-2)E/2 + sum n_k (k(d-2)/2 - d).",
     ["19.1:b3", "19.1:eq1"], "high"),
    ("concept", "How is the path integral derived from the quantum-mechanical transition amplitude?",
     "Slice the time evolution into N steps, insert complete sets of states, and take N -> infinity to get int Dq Dp exp(iS).",
     ["20.2:b1", "20.2:b3", "20.2:eq7", "20.2:eq10", "20.2:eq12"], "high"),
    ("concept", "How is the effective action defined as the Legendre transform of the generating functional W[J]?",
     "Gamma[Phi] = W[J] - int J Phi, generating 1PI diagrams.",
     ["22.1:b12", "22.1:eq10"], "high"),
    ("concept", "What is the renormalization group equation and how is the beta function defined?",
     "mu d Gamma_n/d mu = 0 gives (mu d/dmu + beta d/dg - n gamma) Gamma_n = 0, with beta = mu dg/dmu.",
     ["22.2:eq3", "22.2:eq4"], "high"),
    # ---------------- equation: query is a LaTeX formula, answer is that equation ------------
    ("equation", r"(\Box - m^2)\phi = 0",
     "The Klein-Gordon field equation (appears inside the action variation and in the field-equation summary).",
     ["6:b18", "6:eq3", "3.3:eq4"], "medium"),
    ("equation", r"\gamma^{\mu}\gamma^{\nu} + \gamma^{\nu}\gamma^{\mu} = -2\eta^{\mu\nu}",
     "Defining relation of the Clifford algebra of gamma matrices (mostly-plus metric convention).",
     ["3.3:eq2"], "high"),
    ("equation", r"\left[M^{\mu\nu}, M^{\alpha\beta}\right] = i\left(\eta^{\mu\alpha}M^{\nu\beta} - \dots\right)",
     "Lorentz-algebra commutator of the Poincare generators.",
     ["2.1:eq6"], "high"),
    ("equation", r"W^2|\hat{k},\sigma\rangle = m^2 j(j+1)|\hat{k},\sigma\rangle",
     "Eigenvalue of the squared Pauli-Lubanski vector on massive states.",
     ["2.3:eq6"], "high"),
    ("equation", r"T_{\mu\nu} = \partial_{\mu}\phi\partial_{\nu}\phi + \eta_{\mu\nu}\mathcal{L}",
     "Energy-momentum tensor of the free scalar field.",
     ["6.2:eq1", "6.1:eq8"], "high"),
    ("equation", r"P^0 = \int d^3k\ \omega_{\vec{k}}\ a(\vec{k})a^*(\vec{k})",
     "Field energy in terms of mode amplitudes.",
     ["6.2:eq10"], "high"),
    ("equation", r"\Gamma[\Phi] := W[J] - \int d^4x\, J(x)\Phi(x)",
     "Legendre transform defining the effective action.",
     ["22.1:eq10"], "high"),
    ("equation", r"D = d - \frac{d-2}{2}E + \sum_{k\ge 3}n_k\left\{k\frac{d-2}{2} - d\right\}",
     "Superficial degree of divergence.",
     ["19.1:eq1", "19.1:eq3"], "high"),
    ("equation", r"\frac{1}{\omega'} - \frac{1}{\omega} = \frac{1-\cos\theta}{m}",
     "Compton formula relating scattered photon frequency to angle.",
     ["14.2:eq18"], "high"),
    ("equation", r"\beta(g) := \mu\frac{dg}{d\mu}",
     "Definition of the beta function (together with the anomalous dimension).",
     ["22.2:eq4"], "high"),
    ("equation", r"(\Box_x - m^2)G(x,x') = -\delta^4(x-x')",
     "Equation defining the Green's function of the Klein-Gordon operator.",
     ["7.3:eq1"], "high"),
    ("equation", r"(-i\gamma^{\mu}\partial_{\mu} + m)\Psi = 0",
     "The Dirac equation.",
     ["4:eq11", "4:eq10", "7.2:eq2"], "medium"),
    # ---------------- notation: question built around specific symbols ----------------
    ("notation", r"What is F_{\mu\nu} = \partial_{\mu}A_{\nu} - \partial_{\nu}A_{\mu} in terms of the Fourier modes of the Maxwell field?",
     "Field strength written through mode expansion of A_mu.",
     ["7.1:eq2"], "high"),
    ("notation", r"Why is J^0(v,u) = v^*\partial^0 u - u\partial^0 v^* a conserved inner product?",
     "Current J^mu built from two solutions is divergence-free, so its time component integrates to a constant.",
     ["4:eq3", "4:eq2", "4:eq1"], "high"),
    ("notation", r"What is the Poisson bracket \{a_{\vec{k}}, a^*_{\vec{k}'}\} of the mode amplitudes?",
     "Bracket is -i delta^3(k - k').",
     ["7.2:eq15"], "high"),
    ("notation", r"\sum_{\sigma}u(\hat{k},\sigma)u^{\dagger}(\hat{k},\sigma) spin sum for Dirac spinors",
     "Completeness sum over spinors at rest, (1 + gamma^0)/2.",
     ["4:eq19"], "high"),
    ("notation", r"Which condition do the Lorentz matrices \Lambda^{\mu}_{~\alpha}\Lambda^{\nu}_{~\beta}\eta^{\alpha\beta} satisfy?",
     "Preserving the Minkowski metric: Lambda Lambda eta = eta.",
     ["2.1:eq1", "2.1:eq2"], "high"),
    ("notation", r"[J_i, J_j] = i\epsilon_{ij}^{~~k}J_k commutators of rotation and boost generators",
     "Poincare Lie algebra written in terms of J_i, K_j, P_mu.",
     ["2.1:eq7"], "high"),
    ("notation", r"How do correlation functions \langle 0|T\{\Phi(x_1)\cdots\Phi(x_n)\}|0\rangle follow from derivatives \delta^n Z[J]/\delta J^n?",
     "n-th functional derivative of Z[J] at J = 0, normalized, gives the n-point function (with factors of i).",
     ["20.4:eq7", "20.3:eq4"], "medium"),
    # ---------------- context: answer needs a few paragraphs of explanation ----------------
    ("context", "Why is the integral of the scalar Lagrangian density over Minkowski space Poincare invariant?",
     "Under x -> Lambda x + a the contracted derivative term is a scalar and the measure is unchanged.",
     ["L3127-3134", "6:eq2"], "high"),
    ("context", "How does one derive the relation between photon energy and scattering angle in Compton scattering?",
     "Use four-momentum conservation with p^2 = -m^2 to get m(omega - omega') = omega omega' (1 - cos theta).",
     ["14.2:eq17", "14.2:eq18"], "high"),
    ("context", "How does the Thomson cross-section arise as the low-energy limit of Compton scattering?",
     "For omega/m << 1 the Klein-Nishina result reduces to the classical Thomson cross-section.",
     ["14.2:eq23", "14.2:b58"], "high"),
    ("context", "How is the energy of a free scalar field expressed in terms of mode amplitudes?",
     "Insert the mode expansion into P^0 and use orthogonality; cross terms with a(k)a(-k) drop out, leaving omega|a|^2.",
     ["6.2:eq9", "6.2:eq10"], "high"),
    ("context", "How does the Pauli exclusion principle follow from relativistic causality?",
     "Micro-causality (fields commute/anticommute at spacelike separation) forces anti-commutators for spin-1/2 fields.",
     ["8.4:b1", "8.4:b7", "8.4:b8"], "high"),
    ("context", "Why does a quantum field evolve into a coherent state under a classical source?",
     "A source term linear in the field displaces the vacuum into a coherent state.",
     ["9.4:b1", "9.4:b2"], "high"),
    ("context", "Why is the Lagrangian density invariant under the CPT transformation?",
     "Each tensor index and each derivative contributes a sign; the signs cancel in a Lorentz-scalar Lagrangian.",
     ["8.3:b11", "8.3:b12"], "high"),
]


def resolve(doc: Document, item: str) -> list[str]:
    blocks = [(s, b) for s in doc.iter_sections() for b in s.blocks]
    if item.startswith("L"):
        a, b = map(int, item[1:].split("-"))
        return [bl.block_id for _, bl in blocks if bl.line_start <= b and bl.line_end >= a]
    if ":eq" in item:
        return [bl.block_id for _, bl in blocks if bl.equation_id == item]
    return [bl.block_id for _, bl in blocks if bl.block_id == item]


def similar_equations(doc: Document, gold_eqs: list[str], threshold: float = 0.6) -> list[str]:
    """Other equations with very similar notation: candidates the reviewer may also accept."""
    sets = {b.equation_id: set(symbols_of_latex(b.content)) for _, b in doc.iter_equations()}
    out = set()
    for g in gold_eqs:
        if g not in sets or len(sets[g]) < 3:
            continue
        for e, s in sets.items():
            if e in gold_eqs or e == g or len(s) < 3:
                continue
            if len(sets[g] & s) / len(sets[g] | s) >= threshold:
                out.add(e)
    return sorted(out)


def main() -> None:
    doc = Document.load(DOC)
    sec_of = {b.block_id: s.section_id for s in doc.iter_sections() for b in s.blocks}
    questions = []
    for i, (qtype, query, hint, items, conf) in enumerate(Q):
        blocks, bad = [], []
        for it in items:
            r = resolve(doc, it)
            (blocks.extend(r) if r else bad.append(it))
        assert not bad, f"unresolved gold in Q{i + 1}: {bad}"
        blocks = list(dict.fromkeys(blocks))
        eqs = [it for it in items if ":eq" in it]
        questions.append({
            "id": f"Q{i + 1:02d}",
            "type": qtype,
            "query": query,
            "answer_hint": hint,
            "gold_items": items,
            "gold_blocks": blocks,
            "gold_sections": sorted({sec_of[b] for b in blocks}),
            "similar_equations": similar_equations(doc, eqs) if qtype in ("equation", "notation") else [],
            "confidence": conf,
            "reviewer": REVIEWERS[i % len(REVIEWERS)],
            "status": "draft",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print(f"wrote {len(questions)} questions -> {OUT}")
    print("by type:", dict(Counter(q["type"] for q in questions)))
    print("by confidence:", dict(Counter(q["confidence"] for q in questions)))
    print("by reviewer:", dict(Counter(q["reviewer"] for q in questions)))
    print("gold blocks per question: min", min(len(q["gold_blocks"]) for q in questions),
          "max", max(len(q["gold_blocks"]) for q in questions))


if __name__ == "__main__":
    main()
