"""Signal ablations and failure analysis (hybrid, equal weights, full dense).

    uv run python -m src.evaluation.ablation
"""
from __future__ import annotations

import json
from collections import defaultdict

from ..parser.schema import Document
from ..retrieval.dense import PRESETS, DenseIndex
from ..retrieval.hybrid import Retriever
from .budget_eval import block_tokens, seen_blocks
from .retrieval_eval import QUESTIONS, ROOT, STRATEGIES, cached, rank_of_first_hit

COMBOS = {  # (dense, bm25, symbol)
    "dense+bm25": (1, 1, 0), "bm25+symbol": (0, 1, 1), "dense+symbol": (1, 0, 1),
    "all three": (1, 1, 1),
}
BUDGET = 1000


def main() -> None:
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    btok = block_tokens(Document.load(ROOT / "data/processed/date_qft.json"))
    embedder = PRESETS["nomic"]()
    sec_title = {s.section_id: s.title for s in Document.load(ROOT / "data/processed/date_qft.json").iter_sections()}
    ranks: dict = {}   # (strategy, combo) -> {qid: rank}
    budget_hit: dict = {}
    top1: dict = {}
    for s in STRATEGIES:
        r = cached(Retriever(s, dense=None))
        r.dense = DenseIndex(r.chunks, embedder, s)
        for name, w in COMBOS.items():
            rk, bh = {}, {}
            for q in qs:
                hits = r.retrieve(q["query"], k=30, mode="hybrid", dense_mode="full",
                                  weights=w, with_parent=False)
                gold = set(q["gold_blocks"])
                rk[q["id"]] = rank_of_first_hit(hits[:10], gold)
                bh[q["id"]] = bool(seen_blocks(hits, BUDGET, btok) & gold)
                if name == "all three":
                    top1[(s, q["id"])] = hits[0].chunk.section_id
            ranks[(s, name)], budget_hit[(s, name)] = rk, bh

    print(f"### Ablation: hit@{BUDGET}tok / R@5 / MRR over all 40 questions\n")
    print("| strategy | " + " | ".join(COMBOS) + " |\n|---|" + "---|" * len(COMBOS))
    for s in STRATEGIES:
        cells = []
        for name in COMBOS:
            rk = ranks[(s, name)]
            n = len(rk)
            cells.append(f"{sum(budget_hit[(s, name)].values()) / n:.2f} / "
                         f"{sum(1 for v in rk.values() if v and v <= 5) / n:.2f} / "
                         f"{sum(1 / v for v in rk.values() if v) / n:.2f}")
        print(f"| {s} | " + " | ".join(cells) + " |")

    print("\n### By question type, 'all three' vs best pair (hit@1000tok), averaged over the 5 chunkers\n")
    types = sorted({q["type"] for q in qs})
    print("| type | " + " | ".join(COMBOS) + " |\n|---|" + "---|" * len(COMBOS))
    for t in types:
        ids = [q["id"] for q in qs if q["type"] == t]
        cells = []
        for name in COMBOS:
            v = [budget_hit[(s, name)][i] for s in STRATEGIES for i in ids]
            cells.append(f"{sum(v) / len(v):.2f}")
        print(f"| {t} ({len(ids)}) | " + " | ".join(cells) + " |")

    print("\n### Failures: 'all three', question misses top-5 (chunk-hit) in how many of the 5 chunkers\n")
    fails = defaultdict(list)
    for q in qs:
        miss = [s for s in STRATEGIES if not (ranks[(s, "all three")][q["id"]] and ranks[(s, "all three")][q["id"]] <= 5)]
        if miss:
            fails[q["id"]] = miss
    for q in qs:
        if q["id"] in fails:
            miss = fails[q["id"]]
            gold_secs = sorted({b.split(":")[0] for b in q["gold_items"]})
            got = sorted({top1[(s, q["id"])] for s in miss})
            print(f"- {q['id']} [{q['type']}] misses in {len(miss)}/5: {q['query'][:70]!r}  "
                  f"gold §{','.join(gold_secs)}; top-1 landed in §{','.join(got)}")
    print(f"\nnever found by any chunker: {[i for i, m in fails.items() if len(m) == 5]}")
    (ROOT / "data/processed/results_ablation.json").write_text(json.dumps(
        {f"{s}|{n}": v for (s, n), v in ranks.items()}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
