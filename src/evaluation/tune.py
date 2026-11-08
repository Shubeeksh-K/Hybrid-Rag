"""Tune hybrid weights on the dev half of the benchmark, report on the held-out test half.

    uv run python -m src.evaluation.tune

Weights are (dense, bm25, symbol). Compared per chunker, all measured on TEST questions:
  default  (1,1,1)
  global   one weight triple tuned on dev
  routed   queries containing math (symbols_of_query non-empty) use a triple tuned on the dev
           math queries; the rest use a (dense, bm25) pair tuned on the dev prose queries
Objective on dev: hit@1000-token budget + MRR (ties -> closest to equal weights).
Dense mode is "full" (funnel was equal or slightly worse in the baseline).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from ..parser.normalizer import symbols_of_query
from ..parser.schema import Document
from ..retrieval.dense import PRESETS, DenseIndex
from ..retrieval.hybrid import Retriever
from .budget_eval import block_tokens, seen_blocks
from .retrieval_eval import QUESTIONS, ROOT, STRATEGIES, cached, rank_of_first_hit, split_dev_test

GRID = [0.0, 0.5, 1.0, 2.0]
BUDGET = 1000
DEPTH = 30


def score(r: Retriever, qs: list[dict], weights, btok) -> tuple[float, float, float]:
    """(objective, hit@budget, mrr) averaged over qs."""
    hit = mrr = 0.0
    for q in qs:
        hits = r.retrieve(q["query"], k=DEPTH, mode="hybrid", dense_mode="full",
                          weights=weights, with_parent=False)
        gold = set(q["gold_blocks"])
        hit += 1.0 if seen_blocks(hits, BUDGET, btok) & gold else 0.0
        rk = rank_of_first_hit(hits, gold)
        mrr += 1 / rk if rk and rk <= 10 else 0.0
    n = len(qs)
    return (hit + mrr) / n, hit / n, mrr / n


def best_weights(r, qs, btok, free_symbol: bool) -> tuple:
    best, best_key = (1.0, 1.0, 1.0), None
    for w in itertools.product(GRID, GRID, GRID if free_symbol else [1.0]):
        if not any(w[:2]) and not (free_symbol and w[2]):
            continue
        obj = score(r, qs, w, btok)[0]
        dist = sum(abs(x - 1.0) for x in w)
        key = (round(obj, 6), -dist)
        if best_key is None or key > best_key:
            best, best_key = w, key
    return best


def main() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    split = split_dev_test(questions)
    dev = [q for q in questions if split[q["id"]] == "dev"]
    test = [q for q in questions if split[q["id"]] == "test"]
    has_math = {q["id"]: bool(symbols_of_query(q["query"])) for q in questions}
    btok = block_tokens(Document.load(ROOT / "data/processed/date_qft.json"))
    embedder = PRESETS["nomic"]()
    print(f"dev {len(dev)} / test {len(test)};  math queries: "
          f"{sum(has_math[q['id']] for q in dev)} dev, {sum(has_math[q['id']] for q in test)} test\n")

    results = {}
    for s in STRATEGIES:
        r = cached(Retriever(s, dense=None))
        r.dense = DenseIndex(r.chunks, embedder, s)
        w_glob = best_weights(r, dev, btok, True)
        w_math = best_weights(r, [q for q in dev if has_math[q["id"]]], btok, True)
        w_prose = best_weights(r, [q for q in dev if not has_math[q["id"]]], btok, False)

        def routed(qs):
            hit = mrr = 0.0
            for q in qs:
                w = w_math if has_math[q["id"]] else (w_prose[0], w_prose[1], 0.0)
                _, h, m = score(r, [q], w, btok)
                hit += h
                mrr += m
            return hit / len(qs), mrr / len(qs)

        row = {"weights": {"global": w_glob, "math": w_math, "prose": w_prose[:2]},
               "test": {"default": score(r, test, (1, 1, 1), btok)[1:],
                        "global": score(r, test, w_glob, btok)[1:],
                        "routed": routed(test)},
               "dev": {"default": score(r, dev, (1, 1, 1), btok)[1:],
                       "global": score(r, dev, w_glob, btok)[1:],
                       "routed": routed(dev)}}
        results[s] = row
        print(f"{s:<13} global={w_glob}  math={w_math}  prose(d,b)={w_prose[:2]}")

    print("\nTEST (held out)   hit@1000tok / MRR")
    print("| strategy | default (1,1,1) | global | routed |\n|---|---|---|---|")
    for s, row in results.items():
        print(f"| {s} | " + " | ".join(f"{row['test'][k][0]:.2f} / {row['test'][k][1]:.2f}"
                                       for k in ("default", "global", "routed")) + " |")
    print("\nDEV (tuned on; optimistic)")
    print("| strategy | default | global | routed |\n|---|---|---|---|")
    for s, row in results.items():
        print(f"| {s} | " + " | ".join(f"{row['dev'][k][0]:.2f} / {row['dev'][k][1]:.2f}"
                                       for k in ("default", "global", "routed")) + " |")
    out = ROOT / "data/processed/results_tuned.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
