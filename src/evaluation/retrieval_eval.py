"""Retrieval evaluation: chunking strategy x search method on the benchmark.

    uv run python -m src.evaluation.retrieval_eval [--model nomic] [--out data/processed/results_baseline.json]

A retrieved chunk is a HIT if its block_ids intersect the question's gold blocks (gold is
anchored to source blocks, not to any chunker's chunk ids, so every chunker is judged on
the same ground truth). Recall@k = fraction of questions with a hit in the top k;
MRR = mean of 1/rank of the first hit (ranks beyond MRR_DEPTH count as 0).
Also reports the mean token size of the top-1 chunk, because a giant chunk "hits" easily.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..retrieval.dense import PRESETS, DenseIndex
from ..retrieval.hybrid import Retriever

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "data/benchmark/questions.json"
STRATEGIES = ["fixed", "recursive", "section", "equation", "hierarchical"]
KS = (1, 3, 5)
MRR_DEPTH = 10

# name -> (retrieve mode, dense_mode)
METHODS = {
    "bm25": ("bm25", "funnel"),
    "symbol": ("symbol", "funnel"),
    "dense-full": ("dense", "full"),
    "dense-short": ("dense", "short"),
    "dense-funnel": ("dense", "funnel"),
    "hybrid-full": ("hybrid", "full"),
    "hybrid-funnel": ("hybrid", "funnel"),
    "rrf-funnel": ("rrf", "funnel"),
}


def split_dev_test(questions: list[dict]) -> dict[str, str]:
    """Deterministic split, alternating within each question type (type-balanced)."""
    seen: dict[str, int] = defaultdict(int)
    out = {}
    for q in questions:
        out[q["id"]] = "dev" if seen[q["type"]] % 2 == 0 else "test"
        seen[q["type"]] += 1
    return out


def cached(retriever: Retriever) -> Retriever:
    """Memoize per-(query, dense_mode) signal scores so methods share one embedding pass."""
    raw, memo = retriever.signal_scores, {}

    def wrapped(query, dense_mode="funnel"):
        key = (query, dense_mode)
        if key not in memo:
            memo[key] = raw(query, dense_mode)
        return memo[key]

    retriever.signal_scores = wrapped
    return retriever


def rank_of_first_hit(hits, gold: set[str]) -> int | None:
    for h in hits:
        if gold.intersection(h.chunk.block_ids):
            return h.rank
    return None


def evaluate(retriever: Retriever, questions: list[dict], mode: str, dense_mode: str,
             weights=(1.0, 1.0, 1.0)) -> list[dict]:
    rows = []
    for q in questions:
        hits = retriever.retrieve(q["query"], k=MRR_DEPTH, mode=mode, dense_mode=dense_mode,
                                  weights=weights, with_parent=False)
        r = rank_of_first_hit(hits, set(q["gold_blocks"]))
        rows.append({"id": q["id"], "type": q["type"], "rank": r,
                     "top1_tokens": hits[0].chunk.n_tokens if hits else 0})
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {f"R@{k}": sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n for k in KS}
    out["MRR"] = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    out["top1_tok"] = sum(r["top1_tokens"] for r in rows) / n
    out["n"] = n
    return out


def table(results: dict, metric: str) -> str:
    methods = list(METHODS)
    lines = ["| strategy | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for s in results:
        lines.append(f"| {s} | " + " | ".join(f"{results[s][m]['all'][metric]:.2f}" for m in methods) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(PRESETS), default="nomic")
    ap.add_argument("--out", default=str(ROOT / "data/processed/results_baseline.json"))
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    split = split_dev_test(questions)
    embedder = PRESETS[args.model]()
    results: dict = {}
    for s in args.strategies:
        r = cached(Retriever(s, dense=None))
        r.dense = DenseIndex(r.chunks, embedder, s)
        results[s] = {}
        for name, (mode, dm) in METHODS.items():
            rows = evaluate(r, questions, mode, dm)
            by_type, by_split = defaultdict(list), defaultdict(list)
            for row in rows:
                by_type[row["type"]].append(row)
                by_split[split[row["id"]]].append(row)
            results[s][name] = {"all": summarize(rows),
                                "by_type": {t: summarize(v) for t, v in by_type.items()},
                                "by_split": {t: summarize(v) for t, v in by_split.items()},
                                "ranks": {row["id"]: row["rank"] for row in rows}}
        print(f"done {s}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    for metric in ("R@1", "R@3", "R@5", "MRR", "top1_tok"):
        print(f"\n### {metric}\n" + table(results, metric))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
