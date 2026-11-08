"""Token-budget evaluation: is the gold passage inside the first B tokens of retrieved context?

    uv run python -m src.evaluation.budget_eval [--budgets 500 1000 2000]

Plain Recall@k favours big chunks (a 3,000-token section "hits" because it contains more).
Here every method gets the same context budget B: take ranked chunks in order, charge each
chunk its token count, and truncate the last one at the budget. Within a chunk, blocks are
laid out in order (scaled so they sum to the chunk's token count); a gold block is SEEN if
it starts inside the budget. We report
  hit  = fraction of questions with at least one gold block seen
  cov  = mean fraction of a question's gold blocks seen
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import tiktoken

from ..parser.schema import Document
from ..retrieval.dense import PRESETS, DenseIndex
from ..retrieval.hybrid import Retriever
from .retrieval_eval import METHODS, QUESTIONS, ROOT, STRATEGIES, cached

DEPTH = 30
ENC = tiktoken.get_encoding("cl100k_base")


def block_tokens(doc: Document) -> dict[str, int]:
    return {b.block_id: max(1, len(ENC.encode(b.content))) for s in doc.iter_sections() for b in s.blocks}


def seen_blocks(hits, budget: int, btok: dict[str, int]) -> set[str]:
    seen, remaining = set(), budget
    for h in hits:
        if remaining <= 0:
            break
        c = h.chunk
        raw = [btok.get(b, 1) for b in c.block_ids]
        scale = c.n_tokens / max(1, sum(raw))
        pos = 0.0
        for b, t in zip(c.block_ids, raw):
            if pos < remaining:
                seen.add(b)
            pos += t * scale
        remaining -= c.n_tokens
    return seen


def run(model: str, budgets: list[int], strategies: list[str], weights=(1.0, 1.0, 1.0)) -> dict:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    btok = block_tokens(Document.load(ROOT / "data/processed/date_qft.json"))
    embedder = PRESETS[model]()
    out: dict = {}
    for s in strategies:
        r = cached(Retriever(s, dense=None))
        r.dense = DenseIndex(r.chunks, embedder, s)
        out[s] = {}
        for name, (mode, dm) in METHODS.items():
            acc = {b: {"hit": [], "cov": []} for b in budgets}
            for q in questions:
                hits = r.retrieve(q["query"], k=DEPTH, mode=mode, dense_mode=dm,
                                  weights=weights, with_parent=False)
                gold = set(q["gold_blocks"])
                for b in budgets:
                    seen = seen_blocks(hits, b, btok) & gold
                    acc[b]["hit"].append(1.0 if seen else 0.0)
                    acc[b]["cov"].append(len(seen) / len(gold))
            out[s][name] = {str(b): {k: sum(v) / len(v) for k, v in d.items()} for b, d in acc.items()}
    return out


def table(res: dict, budget: int, key: str) -> str:
    ms = list(METHODS)
    lines = ["| strategy | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
    for s in res:
        lines.append(f"| {s} | " + " | ".join(f"{res[s][m][str(budget)][key]:.2f}" for m in ms) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nomic")
    ap.add_argument("--budgets", type=int, nargs="*", default=[500, 1000, 2000])
    ap.add_argument("--out", default=str(ROOT / "data/processed/results_budget.json"))
    args = ap.parse_args()
    res = run(args.model, args.budgets, STRATEGIES)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for b in args.budgets:
        print(f"\n### hit @ {b} tokens\n" + table(res, b, "hit"))
    mid = args.budgets[len(args.budgets) // 2]
    print(f"\n### gold coverage @ {mid} tokens\n" + table(res, mid, "cov"))


if __name__ == "__main__":
    main()
