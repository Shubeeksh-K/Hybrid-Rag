"""Time exact full, exact short, and funnel search on synthetic vectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from ..retrieval.dense import faiss, np
from .retrieval_eval import ROOT


def measure(size: int, trials: int) -> dict[str, float]:
    # Add in batches so index construction does not copy the entire corpus at once.
    rng = np.random.default_rng(42)
    full = faiss.IndexFlatIP(768)
    short = faiss.IndexFlatIP(128)
    for start in range(0, size, 8192):
        vectors = rng.standard_normal((min(8192, size - start), 768), dtype=np.float32)
        prefix = vectors[:, :128].copy(order="C")
        faiss.normalize_L2(vectors)
        faiss.normalize_L2(prefix)
        full.add(vectors)
        short.add(prefix)

    queries = rng.standard_normal((trials, 768), dtype=np.float32)
    prefixes = queries[:, :128].copy(order="C")
    faiss.normalize_L2(queries)
    faiss.normalize_L2(prefixes)
    times = {"full": [], "short": [], "funnel": []}
    for i in range(trials):
        qf, qs = queries[i:i + 1], prefixes[i:i + 1]
        for mode in list(times)[i % 3:] + list(times)[:i % 3]:
            start = perf_counter()
            if mode == "full":
                full.search(qf, 5)
            elif mode == "short":
                short.search(qs, 5)
            else:
                _, ids = short.search(qs, min(100, size))
                scores = full.reconstruct_batch(ids[0]) @ qf[0]
                np.argsort(-scores)[:5]
            times[mode].append((perf_counter() - start) * 1000)
    return {mode: median(samples) for mode, samples in times.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[2000, 100000, 1000000])
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/results_funnel_timing.json")
    args = parser.parse_args()
    if min(args.sizes) < 5 or args.trials < 1:
        parser.error("sizes must be at least 5 and trials must be positive")

    results = {}
    print("vectors | full ms | short ms | funnel ms | full / funnel", flush=True)
    for size in args.sizes:
        row = measure(size, args.trials)
        results[size] = row
        print(f"{size:>7} | {row['full']:.3f} | {row['short']:.3f} | "
              f"{row['funnel']:.3f} | {row['full'] / row['funnel']:.2f}x", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
