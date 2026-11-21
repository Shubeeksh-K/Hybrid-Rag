"""Run cited answering on the original benchmark with two chunkers."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..rag.answer import Answerer
from ..rag.llm import OllamaLLM
from ..retrieval.dense import PRESETS, DenseIndex
from ..retrieval.hybrid import Retriever
from .retrieval_eval import QUESTIONS, ROOT, STRATEGIES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=["recursive", "equation"])
    parser.add_argument("--ids", nargs="+", help="Run only these question IDs")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/answers.json")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.ids:
        ids = set(args.ids)
        unknown = ids - {question["id"] for question in questions}
        if unknown:
            parser.error(f"unknown question IDs: {', '.join(sorted(unknown))}")
        questions = [question for question in questions if question["id"] in ids]

    embedder = PRESETS["nomic"]()
    llm = OllamaLLM(num_predict=256, think=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out.with_name(args.out.name + ".tmp")
    answerers = {}
    for strategy in args.strategies:
        retriever = Retriever(strategy)
        retriever.dense = DenseIndex(retriever.chunks, embedder, strategy)
        answerers[strategy] = Answerer(retriever, llm)

    results = {strategy: [] for strategy in answerers}
    for question in questions:
        for strategy, answerer in answerers.items():
            answer = answerer.ask(question["query"])
            results[strategy].append({"id": question["id"], **asdict(answer),
                                      "grounded": answer.grounded, "done_reason": llm.last_done_reason})
            checkpoint.write_text(json.dumps(results, indent=2), encoding="utf-8")
            checkpoint.replace(args.out)
            print(f"{strategy} {question['id']}: grounded={answer.grounded}, refused={answer.refused}", flush=True)

    for strategy, rows in results.items():
        grounded = sum(row["grounded"] for row in rows)
        refused = sum(row["refused"] for row in rows)
        print(f"{strategy}: {grounded}/{len(rows)} mechanically grounded, {refused} refusals")


if __name__ == "__main__":
    main()
