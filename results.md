# Phase B results

## Retrieval on the original 40 questions

Nomic results with full vectors. MRR measures the first gold hit within the top ten. Hit@1k is the fraction of questions with a gold block starting within the estimated 1,000-token context.

| Chunker | Hybrid R@5 | Hybrid MRR | Hybrid hit@1k | Dense R@5 | Dense hit@1k |
|---|---:|---:|---:|---:|---:|
| Fixed | .875 | .657 | .825 | .725 | .675 |
| Recursive | .850 | .661 | .875 | .575 | .600 |
| Section | .975 | .827 | .350 | .675 | .225 |
| Equation-aware | .825 | .572 | .800 | .500 | .525 |
| Hierarchical | .725 | .614 | .750 | .525 | .500 |

Section chunking led on Recall@5 and ranked last under the token budget. Recursive chunks returned gold context for 35 of 40 questions within 1,000 tokens. Full and funnel dense search had the same Recall@5 for all five chunkers.

The Mixedbread rerun used its full 1,024-dimensional vector. The dense hit rate stayed below the hybrid hit rate for every chunker with either model.

| Chunker | Mixedbread dense R@5 | Dense MRR | Dense hit@1k | Hybrid R@5 | Hybrid MRR | Hybrid hit@1k |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | .675 | .528 | .650 | .900 | .726 | .900 |
| Recursive | .625 | .493 | .625 | .875 | .652 | .875 |
| Section | .600 | .495 | .225 | .925 | .845 | .350 |
| Equation-aware | .600 | .472 | .600 | .850 | .560 | .850 |
| Hierarchical | .650 | .494 | .550 | .850 | .594 | .700 |

Mixedbread improved full-vector dense Recall@5 for three chunkers and reduced it for two. With hybrid retrieval, fixed edged ahead of recursive under the 1,000-token budget by one question (36/40 vs 35/40).

## Signal ablation

Hit@1k with Nomic hybrid search. Each dropped signal has weight zero. Candidate selection stays shared across the ablations.

| Chunker | Dense + BM25 | BM25 + symbol | Dense + symbol | All three |
|---|---:|---:|---:|---:|
| Fixed | .725 | .800 | .775 | .825 |
| Recursive | .725 | .850 | .800 | .875 |
| Section | .275 | .350 | .350 | .350 |
| Equation-aware | .725 | .775 | .725 | .800 |
| Hierarchical | .750 | .725 | .675 | .750 |

Across chunkers, the equation-query hit rate rose from .53 with dense + BM25 to .78 when symbol scoring was added. Concept queries stayed at .69.

## Student-style questions

The ten new prompts use conversational wording and share topics with the original benchmark. With Nomic hybrid search, they produced the following results. The pooled rate weights the original and student results by their 40 and 10 questions, respectively.

| Chunker | Student R@5 | Student MRR | Student hit@1k | Pooled hit@1k |
|---|---:|---:|---:|---:|
| Fixed | .300 | .267 | .300 | .720 |
| Recursive | .300 | .233 | .300 | .760 |
| Section | .800 | .561 | .200 | .320 |
| Equation-aware | .100 | .100 | .100 | .660 |
| Hierarchical | .100 | .068 | .000 | .600 |

Fixed and recursive tied on the new prompts at three budget hits each. The original ordering under the token budget holds when pooling all 50 questions. Section still leads on Recall@5.

## Retrieval failure audit

The ranks below are the first labeled hit with Nomic full-vector hybrid search. A dash means the labeled passage was absent from the top ten. The audit also checks whether the retrieved passages answer the question.

| Question | Fixed | Recursive | Section | Equation | Hierarchical | Diagnosis |
|---|---:|---:|---:|---:|---:|---|
| Q37 | — | — | 8 | — | — | Prose-to-equation retrieval miss. The mode-energy integral is in `6.2:b22`. The direct LaTeX query for the same equation (Q20) ranked it first in four strategies and second in equation-aware. |
| Q01 | — | — | 1 | — | — | Wrong passage. Most strategies prefer the action-formulation introduction around `6:b10`. The variation and boundary-term argument are later in `6:b16–b18`. |
| Q07 | 5 | 9 | 1 | 10 | — | Potentially incomplete gold. `21:b22` states the normal-ordering-plus-contractions version of the theorem and occurs in high-ranking results. The current gold includes the heading `11.1:b14` and the explicit expansion `11.1:b15`. |
| Q12 | 2 | 9 | 3 | 10 | — | Wrong passage. Several top chunks contain the conclusion at `20.2:b28` but lack the time slicing and intermediate steps. The top section result (§21.2) is the field-theory treatment. |
| Q14 | 2 | 6 | 1 | 9 | 10 | Within-section ranking error. Shorter chunks often rank later mentions of the beta function ahead of the defining equations in `22.2:b6` and `22.2:b8`. |
| Q29 | — | 4 | 4 | 9 | — | Wrong passage. Standard oscillator brackets in §7 and mode occupation numbers in `9.2:b7` share notation with the requested bracket in `7.2:b37`. |
| Q34 | — | 9 | 1 | 6 | 2 | Wrong passage. `6:b10` sets up the scalar action. The change of variables and invariant integration measure are in `6:b15`. |

## Funnel timing

Median top-five search time in milliseconds on local macOS, over 25 synthetic queries at each size. Both indexes use exact FAISS inner-product search on normalized vectors. The funnel retrieves 100 candidates at 128 dimensions and rescores them at 768. Timing includes ranking and excludes index construction and query embedding.

| Vectors | Full | Short | Funnel | Full / funnel |
|---:|---:|---:|---:|---:|
| 2,000 | 0.089 | 0.017 | 0.055 | 1.61× |
| 100,000 | 5.421 | 0.859 | 0.941 | 5.76× |
| 1,000,000 | 52.047 | 8.467 | 8.713 | 5.97× |

At corpus scale the median saving was 0.034 ms per synthetic query. At a million vectors the funnel took about one-sixth of the full search time. These measurements cover search latency only.

## Grounded answering

The live Granite run completed all 40 questions for each chunker. The earlier Q02 smoke test passed the mechanical check for both chunkers.

| Chunker | Completed | Mechanically grounded | Refused | Hit generation limit |
|---|---:|---:|---:|---:|
| Recursive | 40 | 21 (52.5%) | 0 | 10 |
| Equation-aware | 40 | 19 (47.5%) | 0 | 8 |

All 80 answers had nonempty text. Four of the 16 answers selected for human review passed the mechanical check. The source-support and correctness rates for that sample are pending human review.

## Reproduction

Build the corpus and chunks in the order given in the implementation guide. The commands below reproduce the measurements. [Setup and commands](implementation.md#9-setup-and-commands)

```bash
uv run pytest -q
uv run python -m src.evaluation.funnel_timing

for model in nomic mxbai; do
    uv run python -m src.retrieval.build --model "$model"
    uv run python -m src.evaluation.retrieval_eval --model "$model" --out "data/processed/results_${model}_retrieval.json"
    uv run python -m src.evaluation.budget_eval --model "$model" --out "data/processed/results_${model}_budget.json"
done
uv run python -m src.evaluation.ablation
uv run python -m src.evaluation.retrieval_eval --questions data/benchmark/student_questions.json --out data/processed/results_student_retrieval.json
uv run python -m src.evaluation.budget_eval --questions data/benchmark/student_questions.json --out data/processed/results_student_budget.json

uv run python -m src.evaluation.answer_eval --ids Q02 --out data/processed/answers_smoke.json
uv run python -m src.evaluation.answer_eval
```

The answer runs use local Ollama with `granite4.2:8b`, thinking disabled, and a 256-token generation limit for both chunkers. The JSON records Ollama's stop reason so length-limited answers can be identified. Review the same eight question IDs from both strategies for a 16-answer sample (Q01, Q07, Q09, Q12, Q15, Q24, Q34, Q37). Mark a generated answer as supported if every claim follows from its cited passage, and as correct if it answers the question accurately and completely. Track refusals separately as missed answers on this corpus-answerable benchmark. Compare mechanical and human support rates on the same reviewed responses and include the denominators.

## Interpretation limits

A chunk hit can include only part of a gold block. The token-budget metric estimates where each block starts, and hierarchical retrieval ranks sections, derivation groups, and leaves together. Each of the 40 original questions is worth 2.5 percentage points. Each of the ten student-style questions is worth 10 points, and those draft anchors still need independent review. The mechanical citation check also accepts refusals, so its rate alone cannot measure answer quality.

## Verification

The corpus rebuilt with 651 equations and the existing test suite passed all 40 tests. The original benchmark regenerated byte-for-byte. The answer output contains each original question exactly once per chunker, with numbered sources, line ranges, and citation verdicts. Human answer-quality labels and independent review of the new questions are pending.
