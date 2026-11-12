# QFT-RAG: implementation guide

Audience: teammates and their coding agents. Read this before changing anything.
`progress.md` is the short status page. This file explains **what each part does, why it was built that way, and what it assumes**, so you can change it without breaking the comparison the project depends on.

Last updated 2026-11-12. Everything below describes the code as committed through "Phase 6A" plus `progress.md`.

---

## 0. Orientation in five minutes

**Project aim.** Compare *how to chunk* LaTeX-heavy physics notes and *which search signals to combine*, on a hand-labelled benchmark. The product (grounded answers, derivation checking, demo) sits on top of the winning retrieval. The comparison is the core result; a change that makes one chunker or one method look better for a reason unrelated to its design damages the project.

**Corpus.** G. Date, *Lectures on Introduction to Quantum Field Theory*, arXiv:2207.02243. One file, `Final.tex`, about 20,000 lines, mostly-plus metric (`eta = diag(-1,1,1,1)`), many `eqnarray` environments.

**Data flow.**

```
Final.tex --parser--> date_qft.json (Document: Section > Block)
   --normalizer--> Block.symbols filled for equations
   --5 chunkers--> data/processed/chunks_<strategy>.jsonl
   --retrieval.build--> data/processed/index/<strategy>__nomic768.npy (cached embeddings)
   Retriever(strategy) = BM25 + Symbol + Dense(FAISS)  --hybrid--> ranked chunks
   ranked chunks --> evaluation (benchmark)   and   --> rag.answer (LLM with citations)
```

**What is and isn't in git.** Code, tests, `data/benchmark/questions.json`, and the two `.md` files are committed. `data/raw/` and `data/processed/` are gitignored because they contain Date's text. Rebuild them with the commands in section 9.

**The three rules that keep the comparison fair** (details in section 5):
1. All five chunkers read the same parsed `Document`. Only the splitting differs.
2. Gold labels are source *block IDs*, not chunk IDs, so they are chunker-independent.
3. The heading path is added at index time identically for every chunker.

---

## 1. Glossary (terms are used strictly)

| Term | Meaning here |
|---|---|
| **Matryoshka** | Matryoshka *embeddings* (MRL): one model whose vectors can be truncated to the first *d* dimensions. Used for the short-vector shortlist, then full-size rescoring. **Not** the nested chunking. |
| **Hierarchical** | The nested *chunking* method: section > derivation > leaf. Independent of Matryoshka. Both can be combined in any search. |
| **Block** | Smallest unit of the parsed document: one paragraph (`text`) or one display equation (`equation`). |
| **Chunk** | A retrievable unit produced by a chunker, made from one contiguous span of the flattened text. Records which blocks it touches. |
| **Gold blocks** | The block IDs that answer a benchmark question. |
| **Hit** | A retrieved chunk whose `block_ids` intersect the question's gold blocks. |
| **Signal** | One scoring method: `bm25`, `symbol`, or `dense`. |
| **Hybrid** | Weighted sum of normalized signal scores. |
| **Funnel** | Dense mode: shortlist with 128-d vectors, rescore the shortlist with full vectors. |
| **header_path** | `"6 Relativistic Actions > 6.1 Variational ..."`: the section chain of a chunk. |

---

## 2. Repository map

```
scripts/
  fetch_data.py            download + extract the arXiv e-print into data/raw/date2207
  build_benchmark.py       the 40 questions + gold anchors -> data/benchmark/questions.json
  build_review_page.py     (obsolete) HTML review page; xlsx script imports two helpers from it
  build_review_xlsx.py     (obsolete) spreadsheet for label review; benchmark was checked by hand instead
src/parser/
  schema.py                Block, Section, Document dataclasses (+ JSON save/load)
  parser.py                LaTeX -> Document
  normalizer.py            LaTeX math -> canonical symbol tokens
src/chunking/
  base.py                  Chunk record, flatten(), make_chunk(), token counting, JSONL I/O, stats
  fixed.py recursive.py section.py     baselines
  equation.py hierarchical.py          the two proposed methods
  run.py                   build all five chunk files
src/retrieval/
  bm25.py                  BM25 + LaTeX-aware tokenizer (LexicalIndex)
  symbol.py                SymbolIndex (BM25 over normalized symbol tokens)
  dense.py                 embedder wrapper, MRL truncation, FAISS indexes, full/short/funnel
  hybrid.py                Retriever: bm25|dense|symbol|hybrid|rrf
  build.py                 embed all chunk files once (cache)
src/evaluation/
  retrieval_eval.py        Recall@1/3/5, MRR grid; dev/test split helper
  budget_eval.py           hit within a token budget
  tune.py                  weight tuning on dev, report on test
  ablation.py              signal ablations + failure listing
src/rag/
  llm.py                   OllamaLLM (HTTP) and FakeLLM (tests)
  answer.py                grounded answering with citation verification
tests/                     parser, chunking, retrieval (fake embedder), rag (fake LLM)
```

---

## 3. Stage 1: parsing (`src/parser`)

### Aim
Turn one messy LaTeX file into a clean, shared intermediate representation (IR) that every chunker consumes, while **preserving the original LaTeX** of every equation and paragraph. Math must stay raw LaTeX: the symbol signal and any later checking need the exact source.

### Schema (`schema.py`)
- `Block(block_id, type, content, line_start, line_end, equation_id, env, label, symbols)`
  - `type` is `"text"` or `"equation"`. `content` is raw LaTeX (text blocks keep inline `$...$`).
  - `block_id` looks like `6.1:b17` (section id + running block number in that section). `equation_id` looks like `6.1:eq3`: the equation counter is **our own, per section**. It is not the PDF's equation number.
  - `env` is the original environment (`equation`, `eqnarray`, `align`, `\[`, ...). `symbols` is filled by the normalizer for equations.
- `Section(section_id, level, title, parent_id, label, blocks, children)`: a section node's `blocks` are only the blocks before its first subsection; subsections are separate nodes.
- `Document(doc_id, title, source, sections)` with `iter_sections()` (pre-order over the whole tree), `iter_equations()`, `save()`, `load()`.

### Parser behaviour and why (`parser.py`)
- Comments stripped. `\section/\subsection/\subsubsection` become a nested tree.
- Math environments (`equation`, `eqnarray`, `align`, ... plus `\[ ... \]` and a top-level `array`) become **equation blocks**. Blank lines inside an equation are removed, because they would break the recursive splitter and are not meaningful in math.
- **Figures and TikZ are dropped** (decision: they carry no searchable text, and a diagram cannot be matched by any of our signals).
- **Tables keep only their equations** (an early version dropped tables wholesale and lost 7 amplitude equations; fixed).
- Transparent wrappers (`center`, `minipage`, `itemize`, `\fbox`, `\scalebox`, ...) are blanked so their contents flow as normal text.
- Result for Date's notes: 23 top-level / 60 / 9 sections (level 1/2/3), 91 sections in total, 1,485 text blocks, 651 equations. The parser has a cross-check against a raw count of math environments (`--report`).

### Assumptions
- Single `.tex` file with `\section`-style headings. Multi-file sources or heavy custom macros would need work.
- Date's source is `eqnarray`-heavy; the parser is tuned to it but written generically.

### How to change it safely
Any parser change alters block IDs, which silently invalidates the benchmark gold labels and the cached embeddings. After touching the parser: rebuild everything (section 9), re-run `scripts/build_benchmark.py`, and check that every gold anchor still resolves (the script raises on an unresolvable anchor).

---

## 4. Stage 2: symbol normalizer (`src/parser/normalizer.py`)

### Aim
Make notation matchable. `\partial_\mu\phi`, `\partial^\mu \varphi` and `∂_μ ϕ` should look alike to a search engine; embeddings blur symbols and keyword search sees only strings.

### Behaviour
`symbols_of_latex(latex)` returns canonical tokens. Greek variants collapse (`\phi`, `\varphi`, `ϕ`, `φ` -> `PHI`). Each symbol emits up to **three tokens at different granularity**, so partial matches still score:

| Granularity | Example for `\partial_\mu\phi` | Purpose |
|---|---|---|
| atom | `PARTIAL`, `PHI` | match the ingredients |
| generic | `PARTIAL[_](PHI)` | match the structure, index letters relabelled to `_` |
| exact | `PARTIAL[mu](PHI)` | match the exact index choice |

Other normalisations: `g_{\mu\nu}` -> `g[mu,nu]`, `g^{\mu\nu}` -> `g^[mu,nu]`, `m^2` -> `m^[2]`.

Other entry points: `symbols_of_text` (inline math inside prose blocks), `symbols_of_query` (math in a user query: delimited math if present, otherwise the whole query if it *looks like* raw LaTeX, i.e. contains `\`, `^` or `_`, after dropping prose words), `math_segments`, `fill_block_symbols(doc)`.

`python -m src.parser.normalizer` fills `Block.symbols` for all equations and rewrites `date_qft.json`.

### Assumptions and limits
- A pragmatic parser for common physics notation, **not a TeX engine**. Macros, `\left/\right` edge cases and unusual constructs may be tokenised imperfectly.
- Query-side detection is a heuristic: a plain-English question gives no symbols, so the symbol signal contributes zero (it "abstains"). This is intended.

---

## 5. Stage 3: chunking (`src/chunking`)

### Aim
Test whether equation-aware structure beats structure-blind splitting for LaTeX-heavy notes. Five chunkers, **same input, same output type, same index-time treatment**.

### Common machinery (`base.py`)
- `flatten(doc)` produces one string plus `Seg` records mapping character ranges to blocks. Blocks are separated by `"\n\n"`. Equations are re-wrapped (`\begin{env}...\end{env}` or `\[ ... \]`) so a text-only pipeline sees what it would see in real LaTeX.
- `make_chunk(flat, start, end, strategy, n)` builds a chunk from a character span and fills in everything automatically: `block_ids` (every block the span touches, even partly), `line_start/line_end` (span in `Final.tex`), `n_tokens`, `equation_ids` (equations fully inside) and `cut_equation_ids` (equations only partly inside).
- Tokens are counted with tiktoken `cl100k_base`. Defaults: `CHUNK_TOKENS = 256`, `OVERLAP_TOKENS = 32`.
- `Chunk.text` is the raw body only. `header_path` is stored separately and **prepended only when indexing** by `index_text(chunk)`. This is deliberate: every strategy gets identical heading context, so none is advantaged by it. All indexes (BM25, symbol, dense) use `index_text`.
- Chunk IDs are `"<strategy>:<NNNN>"`. Files are `data/processed/chunks_<strategy>.jsonl`.

### The five strategies

| Strategy | Idea | Why it is in the study | Result counts |
|---|---|---|---|
| `fixed` | Windows of 256 tokens, 32 overlap, over the whole flattened document | What a plain RAG tutorial does. Baseline 1. Structure-blind on purpose. | 1,064 chunks; 469 equations cut |
| `recursive` | Split on `\n\n`, then `\n`, then space; merge up to 256 tokens with 32 overlap | The default of common RAG libraries. Baseline 2. Respects paragraphs, ignores equations | 1,191; 145 cut (exactly the equations longer than 256 tokens) |
| `section` | One chunk per section node (only that node's own blocks) | "Natural" document structure. Baseline 3 | 91 chunks, median about 2,500 tokens |
| `equation` | Equations are atomic; nearby equations grouped; each group + the sentences before/after; leftover prose chunked recursively | Proposed method 1 | 1,315 (503 equation + 812 prose); 0 cut |
| `hierarchical` | Section > derivation > leaf, linked by `parent_id` | Proposed method 2: whole derivations are retrievable, small leaves are precise | 2,073 (91 section, 449 derivation, 1,533 leaf); 0 cut |

**Equation-aware details (`equation.py`).** `MERGE_GAP_TOKENS = 48` (prose shorter than this between two equations still lets them merge into one group), `GROUP_CAP_TOKENS = 512` (stop merging once a group is this big), `CONTEXT_TOKENS = 64` (the last sentences before and first sentences after the group are included, because they usually introduce the equation and define its symbols). Equations are never split regardless of length.

**Hierarchical details (`hierarchical.py`).** The source has no derivation markers, so derivations are a heuristic: equations whose gap is at most `GAP_MAX = 2` paragraphs form a cluster; the derivation is the cluster plus one paragraph before and after; prose stretches between clusters become their own derivation chunks; anything over `DERIV_CAP = 1000` tokens is split at block boundaries (cut one block earlier if the cut would separate an equation from its introducing paragraph; this fixed a real bug); fragments under `DERIV_MIN = 64` merge into a neighbour. Leaves are the equation-aware chunks computed *inside* each derivation, so every leaf has exactly one parent. `parent_id` is set on leaves and derivations.

### Assumptions
- The hierarchical index stores **all three levels in one index**. This is simple but flatters hit-rate metrics (a section-level parent trivially contains the gold block). See section 8, caveats.
- Overlap is applied only in `fixed` and `recursive`. Section, equation and hierarchical chunks are contiguous spans without overlap.

### How to add a chunker
1. Write `src/chunking/<name>.py` with a function `Document -> list[Chunk]` that builds chunks only through `make_chunk` (this guarantees correct `block_ids`, lines, equation bookkeeping).
2. Register it in `STRATEGIES` in `run.py` and in `ALL` in `retrieval/build.py` and `STRATEGIES` in `evaluation/retrieval_eval.py`.
3. `uv run python -m src.chunking.run` then `uv run python -m src.retrieval.build --model nomic --strategies <name>`.
4. Add a test in `tests/test_chunking.py` (invariants worth checking: every block is covered, no equation is cut if the method promises that, chunk text is a contiguous span).

---

## 6. Stage 4: retrieval (`src/retrieval`)

### `Retriever` (`hybrid.py`)
`Retriever(strategy, dense=None, chunks=None)` loads `chunks_<strategy>.jsonl` and builds the BM25 and symbol indexes. `dense` is a `DenseIndex` you attach (`r.dense = DenseIndex(...)`). Passing `chunks=` lets tests use in-memory chunks.

`retrieve(query, k, mode, dense_mode, weights, top_n, with_parent)` returns `Hit(rank, score, chunk, parent, signals)`.

Modes: `bm25 | dense | symbol | hybrid | rrf`. Dense modes: `full | short | funnel` (default `funnel`).

### BM25 (`bm25.py`)
Own implementation, `k1 = 1.5`, `b = 0.75`. A LaTeX-aware tokenizer keeps commands (`\Box`, `\partial`) as tokens and drops common English stopwords. The same `BM25` class is reused by the symbol index.

### Symbol signal (`symbol.py`)
Chunk -> normalized symbol tokens (from the normalizer, using `index_text`), then BM25 over those tokens. BM25 gives IDF weighting (a rare structure like `PARTIAL[mu](PHI)` counts far more than a bare `m`) and length normalization. A query with no recognised math gives an all-zero score array.

### Dense (`dense.py`)
- Model: `nomic-ai/nomic-embed-text-v1.5` (768-d, trained for Matryoshka truncation). Requires `trust_remote_code=True`. Prefixes `search_document: ` and `search_query: `. `max_seq_length = 1024`. A second preset `mxbai` (1024-d, standard BERT, no remote code, 512-token limit) is defined but not evaluated.
- Raw embeddings are cached once at full size as `data/processed/index/<strategy>__nomic768.npy` with a `.json` fingerprint of the texts and model; the cache invalidates itself if chunks change.
- **Truncation recipe (nomic):** layer-norm the full vector, keep the first *d* dimensions, L2-normalize. Inner product then equals cosine. Short = 128-d. Both live in `faiss.IndexFlatIP` (exact search).
- Modes: `full` = exact over 768-d; `short` = exact over 128-d; `funnel` = top 100 by the 128-d index, then those 100 rescored with 768-d vectors; chunks outside the shortlist get `-inf`.
- The funnel is **sequential** (low-dim then high-dim). What is parallel is the three *signals* in the hybrid.
- Measured speed: on our 2,073 chunks the search takes tens of microseconds either way, far below the query-embedding time (about a second). On synthetic corpora the funnel is about 5 to 6x faster than exact full search at 100k to 1M vectors (brute-force search). So its advantage is scale and memory, not our benchmark.
- Platform gotchas are in section 10.

### Hybrid fusion (`hybrid.py`)
1. Each signal scores every chunk.
2. Take each signal's top `top_n = 50` and form the union of candidates.
3. **Min-max normalize each signal over that candidate union** (raw BM25, cosine and symbol scores are on incompatible scales).
4. `score = a*dense + b*bm25 + g*symbol`, default `weights = (1, 1, 1)` (order is dense, bm25, symbol). Non-candidates get `-inf`.
5. `rrf`: reciprocal rank fusion (`k = 60`) over the same rankings, as a weight-free baseline. A symbol signal with no matches abstains.

### Assumptions
- Equal weights are the default because tuning on a small dev set did not generalise (section 8).
- The query type is **not** known at runtime; the only routing signal tried was "does the query contain math".

### How to add a signal or an embedder
- **Embedder:** implement the `Embedder` protocol (`key`, `layer_norm`, `embed(texts, kind)`) or add a `PRESETS` entry, then run `src.retrieval.build --model <key>`. The cache key includes the embedder key, so models do not collide. If a model is not MRL-trained, short/funnel results are meaningless; use `full` only.
- **Signal:** give it a `scores(query) -> np.ndarray` of length `n_chunks` (higher is better, `-inf` or 0 for no match), add it to `signal_scores` and to the `names` / weights logic in `retrieve`. Extend the evaluation `METHODS` dict.

---

## 7. Stage 5: benchmark and evaluation

### Benchmark (`scripts/build_benchmark.py` -> `data/benchmark/questions.json`)
- 40 questions in four types: **concept** 14, **equation** 12, **notation** 7, **context** 7. Equation queries are raw LaTeX; notation queries mix prose and math; context queries are "why/how" questions answered by prose.
- Each question has `query`, `answer_hint`, `gold_items`, `gold_blocks`, `gold_sections`, `similar_equations` (other equations with Jaccard >= 0.6 on symbols, for awareness), `confidence`, `status` (currently `draft`).
- **Gold anchors are source anchors**: `gold_items` are written as equation IDs (`6:eq3`), block IDs (`6:b17`) or a line range (`L3127-3134`); `resolve()` maps them to `gold_blocks`. Never reference chunk IDs in gold labels.
- Rebuild with `PYTHONPATH=. uv run python scripts/build_benchmark.py`. The `Q` list in that file is the source of truth; `questions.json` is generated from it.
- The labels were checked by script and by one reader, not by a second person. Four labels were corrected in review (Q06, Q08, Q12, Q19), and six earlier. Treat `confidence: medium` items as less certain.
- Some questions legitimately have several answer locations (for example Q15). Include all acceptable blocks in `gold_items`.

### Metrics
- **Hit:** chunk.block_ids intersects gold blocks.
- **Recall@k** (k = 1, 3, 5): fraction of questions with a hit in the top k. **MRR:** mean of 1/rank of the first hit within the top 10.
- **Token-budget hit:** take ranked chunks, charge each its token count, truncate the last one at the budget; within a chunk, blocks are laid out in order (scaled to the chunk's token count); a gold block is *seen* if it starts inside the budget. Reported at 500, 1000, 2000 tokens, plus gold coverage (fraction of gold blocks seen).
  - Why: plain Recall@k rewards huge chunks. A 3,000-token section chunk contains the answer by size alone. The budget view puts all chunkers on the same amount of returned text.
- **Dev/test split:** `split_dev_test` alternates within each question type (deterministic, type-balanced): 21 dev, 19 test.

### Scripts
| Script | Output |
|---|---|
| `retrieval_eval.py` | 5 chunkers x 8 methods (`bm25, symbol, dense-full/short/funnel, hybrid-full, hybrid-funnel, rrf-funnel`): R@1/3/5, MRR, mean top-1 token size, by type and split. `results_baseline.json` |
| `budget_eval.py` | hit and coverage at 500/1000/2000 tokens. `results_budget.json` |
| `tune.py` | grid over weights {0, 0.5, 1, 2}^3 on dev (objective: hit@1000 + MRR; ties go to the weights closest to equal), compared on test with the default and with math/prose routing. `results_tuned.json` |
| `ablation.py` | signal subsets and a failure list. `results_ablation.json` |

`retrieval_eval.cached(retriever)` memoizes signal scores per (query, dense mode) so a grid does not re-embed queries. Reuse it in new evaluation code.

### Results (as of the last run; see `progress.md` for the summary)
- Hit @ 1000 tokens, hybrid equal weights: recursive 0.88, fixed 0.82, equation 0.80, hierarchical 0.75, section 0.35.
- Hybrid beats any single signal. Symbol matching lifts equation questions (hit@1000 0.53 -> 0.78) and adds nothing on concept questions. Dense is weakest, near useless on equation questions.
- Tuned weights did not beat equal weights on the test half.
- Funnel matches full dense and is slightly worse than full inside hybrid.
- Q37 is found by no chunker.

### Caveats a reader must keep in mind
- 40 questions: one question is 2.5 points. Differences below about 5 points are noise.
- Hierarchical's index mixes section, derivation and leaf chunks, so its hit rate is inflated relative to its precision; its `top1_tokens` is higher than the leaf-only chunkers.
- The budget metric truncates a long chunk at the budget. A reader given a whole chunk would see more; the metric deliberately charges for size.
- The benchmark was written by the same people who designed the system. Phrasing may match the notes' wording more than real user queries do.

---

## 8. Stage 6: grounded answering (`src/rag`)

### Aim
Answer a question from retrieved passages only, with citations that can be checked mechanically, and refuse when the notes do not contain the answer. This is Phase 6A; the derivation checker (6B) is not built.

### Flow (`answer.py`)
1. `Retriever.retrieve(question, k=12, mode="hybrid", dense_mode="full")`.
2. `pack_sources(hits, budget=1500)`: take ranked chunks until the token budget is reached (skip a chunk that would overshoot by more than 50%). Sources are numbered `[1]..[n]`.
3. `SYSTEM` prompt: use only the numbered sources; every sentence ends with a citation like `[1]` or `[2][3]`; keep LaTeX as written; if unsupported, reply exactly `Not found in the notes.`; at most 6 sentences.
4. `LLM.chat(system, user)` -> answer text.
5. `verify(text, n_sources)` checks (a) every cited index exists, (b) every sentence has a citation, (c) the refusal string is accepted as grounded. `Answer.grounded` is true only if these pass.
6. `render(answer)` prints the answer, the sources with `Final.tex` line ranges, cited ones marked `*`, and the status.

### LLM client (`llm.py`)
`OllamaLLM(model="granite4.2:8b", temperature=0, num_ctx=8192)` talks to a local Ollama server at `localhost:11434` using only the standard library. `FakeLLM(replies)` returns canned replies and records prompts; use it in tests. To swap providers, implement `LLM.chat(system, user) -> str`. No API key is configured in this project.

### Assumptions and limits
- The mechanical check proves a citation *exists*, not that it *supports* the sentence. A fluent model can cite the wrong source. Evaluating support is open work.
- Chunker for answering defaults to `recursive` (best hit-rate under a token budget).
- Only three live questions were tried (a derivation, a theorem statement, an off-topic question that was correctly refused). It has not been run on the 40 benchmark questions.

---

## 9. Setup and commands

Python 3.12, managed by [uv](https://docs.astral.sh/uv/). Key dependencies: `numpy`, `faiss-cpu`, `sentence-transformers<6`, `transformers<5`, `einops`, `tiktoken`; dev: `pytest`, `openpyxl`.

```bash
uv sync
uv run python scripts/fetch_data.py                      # data/raw/date2207
uv run python -m src.parser.parser --report              # data/processed/date_qft.json
uv run python -m src.parser.normalizer                   # fills Block.symbols
uv run python -m src.chunking.run                        # chunks_<strategy>.jsonl
uv run python -m src.retrieval.build --model nomic       # embeddings, cached
PYTHONPATH=. uv run python scripts/build_benchmark.py    # questions.json

uv run python -m src.evaluation.retrieval_eval
uv run python -m src.evaluation.budget_eval
uv run python -m src.evaluation.tune
uv run python -m src.evaluation.ablation

uv run python -m src.rag.answer "your question"          # needs `ollama serve` and the model pulled
uv run pytest                                            # 40 tests
```

Order matters: parser, then normalizer, then chunking, then embeddings. If you change anything upstream, re-run everything downstream.

---

## 10. Environment gotchas (each one cost time)

- **`transformers` is pinned `<5`** and `sentence-transformers<6`: the nomic remote code breaks on transformers 5 (`get_extended_attention_mask`).
- **`trust_remote_code=True`** for nomic runs code from the model repo. The team chose this knowingly. The alternative, `mxbai`, needs no remote code.
- **macOS OpenMP clash:** `dense.py` sets `KMP_DUPLICATE_LIB_OK=TRUE`, **imports `torch` before `faiss`** (the reverse order segfaults), and runs FAISS with one thread. Keep that import order in any new module that touches both.
- Embeddings run on Apple MPS when available. First query in a process is slow (model load/warm-up).
- `pytest` needs `pythonpath = ["."]` (set in `pyproject.toml`).
- `scripts/` run with `PYTHONPATH=.` so they can import `src` and each other.
- Do not commit `data/raw` or `data/processed`: they contain Date's text.

---

## 11. Decision log (why things are the way they are)

| Decision | Reason |
|---|---|
| Single shared IR (`Document`) | Fair comparison: chunkers differ only in splitting. |
| Gold = source block IDs | Chunker-independent labels; a hit is defined by overlap, not by identity. |
| `header_path` only at index time, same for all | No chunker gets extra context. |
| Equations atomic in the proposed methods | Splitting an equation destroys its meaning; baselines show how often this happens (145 to 469 cut). |
| Drop figures/TikZ | Not searchable by any signal here. |
| Min-max over a per-signal top-50 union | Signals have incompatible score scales; this keeps fusion simple and tunable. |
| Equal weights default | Tuning overfit 21 dev questions and lost on test. |
| Report token-budget hit besides recall | Section chunks otherwise win by size alone. |
| Two independent axes (Matryoshka embeddings vs hierarchical chunking) | Agreed terminology: "Matryoshka" means embeddings; nested chunks are "hierarchical". |
| FAISS flat indexes | Corpus has about 2,000 chunks; exact search is trivial; no database needed. |
| Local LLM via Ollama | No API key or cost; quality is weaker, especially for LaTeX reasoning. |
| Benchmark labels checked by hand/script, not a team review | Team review was monotonous; I (the agent) cross-checked every question against the parsed text. A second human check is still advisable before publishing results. |

---

## 12. Known weaknesses and open work

**Weaknesses**
- Section chunks exceed the embedding model's useful length (median about 2,500 tokens, limit 1,024 here), so their dense vectors reflect only the start of the section.
- Dense search is weak on equations; the symbol signal does the work there.
- Hierarchical index mixes levels (see caveats).
- Benchmark is small and written by the system's authors.
- Citation verification is structural only.
- The normalizer is a heuristic, not a TeX engine.

**Not built yet**
1. **Derivation checker (6B).** Input: a derivation as LaTeX steps. Output: the first wrong step and why, using retrieved passages as evidence. Needs about 25 examples from the team: steps, correct or not, and for wrong ones the step and reason (sign error, dropped factor, and similar).
2. **Answer evaluation (6C).** Run the 40 questions through `Answerer` and score whether each cited source supports each sentence (a rubric or an LLM judge; calibrate on a few hand-labelled answers).
3. **Streamlit demo (7).** Question box, chunker and method selectors, sources panel, answer with citations.
4. **Report and README.**
5. **Optional:** evaluate `mxbai` to show the dense findings are not specific to nomic; commit the funnel timing script (the scaling measurement in section 6 came from a throwaway script, not in the repo).

---

## 13. Checklist before you change something

- [ ] Does the change keep all five chunkers on the same `Document` and the same `index_text` treatment?
- [ ] If block IDs could change (parser, normalizer, flattening), did you rebuild chunks, embeddings and the benchmark, and re-run the tests?
- [ ] If you add benchmark questions, are the gold labels source block IDs, with every acceptable location listed?
- [ ] If you change weights or fusion, did you tune on dev only and report on test?
- [ ] If you compare chunkers, did you look at the token-budget numbers, not only Recall@k?
- [ ] Did `uv run pytest` pass (40 tests at the time of writing)?
- [ ] Did you keep Date's text out of git?
