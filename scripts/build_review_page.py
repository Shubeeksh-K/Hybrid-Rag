"""Generate the team's label-review page (HTML for the Artifact tool).

Reads data/benchmark/questions.json + data/processed/date_qft.json and writes
data/processed/review_page.html. The page embeds short excerpts of the source notes
for internal verification only, so the output lives under the gitignored data/processed/.

    uv run python scripts/build_review_page.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from src.parser.schema import Document

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/benchmark/questions.json"
DOC = ROOT / "data/processed/date_qft.json"
OUT = ROOT / "data/processed/review_page.html"
REVIEWERS = ["Shubeeksh", "Helen", "Sanjay", "Shreya"]
TEXT_CAP, EQ_CAP, BEFORE_CAP = 560, 1800, 220

esc = html.escape


def clean_text(t: str, cap: int) -> str:
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    t = re.sub(r"\{\\(?:em|bf|it|sl)\s+([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\(?:underline|textbf|textit|emph)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\(?:eq)?ref\{[^}]*\}", "(ref)", t)
    t = t.replace("``", "\u201c").replace("''", "\u201d")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > cap:
        t = t[:cap].rsplit(" ", 1)[0] + " \u2026"
    return t


def math_source(b) -> str:
    c = re.sub(r"\\label\{[^}]*\}", "", b.content).replace("\\nonumber", "")
    if b.env == "\\[":
        return "\\[" + c + "\\]"
    if b.env == "array":
        return "\\[\\begin{array}" + c + "\\end{array}\\]"
    return f"\\begin{{{b.env}}}{c}\\end{{{b.env}}}"


def main() -> None:
    doc = Document.load(DOC)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    sec = {s.section_id: s for s in doc.iter_sections()}
    where = {}
    for s in doc.iter_sections():
        for i, b in enumerate(s.blocks):
            where[b.block_id] = (s, i)
    eqblock = {b.equation_id: (s, b) for s in doc.iter_sections() for b in s.blocks if b.type == "equation"}

    def loc(s) -> str:
        return f"\u00a7{s.section_id} {s.title}"

    def passage(bid: str) -> str:
        s, i = where[bid]
        b = s.blocks[i]
        meta = (f'<span class="kindtag kind-{b.type}">{b.type}</span>'
                f'<span class="loc">{esc(loc(s))}</span>'
                f'<code class="ref">{esc(b.equation_id or b.block_id)}</code>'
                f'<span class="lines">Final.tex L{b.line_start}\u2013{b.line_end}</span>')
        if b.type == "text":
            body = f'<p class="ptext">{esc(clean_text(b.content, TEXT_CAP))}</p>'
        else:
            before = ""
            if i > 0 and s.blocks[i - 1].type == "text":
                before = f'<p class="before">\u2026 {esc(clean_text(s.blocks[i - 1].content, BEFORE_CAP)[-BEFORE_CAP:])}</p>'
            raw = math_source(b)
            if len(raw) > EQ_CAP:
                body = (before + '<p class="longeq">Long equation, not rendered. Opening of the LaTeX:</p>'
                        f'<pre class="raw">{esc(raw[:520])} \u2026</pre>')
            else:
                body = (before + f'<div class="math">{esc(raw)}</div>'
                        f'<details class="rawtex"><summary>LaTeX source</summary><pre class="raw">{esc(raw)}</pre></details>')
        return f'<div class="passage kind-{b.type}"><div class="pmeta">{meta}</div>{body}</div>'

    cards, data = [], []
    for q in questions:
        qid = q["id"]
        sim = ""
        if q["similar_equations"]:
            items = []
            for e in q["similar_equations"]:
                s, b = eqblock[e]
                raw = math_source(b)
                shown = (f'<div class="math">{esc(raw)}</div>' if len(raw) <= EQ_CAP
                         else f'<pre class="raw">{esc(raw[:300])} \u2026</pre>')
                items.append(
                    f'<label class="sim"><input type="checkbox" class="also" id="a-{qid}-{e}" value="{esc(e)}">'
                    f'<span class="simbody"><span class="simmeta"><span class="loc">{esc(loc(s))}</span>'
                    f'<code class="ref">{esc(e)}</code></span>{shown}</span></label>')
            sim = ('<section class="similar"><h4>Similar equations elsewhere in the notes</h4>'
                   '<p class="hint">Tick any that would also be a correct answer to this question.</p>'
                   + "".join(items) + "</section>")
        psgs = "".join(passage(b) for b in q["gold_blocks"])
        qmath = ""
        if q["type"] == "equation":
            qmath = f'<div class="math qmath">\\[{esc(q["query"])}\\]</div>'
        cards.append(f'''
<article class="card" id="{qid}" data-type="{q["type"]}" data-reviewer="{q["reviewer"]}" data-conf="{q["confidence"]}" data-state="todo">
  <header class="chead">
    <span class="qid">{qid}</span>
    <span class="tag type-{q["type"]}">{q["type"]}</span>
    <span class="tag conf-{q["confidence"]}">{q["confidence"]} confidence</span>
    <span class="assignee">Assigned to <b>{q["reviewer"]}</b></span>
    <span class="statelabel" data-statelabel>To review</span>
  </header>
  <div class="qwrap">
    <div class="qlabel">Question typed into the search box</div>
    <p class="query">{esc(q["query"])}</p>{qmath}
  </div>
  <p class="expect"><span class="lbl">A good answer shows</span>{esc(q["answer_hint"])}</p>
  <section class="passages">
    <h4>Proposed answer: where it is in the notes</h4>
    {psgs}
  </section>
  {sim}
  <form class="review" onsubmit="return false">
    <fieldset class="verdicts">
      <legend>Your verdict</legend>
      <label class="vopt v-ok"><input type="radio" name="v-{qid}" id="v-{qid}-ok" value="ok"><span>Correct as proposed</span></label>
      <label class="vopt v-fix"><input type="radio" name="v-{qid}" id="v-{qid}-fix" value="fix"><span>Needs a different location</span></label>
      <label class="vopt v-bad"><input type="radio" name="v-{qid}" id="v-{qid}-bad" value="bad"><span>Question unclear or unanswerable</span></label>
    </fieldset>
    <label class="notelabel" for="n-{qid}">Note <span class="opt">(if not correct: give the section and a few words from the right passage)</span></label>
    <textarea id="n-{qid}" rows="2" maxlength="1200" placeholder="Optional for Correct. Say where the right passage is, or what is wrong with the question."></textarea>
    <div class="savedline" data-saved aria-live="polite"></div>
  </form>
</article>''')
        data.append({"id": qid, "reviewer": q["reviewer"], "type": q["type"], "conf": q["confidence"]})

    counts = {r: sum(1 for q in questions if q["reviewer"] == r) for r in REVIEWERS}
    chips = "".join(
        f'<button type="button" class="chip" data-reviewer-chip="{r}"><span class="chipname">{r}</span>'
        f'<span class="chipcount" data-chipcount="{r}">0/{counts[r]}</span></button>' for r in REVIEWERS)
    page = TEMPLATE.replace("%%CARDS%%", "".join(cards)).replace("%%CHIPS%%", chips) \
        .replace("%%TOTAL%%", str(len(questions))) \
        .replace("%%DATA%%", json.dumps(data).replace("</", "<\\/"))
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) / 1024:.0f} KB, {len(questions)} questions)")


TEMPLATE = r'''<title>QFT-RAG Benchmark Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:wght@500;600&display=swap">
<style>
/* Layout: sticky progress/filter bar over a single column of review cards, one per benchmark question. */
:root {
  --bg: #edf0f4; --surface: #ffffff; --surface-2: #f5f7fa; --fg: #17202a; --muted: #566270;
  --line: #d3d9e1; --accent: #0b6b6f; --accent-soft: #d9eeee; --accent-fg: #ffffff;
  --ok: #1d7a46; --ok-soft: #dcf1e4; --warn: #94570a; --warn-soft: #fbebc9;
  --bad: #b3261e; --bad-soft: #f8dcd9; --math-bg: #f2f5f8;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #10161c; --surface: #18212a; --surface-2: #1e2933; --fg: #e6ebf0; --muted: #9aa7b4;
  --line: #2b3846; --accent: #58c4c8; --accent-soft: #16373a; --accent-fg: #06282a;
  --ok: #5fd08a; --ok-soft: #173424; --warn: #e8b04a; --warn-soft: #3a2e12;
  --bad: #ff8a80; --bad-soft: #3e1d1a; --math-bg: #141c24; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #10161c; --surface: #18212a; --surface-2: #1e2933; --fg: #e6ebf0; --muted: #9aa7b4;
  --line: #2b3846; --accent: #58c4c8; --accent-soft: #16373a; --accent-fg: #06282a;
  --ok: #5fd08a; --ok-soft: #173424; --warn: #e8b04a; --warn-soft: #3a2e12;
  --bad: #ff8a80; --bad-soft: #3e1d1a; --math-bg: #141c24; color-scheme: dark; }

*, *::before, *::after { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font: 15px/1.5 var(--sans);
  padding-block: 0 56px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 30; background: var(--surface); }
.wrap { max-width: 880px; margin: 0 auto; }
h1, h2, h3, h4, p { margin: 0; }

.top { background: var(--surface); border-bottom: 1px solid var(--line); margin-inline: calc(-1 * max(16px, env(safe-area-inset-left, 0px))); padding: 20px max(16px, env(safe-area-inset-left, 0px)) 0; }
.top-in { max-width: 880px; margin: 0 auto; }
h1 { font: 600 26px/1.2 var(--serif); letter-spacing: -0.01em; text-wrap: balance; }
.sub { color: var(--muted); margin-top: 6px; max-width: 64ch; }
.sub a { color: var(--accent); }
.how { margin: 16px 0 4px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.how > div { background: var(--surface-2); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; min-width: 0; }
.how b { display: block; font-size: 13px; margin-bottom: 2px; }
.how span { color: var(--muted); font-size: 13px; }

.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 20; background: var(--surface); border-bottom: 1px solid var(--line);
  margin-inline: calc(-1 * max(16px, env(safe-area-inset-left, 0px))); padding: 12px max(16px, env(safe-area-inset-left, 0px)) 12px; }
.bar-in { max-width: 880px; margin: 0 auto; display: grid; gap: 10px; }
.prog { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.meter { flex: 1 1 200px; height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; min-width: 120px; }
.meter > i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .25s; }
.pcount { font: 500 13px var(--mono); font-variant-numeric: tabular-nums; }
.tally { display: flex; gap: 10px; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.tally b { color: var(--fg); font-weight: 600; }
.chips { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { font: inherit; font-size: 13px; display: inline-flex; gap: 8px; align-items: center; padding: 4px 10px; border: 1px solid var(--line);
  background: var(--surface); color: var(--fg); border-radius: 999px; cursor: pointer; }
.chip[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--accent-fg); }
.chipcount { font: 500 12px var(--mono); font-variant-numeric: tabular-nums; opacity: .85; }
.filters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.filters label { font-size: 12px; color: var(--muted); display: inline-flex; gap: 6px; align-items: center; }
select { font: inherit; font-size: 13px; color: var(--fg); background: var(--surface-2); border: 1px solid var(--line); border-radius: 6px; padding: 4px 8px; }
.showing { margin-left: auto; font-size: 12px; color: var(--muted); }

.notice { margin: 16px 0 0; padding: 10px 12px; border-radius: 6px; background: var(--warn-soft); color: var(--warn); font-size: 14px; border: 1px solid var(--line); }
.list { display: grid; gap: 16px; margin-top: 16px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px; display: grid; gap: 12px; min-width: 0; scroll-margin-top: 150px; }
.card[data-state="ok"] { border-color: var(--ok); }
.card[data-state="fix"] { border-color: var(--warn); }
.card[data-state="bad"] { border-color: var(--bad); }
.chead { display: flex; flex-wrap: wrap; gap: 8px 10px; align-items: center; }
.qid { font: 500 14px var(--mono); background: var(--fg); color: var(--bg); padding: 1px 8px; border-radius: 4px; }
.tag { font-size: 12px; padding: 1px 8px; border-radius: 999px; background: var(--surface-2); border: 1px solid var(--line); color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.conf-medium { background: var(--warn-soft); color: var(--warn); border-color: transparent; }
.assignee { font-size: 13px; color: var(--muted); margin-left: auto; }
.assignee b { color: var(--fg); font-weight: 600; }
.statelabel { font-size: 12px; font-weight: 600; padding: 1px 8px; border-radius: 4px; background: var(--surface-2); color: var(--muted); }
.card[data-state="ok"] .statelabel { background: var(--ok-soft); color: var(--ok); }
.card[data-state="fix"] .statelabel { background: var(--warn-soft); color: var(--warn); }
.card[data-state="bad"] .statelabel { background: var(--bad-soft); color: var(--bad); }

.qlabel, .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.query { font: 600 19px/1.4 var(--serif); margin-top: 2px; overflow-wrap: anywhere; text-wrap: pretty; }
.qmath { margin-top: 6px; }
.expect { color: var(--fg); font-size: 14px; display: grid; gap: 2px; }
h4 { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
.hint { font-size: 12px; color: var(--muted); margin-bottom: 6px; }
.passages, .similar { border-top: 1px solid var(--line); padding-top: 12px; min-width: 0; }
.passage { background: var(--surface-2); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 4px; padding: 10px 12px; margin-bottom: 8px; min-width: 0; }
.passage.kind-text { border-left-color: var(--muted); }
.pmeta, .simmeta { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; font-size: 12px; margin-bottom: 6px; }
.kindtag { text-transform: uppercase; letter-spacing: .05em; font-weight: 600; color: var(--accent); }
.kind-text .kindtag { color: var(--muted); }
.loc { color: var(--fg); font-weight: 500; }
.ref { font: 500 12px var(--mono); background: var(--accent-soft); color: var(--accent); padding: 0 6px; border-radius: 3px; }
.lines { color: var(--muted); font-family: var(--mono); font-size: 11px; }
.ptext { font: 400 14.5px/1.55 var(--serif); }
.before { font: 400 13px/1.5 var(--serif); color: var(--muted); font-style: italic; margin-bottom: 6px; }
.math { background: var(--math-bg); border-radius: 4px; padding: 8px 10px; overflow-x: auto; }
.rawtex { margin-top: 6px; font-size: 12px; }
.rawtex summary { cursor: pointer; color: var(--muted); }
pre.raw { margin: 6px 0 0; font: 400 12px/1.5 var(--mono); white-space: pre-wrap; overflow-wrap: anywhere; background: var(--math-bg); padding: 8px 10px; border-radius: 4px; }
.longeq { font-size: 12px; color: var(--muted); }
.sim { display: flex; gap: 10px; align-items: flex-start; padding: 8px 10px; border: 1px dashed var(--line); border-radius: 6px; margin-bottom: 8px; cursor: pointer; }
.simbody { flex: 1; min-width: 0; }
.sim input { margin-top: 4px; }

.review { border-top: 1px solid var(--line); padding-top: 12px; display: grid; gap: 8px; }
fieldset.verdicts { border: 0; padding: 0; margin: 0; display: flex; gap: 8px; flex-wrap: wrap; min-width: 0; }
.verdicts legend { font-size: 13px; font-weight: 600; margin-bottom: 6px; padding: 0; }
.vopt { position: relative; display: inline-flex; }
.vopt input { position: absolute; opacity: 0; inset: 0; cursor: pointer; }
.vopt span { padding: 6px 12px; border: 1px solid var(--line); border-radius: 6px; font-size: 13.5px; background: var(--surface); }
.vopt input:checked + span { color: #fff; font-weight: 600; }
.v-ok input:checked + span { background: var(--ok); border-color: var(--ok); }
.v-fix input:checked + span { background: var(--warn); border-color: var(--warn); }
.v-bad input:checked + span { background: var(--bad); border-color: var(--bad); }
:root[data-theme="dark"] .vopt input:checked + span { color: #0b1116; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .vopt input:checked + span { color: #0b1116; } }
.vopt input:focus-visible + span, .chip:focus-visible, select:focus-visible, textarea:focus-visible, .sim input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.notelabel { font-size: 13px; font-weight: 500; }
.opt { font-weight: 400; color: var(--muted); }
textarea { width: 100%; font: inherit; font-size: 14px; color: var(--fg); background: var(--surface-2); border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; resize: vertical; }
.savedline { min-height: 18px; font-size: 12px; color: var(--muted); }
.savedline.err { color: var(--bad); }
.empty { text-align: center; color: var(--muted); padding: 32px 0; }
input:disabled + span, textarea:disabled { opacity: .55; cursor: not-allowed; }
mjx-container { max-width: 100%; }
@media (max-width: 640px) { .how { grid-template-columns: 1fr; } .assignee { margin-left: 0; } h1 { font-size: 22px; } .query { font-size: 17px; } }
@media (prefers-reduced-motion: reduce) { .meter > i { transition: none; } }
</style>

<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['\\[', '\\]']],
    macros: { dsl: ['{\\not{#1}}', 1], Case: ['{\\textstyle\\frac{#1}{#2}}', 2], hspace: ['\\quad', 1], lP: '\\ell_{\\mathrm P}', sgn: '\\mathrm{sgn}(e)', cross: ['{#1}', 1], circled: ['{#1}', 1] } },
  svg: { fontCache: 'global' },
  options: { enableMenu: false },
  startup: { typeset: false }
};
</script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js"></script>

<div class="top"><div class="top-in">
  <h1>QFT-RAG Benchmark Review</h1>
  <p class="sub">We are testing whether our search finds the right passage in Date's QFT lecture notes
    (<a href="https://arxiv.org/abs/2207.02243" target="_blank" rel="noopener">arXiv:2207.02243</a>).
    Each of the %%TOTAL%% questions below has a proposed answer location. The scores in our final results are only
    meaningful if those locations are correct, so each one needs a human check. Your verdicts save automatically.</p>
  <div class="how">
    <div><b>1. Pick your name</b><span>Use the name chips below. Each person has 10 questions. Take a few extra if you have time.</span></div>
    <div><b>2. Read the passage</b><span>Would a student who asked this question be satisfied by the passages shown? Check the surrounding text in the PDF if unsure.</span></div>
    <div><b>3. Give a verdict</b><span>Correct, wrong location (say where in the note), or the question itself is unclear. Medium-confidence questions need the closest look.</span></div>
  </div>
</div></div>

<div class="bar"><div class="bar-in">
  <div class="prog">
    <div class="meter" role="progressbar" aria-label="Questions reviewed" aria-valuemin="0" aria-valuemax="%%TOTAL%%" id="meter"><i id="meterfill"></i></div>
    <span class="pcount" id="pcount">0 / %%TOTAL%% reviewed</span>
    <span class="tally"><span>correct <b id="t-ok">0</b></span><span>change <b id="t-fix">0</b></span><span>unclear <b id="t-bad">0</b></span></span>
  </div>
  <div class="chips" id="chips" role="group" aria-label="Filter by reviewer">%%CHIPS%%</div>
  <div class="filters">
    <label>Type <select id="f-type"><option value="">All</option><option>concept</option><option>equation</option><option>notation</option><option>context</option></select></label>
    <label>Status <select id="f-status"><option value="">All</option><option value="todo">To review</option><option value="ok">Correct</option><option value="fix">Needs change</option><option value="bad">Unclear</option></select></label>
    <label>Confidence <select id="f-conf"><option value="">All</option><option>high</option><option>medium</option></select></label>
    <span class="showing" id="showing"></span>
  </div>
</div></div>

<main class="wrap">
  <div class="notice" id="notice" hidden></div>
  <div class="list" id="list">%%CARDS%%</div>
  <p class="empty" id="empty" hidden>No questions match these filters.</p>
</main>

<script type="application/json" id="data">%%DATA%%</script>
<script>
(function () {
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var DATA = JSON.parse($('#data').textContent);
  var LABEL = { todo: 'To review', ok: 'Correct', fix: 'Needs change', bad: 'Unclear' };
  var S = { reviews: {}, db: null, user: null, myId: null, canWrite: false, names: {}, reviewer: '' };

  function lsGet(k) { try { return localStorage.getItem(k) || ''; } catch (e) { return ''; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }

  /* ---------- filters ---------- */
  function stateOf(id) { var r = S.reviews[id]; return r && r.verdict ? r.verdict : 'todo'; }
  function applyFilters() {
    var type = $('#f-type').value, status = $('#f-status').value, conf = $('#f-conf').value, shown = 0;
    $$('.card').forEach(function (c) {
      var ok = (!S.reviewer || c.dataset.reviewer === S.reviewer) && (!type || c.dataset.type === type) &&
        (!conf || c.dataset.conf === conf) && (!status || c.dataset.state === status);
      c.hidden = !ok; if (ok) shown++;
    });
    $('#showing').textContent = 'Showing ' + shown + ' of ' + DATA.length;
    $('#empty').hidden = shown > 0;
    $$('.chip').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.reviewerChip === S.reviewer)); });
  }
  $$('.chip').forEach(function (b) {
    b.addEventListener('click', function () {
      S.reviewer = S.reviewer === b.dataset.reviewerChip ? '' : b.dataset.reviewerChip;
      lsSet('qftrag-reviewer', S.reviewer); applyFilters();
    });
  });
  ['#f-type', '#f-status', '#f-conf'].forEach(function (s) { $(s).addEventListener('change', applyFilters); });

  /* ---------- painting ---------- */
  function paintProgress() {
    var done = 0, t = { ok: 0, fix: 0, bad: 0 }, per = {};
    DATA.forEach(function (q) {
      var st = stateOf(q.id); per[q.reviewer] = per[q.reviewer] || 0;
      if (st !== 'todo') { done++; t[st]++; per[q.reviewer]++; }
    });
    $('#pcount').textContent = done + ' / ' + DATA.length + ' reviewed';
    $('#meterfill').style.width = (100 * done / DATA.length) + '%';
    $('#meter').setAttribute('aria-valuenow', String(done));
    $('#t-ok').textContent = t.ok; $('#t-fix').textContent = t.fix; $('#t-bad').textContent = t.bad;
    $$('[data-chipcount]').forEach(function (el) {
      var r = el.dataset.chipcount, n = DATA.filter(function (q) { return q.reviewer === r; }).length;
      el.textContent = (per[r] || 0) + '/' + n;
    });
  }
  function nameOf(id) { return S.names[id] || ''; }
  function paintCard(id) {
    var card = document.getElementById(id), r = S.reviews[id] || {};
    card.dataset.state = stateOf(id);
    $('[data-statelabel]', card).textContent = LABEL[card.dataset.state];
    $$('input[type=radio]', card).forEach(function (x) { x.checked = x.value === r.verdict; });
    var note = $('textarea', card);
    if (document.activeElement !== note) note.value = r.note || '';
    var also = r.also || [];
    $$('input.also', card).forEach(function (x) { x.checked = also.indexOf(x.value) >= 0; });
    var line = $('[data-saved]', card);
    if (r.at) {
      var who = r.by === S.myId ? 'you' : (nameOf(r.by) || 'a teammate');
      line.className = 'savedline';
      line.textContent = 'Saved by ' + who + ' at ' + new Date(r.at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } else { line.textContent = ''; }
  }
  function paintAll() { DATA.forEach(function (q) { paintCard(q.id); }); paintProgress(); applyFilters(); }

  /* ---------- saving: one write at a time per question ---------- */
  var Q = {};
  function readForm(id) {
    var card = document.getElementById(id), v = $('input[type=radio]:checked', card);
    return { verdict: v ? v.value : '', note: $('textarea', card).value.trim(),
      also: $$('input.also:checked', card).map(function (x) { return x.value; }) };
  }
  function schedule(id, now) {
    var q = Q[id] = Q[id] || { busy: false, dirty: false, timer: 0 };
    clearTimeout(q.timer);
    q.timer = setTimeout(function () { flush(id); }, now ? 0 : 700);
  }
  async function flush(id) {
    var q = Q[id];
    if (!S.db || !S.canWrite) return;
    if (q.busy) { q.dirty = true; return; }
    q.busy = true;
    var line = $('[data-saved]', document.getElementById(id));
    try {
      do {
        q.dirty = false;
        var d = readForm(id);
        line.className = 'savedline'; line.textContent = 'Saving\u2026';
        await S.db.doc('reviews/' + id).set({ verdict: d.verdict, note: d.note, also: d.also, by: S.myId || '', at: Date.now() });
      } while (q.dirty);
    } catch (e) {
      line.className = 'savedline err';
      line.textContent = e && e.code === 'invalid_argument'
        ? 'Not saved: you need Contributor access to this page. Ask the owner to share it as Contributor.'
        : 'Not saved (' + ((e && e.code) || 'error') + '). Try again.';
    }
    q.busy = false;
  }
  $$('.card').forEach(function (card) {
    card.addEventListener('change', function (ev) {
      var t = ev.target;
      if (t.matches('input[type=radio]')) { card.dataset.state = t.value; paintLocal(card.id); schedule(card.id, true); }
      else if (t.matches('input.also')) schedule(card.id, true);
    });
    $('textarea', card).addEventListener('input', function () { schedule(card.id, false); });
  });
  function paintLocal(id) {
    var v = readForm(id).verdict || 'todo', card = document.getElementById(id);
    card.dataset.state = v; $('[data-statelabel]', card).textContent = LABEL[v];
    S.reviews[id] = Object.assign({}, S.reviews[id], { verdict: v === 'todo' ? '' : v });
    paintProgress();
  }

  /* ---------- notice + controls ---------- */
  function notice(msg) { var n = $('#notice'); n.textContent = msg; n.hidden = !msg; }
  function setEditable(on) {
    S.canWrite = on;
    $$('.card input, .card textarea').forEach(function (el) { el.disabled = !on; });
  }

  /* ---------- connect to the shared store ---------- */
  async function connect() {
    setEditable(false);
    if (!window.claude || !window.claude.use) { notice('Open this page in claude.ai to record verdicts. You can still read the questions.'); return; }
    var timer = setTimeout(function () { notice('Still trying to reach the shared review data. Verdicts may be out of date.'); }, 10000);
    var db = await window.claude.use('db'), user = await window.claude.use('user');
    if (!db) { clearTimeout(timer); notice('Sign in to claude.ai to record verdicts. You can still read the questions.'); return; }
    S.db = db; S.user = user;
    try { S.myId = user && user.id ? await user.id() : null; } catch (e) {}
    var can = user && user.can ? user.can('data.write') : null;
    if (can === false) notice('You have view-only access, so verdicts cannot be saved. Ask Shubeeksh to share this page with you as a Contributor.');
    setEditable(can !== false);
    db.collection('reviews').onSnapshot(async function (snap) {
      clearTimeout(timer);
      if (!snap.metadata.fromCache) { var n = $('#notice'); if (n.textContent.indexOf('Still trying') === 0) notice(''); }
      var next = {}, ids = {};
      snap.docs.forEach(function (d) { var v = d.data(); next[d.id] = v; if (v && v.by) ids[v.by] = 1; });
      S.reviews = next;
      var need = Object.keys(ids).filter(function (i) { return !S.names[i]; });
      if (need.length && user && user.profiles) {
        try { var ps = await user.profiles(need); need.forEach(function (i) { S.names[i] = (ps[i] && ps[i].name) || ''; }); } catch (e) {}
      }
      paintAll();
    }, function (e) {
      clearTimeout(timer);
      notice(e && e.code === 'revoked' ? 'Your access to this page changed. Reload to continue.' : 'Could not load saved verdicts (' + ((e && e.code) || 'error') + ').');
    });
  }

  /* ---------- init ---------- */
  S.reviewer = lsGet('qftrag-reviewer');
  if (['', 'Shubeeksh', 'Helen', 'Sanjay', 'Shreya'].indexOf(S.reviewer) < 0) S.reviewer = '';
  paintAll();
  connect();
  window.addEventListener('load', function () {
    if (window.MathJax && MathJax.startup) MathJax.startup.promise.then(function () { return MathJax.typesetPromise([document.getElementById('list')]); }).catch(function () {});
  });
})();
</script>
'''

if __name__ == "__main__":
    main()
