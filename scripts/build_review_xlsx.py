"""Build the simple team review spreadsheet: data/processed/benchmark_review.xlsx.

One sheet, one row per benchmark question. Contains short excerpts of the source notes,
so it lives under the gitignored data/processed/ (share with the team only).

    PYTHONPATH=. uv run python scripts/build_review_xlsx.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from scripts.build_review_page import TEXT_CAP, clean_text
from src.parser.schema import Document

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/benchmark/questions.json"
DOC = ROOT / "data/processed/date_qft.json"
OUT = ROOT / "data/processed/benchmark_review.xlsx"
EQ_CAP = 700
FONT = "Arial"
HEAD = PatternFill("solid", fgColor="1F3A4D")
YELLOW = PatternFill("solid", fgColor="FFF4C2")


def latex_view(content: str) -> str:
    c = re.sub(r"\\label\{[^}]*\}", "", content).replace("\\nonumber", "")
    c = re.sub(r"[ \t]+", " ", c)
    c = re.sub(r"\n\s*", "\n", c).strip()
    return c if len(c) <= EQ_CAP else c[:EQ_CAP].rsplit(" ", 1)[0] + " ..."


def main() -> None:
    doc = Document.load(DOC)
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    where = {b.block_id: (s, b) for s in doc.iter_sections() for b in s.blocks}

    wb = Workbook()
    ws = wb.active
    ws.title = "Review"
    cols = [("ID", 6), ("Question", 44), ("Proposed answer (section)", 28),
            ("Proposed answer (text from the notes)", 80),
            ("Is it right?\n(pick one)", 18), ("If not: where is the right passage?  /  notes", 40),
            ("Your name", 12)]
    ws.append([c for c, _ in cols])
    for i, (_, w) in enumerate(cols, 1):
        cell = ws.cell(row=1, column=i)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF" if i < 5 else "5A4300")
        cell.fill = HEAD if i < 5 else YELLOW
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[cell.column_letter].width = w
    ws.row_dimensions[1].height = 34

    for r, q in enumerate(qs, 2):
        secs, parts = [], []
        for bid in q["gold_blocks"]:
            s, b = where[bid]
            name = f"\u00a7{s.section_id} {s.title}"
            if name not in secs:
                secs.append(name)
            body = clean_text(b.content, TEXT_CAP) if b.type == "text" else "LaTeX: " + latex_view(b.content)
            parts.append(body)
        row = [q["id"], q["query"], "\n".join(secs), "\n\n".join(parts), None, None, None]
        for c, v in enumerate(row, 1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.font = Font(name=FONT, size=10, bold=(c == 1))
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if c >= 5:
                cell.fill = YELLOW
        lines = sum(max(1, math.ceil(len(l) / 85)) for l in row[3].split("\n"))
        ws.row_dimensions[r].height = min(409, max(32, lines * 13 + 6))

    dv = DataValidation(type="list", formula1='"Yes,No,Question unclear"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"E2:E{len(qs) + 1}")
    ws.freeze_panes = "C2"
    wb.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
