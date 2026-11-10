"""Build report.pdf (2-3 pages, Times 12 pt, 1.5 line spacing) from outputs/results.json.

Sections required by the brief: Overview, Part 1 Results, Part 2 Results,
Part 3 Results, Conclusion. All numbers are read from outputs/results.json,
which the notebook writes; the prose is in report_text.json.

    python build_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

R = json.loads((C.OUT / "results.json").read_text(encoding="utf-8"))
T = json.loads((ROOT / "report_text.json").read_text(encoding="utf-8"))

FONT, BOLD = "Times-Roman", "Times-Bold"
body = ParagraphStyle("body", fontName=FONT, fontSize=12, leading=18, alignment=TA_JUSTIFY,
                      spaceAfter=4)
h1 = ParagraphStyle("h1", fontName=BOLD, fontSize=13, leading=18, spaceBefore=6, spaceAfter=2,
                    keepWithNext=1)
title = ParagraphStyle("title", fontName=BOLD, fontSize=14, leading=18, alignment=1)
sub = ParagraphStyle("sub", fontName=FONT, fontSize=12, leading=18, alignment=1)
cell = ParagraphStyle("cell", fontName=FONT, fontSize=10, leading=12)
cellb = ParagraphStyle("cellb", fontName=BOLD, fontSize=10, leading=12)


def table(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="CENTER", repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t


def f(x, d=1):
    return f"{float(x):.{d}f}"


def build():
    story = [Paragraph(f"{C.COURSE} — Assignment 2: Neural NLP Pipeline on BBC Urdu", title),
             Paragraph(f"{C.STUDENT_NAME} · {C.STUDENT_ID} · Section {C.SECTION} · "
                       f"Repository: {escape(C.REPO_URL)}", sub), Spacer(1, 4)]

    story += [Paragraph("1. Overview", h1), Paragraph(T["overview"], body)]

    # ---------------- Part 1
    story += [Paragraph("2. Part 1 Results — Word Embeddings", h1), Paragraph(T["part1_a"], body)]
    rows = [["Condition", "MRR (20 pairs)", "median rank", "rank 1", "in top-10"]]
    for r in R["mrr"]:
        rows.append([r["condition"], f(r["MRR"], 3), r["median rank"], r["related word ranked 1st"], r["in top-10"]])
    story += [table(rows, [5.0, 3.0, 2.6, 2.0, 2.2]), Spacer(1, 4), Paragraph(T["part1_b"], body)]

    # ---------------- Part 2
    story += [Paragraph("3. Part 2 Results — POS Tagging and NER", h1), Paragraph(T["part2_a"], body)]
    rows = [["Model (test set)", "POS macro-F1", "NER entity F1"]]
    S = {r["model"]: r for r in R["seq_summary"]}
    pairs = [("Frozen W2V embeddings", "POS frozen", "NER frozen"),
             ("Fine-tuned W2V embeddings (full model)", "POS fine-tuned", "NER fine-tuned (CRF)"),
             ("A1: unidirectional LSTM", "POS A1 unidirectional", "NER A1 unidirectional"),
             ("A2: no dropout", "POS A2 no dropout", "NER A2 no dropout"),
             ("A3: random embedding initialisation", "POS A3 random init", "NER A3 random init"),
             ("A4: softmax instead of CRF", None, "NER A4 softmax (no CRF)")]
    for lab, p, n in pairs:
        rows.append([lab, S[p]["test F1 mean ± std"] if p else "—", S[n]["test F1 mean ± std"]])
    story += [table(rows, [7.2, 3.6, 3.6]),
              Paragraph("<i>Mean ± std over seeds 42, 7, 123 (validation-F1 early stopping).</i>", cell),
              Spacer(1, 4), Paragraph(T["part2_b"], body)]

    # ---------------- Part 3
    story += [Paragraph("4. Part 3 Results — Transformer Topic Classifier", h1), Paragraph(T["part3_a"], body)]
    rows = [["Model", "test acc. (seed 42)", "macro-F1 (seed 42)", "acc. mean ± std", "best epoch / seed", "s / epoch"]]
    for r in R["bilstm_vs_transformer"]:
        rows.append([r["model"], f(r["test acc (seed 42)"]), f(r["test macro-F1 (seed 42)"]),
                     r["test acc mean ± std"], ", ".join(map(str, r["best val epoch per seed"])),
                     f(r["seconds / epoch (mean)"], 2)])
    story += [table(rows, [2.6, 2.7, 2.8, 2.9, 2.8, 1.8]), Spacer(1, 4), Paragraph(T["part3_b"], body)]

    story += [Paragraph("5. Conclusion", h1), Paragraph(T["conclusion"], body)]

    doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=A4, leftMargin=2.0 * cm,
                            rightMargin=2.0 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="DS5007 Assignment 2 Report", author=C.STUDENT_NAME)
    doc.build(story)


if __name__ == "__main__":
    build()
    from pypdf import PdfReader  # noqa: E402
    print("pages:", len(PdfReader(str(ROOT / "report.pdf")).pages))
