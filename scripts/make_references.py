# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
# Authors: Dara, Hoshino
"""Render the bibliography in ``polymer_nn.references`` to Markdown, BibTeX and PDF.

    uv run python scripts/make_references.py            # writes into docs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape

from polymer_nn import references as R


def write_pdf(path: Path) -> None:
    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, spaceAfter=6)
    sub = ParagraphStyle("s", parent=ss["Normal"], alignment=TA_CENTER, fontSize=10, textColor=colors.HexColor("#444444"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13, spaceBefore=12,
                        textColor=colors.HexColor("#1f3a5f"))
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=12.5)
    note = ParagraphStyle("n", parent=body, fontSize=8.5, textColor=colors.HexColor("#555555"))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(2 * cm, 1.2 * cm, f"{R.COPYRIGHT}  ·  Authors: {', '.join(R.AUTHORS)}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()

    story = [
        Paragraph(escape(R.TITLE), title),
        Paragraph(f"Authors: {escape(', '.join(R.AUTHORS))}", sub),
        Paragraph(f'{escape(R.COPYRIGHT)} &nbsp; <link href="{R.LICENSE_URL}" color="blue">{R.LICENSE_URL}</link>', sub),
        Spacer(1, 12),
        Paragraph("Sources for the graph neural networks, equivariant force fields, coarse-graining, generative "
                  "models, polymer physics, integrators and open-source software used in this project.", body),
    ]
    n = 0
    for topic in R.TOPICS:
        refs = [r for r in R.REFERENCES if r.topic == topic]
        if not refs:
            continue
        story.append(Paragraph(escape(topic), h2))
        items = []
        for r in refs:
            n += 1
            url = R.link(r)
            txt = escape(R.plain(r))
            if url:
                txt += f' <link href="{escape(url)}" color="blue">{escape(url)}</link>'
            items.append(ListItem([Paragraph(txt, body),
                                   Paragraph(f"<i>Used for:</i> {escape(r.used_for)} &nbsp;[{r.key}]", note)],
                                  value=n))
        story.append(ListFlowable(items, bulletType="1", start=n - len(refs) + 1, bulletFontSize=9.5,
                                  leftIndent=18))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm, title=R.TITLE, author=", ".join(R.AUTHORS),
                            subject=R.COPYRIGHT, creator="Computational Classnotes")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    out = p.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    (out / "REFERENCES.md").write_text(R.to_markdown(), encoding="utf-8")
    (out / "references.bib").write_text(R.to_bibtex(), encoding="utf-8")
    write_pdf(out / "REFERENCES.pdf")
    print(f"wrote {len(R.REFERENCES)} references to {out}/REFERENCES.md, references.bib, REFERENCES.pdf")


if __name__ == "__main__":
    main()
