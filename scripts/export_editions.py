# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Make the Markdown (.md) and printed (.pdf) editions of every evaluated notebook.

For each notebook ``NN-folder/Name.ipynb`` this writes, next to it,

* ``Name.md``: the text, the code cells (with their c.N indices), the tables
  (from the Markdown representation that ``polymer_nn.nbtools.table`` stores
  with every table), printed output, and every figure as a PNG file in
  ``figures/`` named after its cell (``out-c.N.png``). Display equations are
  written as GitHub ```` ```math ```` blocks so that GitHub renders them.
* ``Name.pdf``: the same content typeset with pandoc and XeLaTeX on A4 paper
  (DejaVu fonts, so that Greek letters and symbols print).

Run after executing the notebooks::

    uv run python scripts/export_editions.py
"""

from __future__ import annotations

import base64
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ["00-theory", "01-data-and-graph", "02-reference-simulations", "03-training", "04-learned-dynamics"]
CR = "© Computational Classnotes · CC BY-NC-SA 4.0"

NAME_IMG = re.compile(r'<img alt="[^"]*" src="(figures/noor-name\.png)" height="22">')
CR_P = re.compile(r"<p align='(?:left|center)' style='color:#666666'><i>(.*?)</i></p>")


def cell_index(src: str) -> str:
    m = re.match(r"#\s*(c\.[\d.]+)", src)
    return m.group(1) if m else "c"


def convert(nb_path: Path) -> tuple[str, str]:
    """Return (GitHub markdown, pandoc markdown) for one notebook."""
    nb = nbformat.read(nb_path, as_version=4)
    figdir = nb_path.parent / "figures"
    figdir.mkdir(exist_ok=True)
    gh, pd = [], []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            text = cell.source
            pd_text = NAME_IMG.sub(r"![](\1){height=0.45cm}", text)
            pd_text = CR_P.sub(r"*\1*", pd_text)
            gh_text = CR_P.sub(r"*\1*", text)
            # display math: $$ ... $$  ->  ```math ... ``` for GitHub
            gh_text = re.sub(r"\$\$\s*\n(.*?)\n\s*\$\$", lambda m: "```math\n" + m.group(1).strip() + "\n```", gh_text, flags=re.S)
            gh.append(gh_text)
            pd.append(pd_text)
            continue
        if cell.cell_type != "code":
            continue
        idx = cell_index(cell.source)
        block = "```python\n" + cell.source + "\n```"
        gh.append(block)
        pd.append(block)
        n_img = 0
        for out in cell.get("outputs", []):
            if out.output_type == "stream":
                txt = out.text.rstrip()
                if txt:
                    s = "```text\n" + txt + "\n```"
                    gh.append(s)
                    pd.append(s)
            elif out.output_type in ("display_data", "execute_result"):
                data = out.get("data", {})
                if "image/png" in data:
                    n_img += 1
                    name = f"out-{idx}" + ("" if n_img == 1 else chr(ord("a") + n_img - 2)) + ".png"
                    (figdir / name).write_bytes(base64.b64decode(data["image/png"]))
                    gh.append(f"![Output of cell {idx}](figures/{name})")
                    pd.append(f"![](figures/{name}){{width=95%}}")
                elif "text/markdown" in data:
                    gh.append(data["text/markdown"])
                    pd.append(data["text/markdown"])
                elif "text/plain" in data:
                    s = "```text\n" + data["text/plain"] + "\n```"
                    gh.append(s)
                    pd.append(s)
            elif out.output_type == "error":
                raise RuntimeError(f"{nb_path.name}: cell {idx} has an error output; run the notebook first")
    note = (f"> **About this Markdown edition.** Made from [`{nb_path.name}`]({nb_path.name}) after every cell was run, by "
            "`scripts/export_editions.py`. Equations are LaTeX; code cells keep their indices (c.N); every figure is in "
            "[`figures/`](figures). A printed edition is "
            f"[`{nb_path.with_suffix('.pdf').name}`]({nb_path.with_suffix('.pdf').name}).")
    first, *rest = gh
    lines = first.split("\n", 1)
    gh_doc = "\n\n".join([lines[0] + "\n\n" + note + ("\n" + lines[1] if len(lines) > 1 else "")] + rest) + "\n"
    return gh_doc, "\n\n".join(pd) + "\n"


HEADER_TEX = r"""
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
\fvset{breaklines=true,breakanywhere=true}
\usepackage{fancyhdr}
\pagestyle{fancy}\fancyhf{}
\fancyfoot[L]{\footnotesize\color{gray} © Computational Classnotes · CC BY-NC-SA 4.0}
\fancyfoot[R]{\footnotesize\thepage}
\renewcommand{\headrulewidth}{0pt}
\setlength{\emergencystretch}{3em}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize}
"""


def to_pdf(pd_md: str, nb_path: Path) -> Path:
    out = nb_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.md"
        src.write_text(pd_md, encoding="utf-8")
        hdr = Path(tmp) / "header.tex"
        hdr.write_text(HEADER_TEX, encoding="utf-8")
        cmd = ["pandoc", str(src), "-o", str(out), "--pdf-engine=xelatex", "-f", "markdown+tex_math_dollars+raw_tex-yaml_metadata_block",
               "--resource-path", str(nb_path.parent), "-H", str(hdr), "--highlight-style=tango",
               "-V", "geometry:a4paper,margin=2cm", "-V", "mainfont=DejaVu Serif", "-V", "sansfont=DejaVu Sans",
               "-V", "monofont=DejaVu Sans Mono", "-V", "mathfont=DejaVu Math TeX Gyre", "-V", "fontsize=10pt",
               "-V", "colorlinks=true", "-V", f"pdfauthor=Dara O Shayda, Noor Hoshino",
               "-M", f"title-meta={nb_path.stem.replace('-', ' ')}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stderr[-3000:])
            raise RuntimeError(f"pandoc failed for {nb_path.name}")
    return out


def main(folders=FOLDERS) -> None:
    for folder in folders:
        for nb_path in sorted((ROOT / folder).glob("*.ipynb")):
            gh, pd = convert(nb_path)
            nb_path.with_suffix(".md").write_text(gh, encoding="utf-8")
            pdf = to_pdf(pd, nb_path)
            print(f"{nb_path.relative_to(ROOT)} -> {nb_path.with_suffix('.md').name}, {pdf.name} ({pdf.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main(sys.argv[1:] or FOLDERS)
