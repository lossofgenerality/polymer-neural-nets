# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Small helpers shared by the notebooks: figure style, tables, number formats.

Every notebook of this repository starts with cell ``c.0``, which calls
:func:`setup`. Keeping these helpers in one module means that all notebooks
print tables and figures in the same way, and that the Markdown and PDF
editions (made by ``scripts/export_editions.py``) can find the tables: every
table is displayed both as HTML (for Jupyter and GitHub) and as Markdown (for
the Markdown and PDF editions).

The style follows the Computational Classnotes notebooks *S Shape Functions
Theory* and *Feedforward Neural Networks*: font size 14 in figures, 10 in
tables, a full figure width of 440 points and a half width of 215 points, so
that every figure also prints well on A4 paper.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COPYRIGHT = "© Computational Classnotes · CC BY-NC-SA 4.0"

figFont = 14  # font size in figures and diagrams
tabFont = 10  # font size in tables
wFull = 440 / 72  # full figure width in inches (440 pt); A4 text width is ~487 pt
wHalf = 215 / 72  # width of each panel in a two-column figure (215 pt)

red, blue, green, orange, grey = (0.85, 0.2, 0.1), (0.1, 0.35, 0.85), (0.15, 0.6, 0.3), (0.9, 0.45, 0.1), (0.45, 0.45, 0.45)


def repo_root(start: Path | None = None) -> Path:
    """The folder that holds ``pyproject.toml``, searched upwards from ``start``.

    Notebooks live one folder below the root (``00-theory/`` and so on), so a
    notebook finds the data in ``<root>/data`` wherever it is opened from.
    """
    p = (start or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "pyproject.toml").exists():
            return q
    raise FileNotFoundError("pyproject.toml not found above " + str(p))


def setup(threads: int = 2) -> Path:
    """Set the plotting style, the number of CPU threads for PyTorch and return the repo root."""
    import torch

    torch.set_num_threads(threads)
    plt.rcParams.update({
        "font.size": figFont, "axes.titlesize": figFont, "axes.labelsize": figFont,
        "legend.fontsize": figFont - 3, "legend.frameon": False, "xtick.labelsize": figFont - 2,
        "ytick.labelsize": figFont - 2, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "cm"})
    try:
        return repo_root()
    except FileNotFoundError:  # a notebook run outside the repository folder
        return Path(__file__).resolve().parents[2]


def nform(v, d: int = 4) -> str:
    """``v`` with ``d`` significant digits (like Wolfram's NumberForm[v, d])."""
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", " ")
    if isinstance(v, str):
        return v
    v = float(v)
    if v == 0 or not math.isfinite(v):
        return str(v)
    if 1e-3 <= abs(v) < 1e5:
        s = f"{v:.{d}g}"
        return s
    return sci(v, d - 1)


def sci(v, d: int = 2) -> str:
    """Scientific notation such as 2.3 × 10⁻⁵ (like ScientificForm[v, d])."""
    v = float(v)
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10**e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.{d}g} × 10{sup}"


def table(rows, label: str) -> None:
    """Show a framed table with a grey header row and a bold label on top.

    ``rows[0]`` is the header. The table is displayed with two
    representations: HTML for Jupyter and GitHub, and Markdown, which the
    Markdown and PDF editions use.
    """
    from IPython.display import display

    rows = [[c if isinstance(c, str) else nform(c) for c in r] for r in rows]
    cell = f"border:1px solid #999;padding:3px 8px;text-align:left;font-size:{tabFont}pt;vertical-align:middle"
    html = [f"<p style='font-weight:bold;font-size:{tabFont + 1}pt;margin-bottom:2px'>{label}</p>",
            "<table style='border-collapse:collapse'>"]
    for i, row in enumerate(rows):
        bg = "background:#ebebeb;" if i == 0 else ""
        html.append("<tr>" + "".join(f"<td style='{bg}{cell}'>{v}</td>" for v in row) + "</tr>")
    html.append("</table>")
    esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
    md = [f"**{label}**", "", "| " + " | ".join(esc(c) for c in rows[0]) + " |",
          "|" + "---|" * len(rows[0])]
    md += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    display({"text/html": "".join(html), "text/markdown": "\n".join(md)}, raw=True)


def stamp(fig) -> None:
    """Put the copyright line in the bottom-right corner of a figure."""
    fig.text(0.995, 0.003, COPYRIGHT, ha="right", va="bottom", fontsize=7, color="0.45")


def savefig(fig, name: str, folder: str | Path = "figures") -> Path:
    """Stamp, save as PNG into the notebook's ``figures/`` folder, and return the path."""
    stamp(fig)
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


def logticks(ax, xticks, yticks=None) -> None:
    """Plain numbers (not 10^k) as tick labels on log-log axes."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter

    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    if yticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
