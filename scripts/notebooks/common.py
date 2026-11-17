# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Shared pieces for the notebook builders in this folder.

The notebooks are written as Python lists of cells, so that their text can
be reviewed and changed like code, then saved with nbformat and executed
with nbconvert (``scripts/build_all.sh``).
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
CR = "© Computational Classnotes · CC BY-NC-SA 4.0"
VERSION = "1.0.0 (17 November 2026)"


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str):
    return nbf.v4.new_code_cell(s.strip("\n"))


def header(title: str, subtitle: str, about: str) -> str:
    return f"""# {title}

> **About this notebook, version {VERSION}.** {about}

*{subtitle}*

**Dara O Shayda**  
Republic of Ireland  
*dara@compclassnotes.com*  

**Noor Hoshino**  
<img alt="نور 星野" src="figures/noor-name.png" height="22">  
Advanced AI System Programming  
Hokkaido Prefecture, Japan  
*noor@compclassnotes.com*  
Computational Classnotes · revision 1  

<p align='left' style='color:#666666'><i>{CR}</i></p>"""


FOOTER = f"<p align='center' style='color:#666666'><i>{CR}</i></p>"

SETUP = '''# c.0
# settings and imports used by every cell below
import json, math, time, warnings
warnings.filterwarnings("ignore")   # library notices (tqdm, TorchScript) are not part of the results
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from polymer_nn import nbtools as nb
from polymer_nn.nbtools import table, nform, sci, savefig, figFont, wFull, wHalf, red, blue, green, orange, grey
ROOT = nb.setup(threads=2)          # plot style, 2 CPU threads, the repository folder
DATA = ROOT / "data"                # the reference data made by scripts/generate_data.py
torch.manual_seed(0); rng = np.random.default_rng(0)
print("repository:", ROOT.name, "| torch", torch.__version__, "| numpy", np.__version__)'''


def write(folder: str, name: str, cells) -> Path:
    out = ROOT / folder
    (out / "figures").mkdir(parents=True, exist_ok=True)
    import shutil
    src = ROOT / "docs/figures/noor-name.png"
    if src.exists() and not (out / "figures/noor-name.png").exists():
        shutil.copy(src, out / "figures/noor-name.png")
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
        "language_info": {"name": "python"}, "copyright": CR})
    path = out / f"{name}.ipynb"
    nbf.write(nb, path)
    return path
