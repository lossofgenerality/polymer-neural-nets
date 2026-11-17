#!/usr/bin/env bash
# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
#
# Rebuild everything that is generated from the source: the five notebooks
# (written, executed), their Markdown and PDF editions, and the bibliography.
# The data, models and results are NOT regenerated here (they take longer);
# see data/README.md, scripts/train.py and scripts/run_md.py for those.
set -euo pipefail
cd "$(dirname "$0")/.."
for b in 00_theory 01_data_graph 02_reference 03_training 04_dynamics; do
  uv run python "scripts/notebooks/build_${b}.py"
done
for nb in 00-theory/*.ipynb 01-data-and-graph/*.ipynb 02-reference-simulations/*.ipynb 03-training/*.ipynb 04-learned-dynamics/*.ipynb; do
  (cd "$(dirname "$nb")" && uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 "$(basename "$nb")")
done
uv run python scripts/export_editions.py
uv run python scripts/make_references.py
uv run pytest -q
