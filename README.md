# Polymer Neural Nets · Computational Classnotes

Graph neural networks that learn the forces of a polymer chain and then drive its simulation. The repository starts from one neuron and ends with a learned force field that conserves energy, keeps the bonds and angles of the chain, and makes chains of lengths it has never seen swell exactly as real self-avoiding chains do. All the data are made here, with OpenMM, from a physical model whose exact forces are known, so every learned number can be checked. Five notebooks explain every step: the mathematics, the data and the graph, the reference simulations, the training, and the learned dynamics. Each one states its results and checks every formula with code you can run.

Dara O Shayda · Noor Hoshino

Version **1.0.0** (17 November 2026) · Python 3.14 · free and open-source software only: NumPy, SciPy, SymPy, Matplotlib, PyTorch, PyTorch Geometric, OpenMM.

## Contents

### [00 · Graph Networks for Polymers: The Theory](00-theory/)

*From one neuron to an energy-conserving graph network for a polymer chain.*

- the neuron, the layer as one matrix product, the network, the loss, backpropagation and Adam, repeated from [Feedforward Neural Networks](https://github.com/lossofgenerality/neural-networks-things) and checked against PyTorch; a small network learns the bond energy of the chain;
- a polymer as a graph; message passing, and why it does not care how the beads are numbered (Theorem 3);
- symmetry: a network that sees only distances gives an energy that does not change when the chain is moved or turned, and forces that turn with it (Theorems 4–5);
- the two networks of the repository, SchNet style and EGNN style, with the radial basis and the smooth cutoff that keeps the energy continuous (Theorems 6–7);
- why forces computed as F = −∇E conserve energy, and a direct force predictor that heats a system up (Theorem 8);
- force matching and the theorem behind coarse-graining (Theorem 9); velocity Verlet and BAOAB with their error laws (Theorem 10);
- the scaling laws of polymers: the exact R_g of an ideal chain derived with SymPy, Flory's ν = 3/5, and why pivot Monte Carlo is correct (Theorem 11).

### [01 · The Data and the Graph](01-data-and-graph/)

*Where every number comes from, how it is stored, what goes into the network and what comes out.*

- the data are made by our own OpenMM simulations, and Table 1 lists the command that made each file;
- every array of every file, with shape, type, unit and meaning; the stored forces checked against OpenMM; the correlation between frames;
- the complete graph of a real 32-bead frame, drawn, and its adjacency matrix; the exact tensors the network receives, for one chain and for a batch;
- the full computational graph of the network with the symbol and the shape of every tensor;
- **symbolics on the graph:** the network written out with SymPy on a three-bead chain, its forces in closed form, and the theorem that an invariant network gives forces that are sums of pair forces along the edges, checked on the trained model;
- the outputs on a test chain: total energy, energy of every bead, force on every bead, against OpenMM.

### [02 · Reference Simulations and Physics Checks](02-reference-simulations/)

*The yardstick: the reference chains must pass the checks themselves.*

- R_g ∼ N^ν for ideal chains (ν = 1/2, against the exact formula) and Kremer–Grest chains, from independent replicas, with pivot Monte Carlo as an independent check;
- bond-length and bond-angle distributions, with the bond lengths predicted from the bond potential;
- energy conservation with the exact forces, and its Δt² law.

### [03 · Training the Force Fields](03-training/)

*Force matching on OpenMM data.*

- the loss, and one gradient step taken apart: double backpropagation checked against finite differences;
- a live training run; the learning curves of the two full models (about 1% force error);
- tests on a 64-bead chain never used in training; why the energy there carries a constant shift per bead, and how to measure it.

### [04 · Learned Dynamics](04-learned-dynamics/)

*The trained networks drive OpenMM.*

- a network inside OpenMM through a per-step custom force;
- energy conservation with learned forces, as good as with the exact forces;
- bonds and angles from OpenMM driven by the networks; R_g at 16, 32 and 64 beads matching the reference;
- cost, limits and next steps.

Each folder holds the notebook in three editions:

| file | what it is |
|---|---|
| `*.ipynb` | the Jupyter notebook, evaluated (Python 3.14) |
| `*.md` | Markdown edition with LaTeX equations, tables and every figure (GitHub renders it) |
| `*.pdf` | printed edition, A4 |
| `figures/` | every figure as PNG |

### References

The bibliography, by Dara and Hoshino: [`docs/REFERENCES.md`](docs/REFERENCES.md) · [`docs/REFERENCES.pdf`](docs/REFERENCES.pdf) · [`docs/references.bib`](docs/references.bib), all generated from [`src/polymer_nn/references.py`](src/polymer_nn/references.py).

## Using the notebooks

### In the browser, with nothing to install

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/lossofgenerality/polymer-neural-nets?quickstart=1)

Click the button and then **Create codespace**. GitHub starts a Linux machine with Python 3.14 and the libraries of `uv.lock`; the first start takes a few minutes. The theory notebook opens by itself: click **Run All**, and if you are asked for a kernel, choose `.venv`. Stop the codespace when you are done (github.com/codespaces → ··· → Stop codespace).

### On your own computer

Install [uv](https://docs.astral.sh/uv/) and, in the folder of this repository:

```
uv sync            # Python 3.14 and the exact library versions of uv.lock, in .venv
uv run pytest      # 24 tests, about 10 s
uv run jupyter lab
```

Open a notebook and choose Kernel → Restart Kernel and Run All Cells. Every code cell starts with an index comment such as `# c.7`, and the cells run in order from c.0. Each notebook runs in under a minute on a laptop, using the data in `data/`, the trained models in `models/` and the check results in `results/`.

### Making everything again

| what | how | time on one CPU core |
|---|---|---|
| the data | `scripts/generate_data.py` (commands in [`data/README.md`](data/README.md)) | a few minutes per file; about 15 min for the scaling data |
| the two models | `scripts/train.py` (commands in notebook 03, section 4) | about 5 min each |
| the checks with the learned forces | `scripts/run_md.py --model models/invariant.pt --ref data/kg_N32.npz --scaling data/scaling_kg.npz --replicas 8` | about 15–25 min per model |
| notebooks, editions, bibliography | `scripts/build_all.sh` | about 3 min |

## The code

```
src/polymer_nn/
  sim.py           the data: OpenMM Kremer–Grest and ideal chains, replicas, pivot Monte Carlo
  graph.py         a frame as a graph: the network input
  models.py        InvariantGNN (SchNet style), EquivariantGNN (EGNN style); forces by autograd
  train.py         force matching
  md.py            the trained network inside OpenMM; velocity Verlet; batched BAOAB Langevin
  analysis.py      the physics checks: R_g and ν, bonds and angles, energy drift
  coarse_grain.py  mappings for coarse-grained force matching
  references.py    the bibliography (Dara, Hoshino)
  nbtools.py       table and figure helpers for the notebooks
scripts/           generate_data, train, run_md, make_references, export_editions, build_all.sh,
                   notebooks/build_*.py (the notebook sources)
tests/             24 tests: physics, symmetries, gradients, cutoff smoothness, MD, training
data/  models/  results/   reference data, trained checkpoints, outputs of the checks
docs/              how-it-works.md (a map of the code), running-large.md, open-source.md, references
```

Every module begins with a long explanation of what it does and why, with the equations it implements; [`docs/how-it-works.md`](docs/how-it-works.md) maps each step of the pipeline to its code and to the notebook section that explains it.

## Results in one table

Both networks were trained on 32-bead Kremer–Grest chains only (details in notebooks 03 and 04).

| check | InvariantGNN | EquivariantGNN | reference (exact forces) |
|---|---|---|---|
| force error, 32-bead validation frames | 0.24 ε/σ (0.9%) | 0.23 ε/σ (0.9%) | RMS force 25.3 ε/σ |
| force error, 64-bead chain never seen | 0.29 ε/σ (1.1%) | 0.27 ε/σ (1.1%) | – |
| energy error per bead, 64 beads, up to a constant | 0.0008 ε | 0.0009 ε | – |
| NVE energy fluctuation per bead, Δt = 0.002 τ | 2.0 × 10⁻⁴ ε | 2.2 × 10⁻⁴ ε | 2.2 × 10⁻⁴ ε |
| mean bond length shift | −0.0004 σ | −0.0005 σ | 0.971 σ |
| R_g at N = 16 / 32 / 64 | 2.28 / 3.64 / 5.54 σ | 2.34 / 3.67 / 5.59 σ | 2.29 / 3.64 / 5.59 σ (MD) |
| effective ν over N = 16–64 | 0.640 ± 0.020 | 0.626 ± 0.011 | 0.643 ± 0.014 |

## Next steps

Coarse-graining for chains of 10⁴–10⁶ beads, NequIP/MACE-style equivariant layers, generative models for chain conformations, and property prediction; see notebook 04, section 6, and [`docs/running-large.md`](docs/running-large.md).

## License

© Computational Classnotes · CC BY-NC-SA 4.0. See [LICENSE.md](LICENSE.md). All libraries used are free and open source ([`docs/open-source.md`](docs/open-source.md)).
