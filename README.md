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

Every notebook runs on its own in any Jupyter. Its first cell, c.0, installs any missing libraries with pip, finds the repository folder, and, if the notebook was opened by itself, downloads the code and data of this repository once from GitHub into `polymer-neural-nets` in your home folder; the details are in [Running a notebook on its own in Jupyter](#running-a-notebook-on-its-own-in-jupyter). Then choose Kernel → Restart Kernel and Run All Cells. Every code cell starts with an index comment such as `# c.7`, and the cells run in order from c.0. Each notebook runs in under a minute on a laptop.

### In your own Jupyter (JupyterLab, Jupyter Notebook, VS Code, Anaconda)

Either download one notebook and open it, which is all you need, or get the whole repository and install the libraries first:

```
git clone https://github.com/lossofgenerality/polymer-neural-nets
cd polymer-neural-nets
pip install -r requirements.txt     # NumPy, SciPy, SymPy, Matplotlib, PyTorch, PyTorch Geometric, OpenMM, Jupyter
jupyter lab
```

Python 3.10 or later works; the notebooks were run with Python 3.14. With conda, `conda install -c conda-forge openmm pytorch` first, then the same `pip install -r requirements.txt`.

### With uv, using the exact versions of `uv.lock`

```
uv sync            # Python 3.14 and the exact library versions of uv.lock, in .venv
uv run pytest      # 24 tests, about 10 s
uv run jupyter lab
```

### In the browser, with nothing to install

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/lossofgenerality/polymer-neural-nets?quickstart=1)

Click the button and then **Create codespace**; the theory notebook opens by itself. Choose the `.venv` kernel and click **Run All**.

### Making everything again

| what | how | time on one CPU core |
|---|---|---|
| the data | `scripts/generate_data.py` (commands in [`data/README.md`](data/README.md)) | a few minutes per file; about 15 min for the scaling data |
| the two models | `scripts/train.py` (commands in notebook 03, section 4) | about 5 min each |
| the checks with the learned forces | `scripts/run_md.py --model models/invariant.pt --ref data/kg_N32.npz --scaling data/scaling_kg.npz --replicas 8` | about 15–25 min per model |
| notebooks, editions, bibliography | `scripts/build_all.sh` | about 3 min |

## Running a notebook on its own in Jupyter

Every notebook can run by itself, outside this repository, in any Jupyter that runs a real Python on your computer: JupyterLab, Jupyter Notebook, VS Code, PyCharm or Anaconda Navigator. Jupyters that run Python inside the web page (JupyterLite, Pyodide) cannot run PyTorch and OpenMM; see the troubleshooting table below. Download one `.ipynb` file (on GitHub: open the file and click **Download raw file**, the download arrow above the file), open it in Jupyter and choose **Kernel → Restart Kernel and Run All Cells**. Nothing else needs to be set up by hand.

### What you need

| | |
|---|---|
| Python | 3.10 or later, with Jupyter (the notebooks were run with Python 3.14) |
| internet | on the first run only: to install the libraries and download the repository |
| disk space | about 15 MB for the repository; PyTorch is the largest library, a few hundred MB on macOS and Windows and a few GB on Linux, where the standard wheel includes the NVIDIA GPU libraries |
| a GPU | not needed; everything runs on the CPU |

### What the first cell (c.0) does, step by step

1. **Checks the libraries.** It looks for each library below in the Python of the running Jupyter kernel and installs the missing ones with `pip` into that same Python (`python -m pip install …`). In environments made by uv, which have no pip, it uses `uv pip install --python …` instead. The first install can take several minutes because of PyTorch; later runs skip this step.

   | imported as | installed as | used for |
   |---|---|---|
   | `numpy` | `numpy` | arrays |
   | `scipy` | `scipy` | fits and statistical tests |
   | `sympy` | `sympy` | the symbolic derivations of notebooks 00 and 01 |
   | `matplotlib` | `matplotlib` | figures |
   | `torch` | `torch` | the networks, autograd, training |
   | `torch_geometric` | `torch-geometric` | graphs and message passing |
   | `openmm` | `openmm` | the molecular dynamics |
   | `IPython` | `ipython` | tables in the notebook |

2. **Finds the repository.** It looks in the notebook's folder and in every folder above it for `src/polymer_nn` and `data/`. If the notebook is inside a copy of this repository, that copy is used and nothing is downloaded.

3. **Otherwise downloads it once.** It downloads `https://github.com/lossofgenerality/polymer-neural-nets/archive/refs/heads/main.zip` (about 15 MB) and unpacks it into a folder `polymer-neural-nets` in your home folder: `~/polymer-neural-nets` on macOS and Linux, `C:\Users\<you>\polymer-neural-nets` on Windows. Every later run, of this or any other notebook, finds that folder and downloads nothing.

4. **Uses the repository's code.** It puts the repository's `src` folder first on Python's search path, so `import polymer_nn` loads the code that belongs to the data and models.

5. **Sets up the notebook.** It applies the figure and table style, uses 2 CPU threads, fixes the random seeds, and prints one line such as `repository folder: polymer-neural-nets | Python 3.14.6 | torch 2.15.0 | numpy 2.5.5`.

### What each notebook reads and writes

| notebook | reads from the repository | run time on a laptop |
|---|---|---|
| 00 · The Theory | `data/kg_N32.npz` | about 30 s |
| 01 · The Data and the Graph | `data/kg_N32.npz`, `data/kg_N64.npz`, `data/scaling_kg.npz`, `models/invariant.pt`, `models/egnn.pt`, `models/invariant.json`, `results/checks_invariant.npz` | about 15 s |
| 02 · Reference Simulations | `data/kg_N32.npz`, `data/scaling_kg.npz`, `data/scaling_ideal.npz` | about 20 s |
| 03 · Training | `data/kg_N32.npz`, `data/kg_N64.npz`, `models/*.pt`, `models/*.json` | about 15 s |
| 04 · Learned Dynamics | `data/kg_N32.npz`, `data/kg_N64.npz`, `data/scaling_kg.npz`, `models/*.pt`, `results/checks_*.json`, `results/checks_*.npz` | about 15 s |

The run times are for a run after the first; the first run adds the time to install the libraries and download the repository. Each notebook writes only its figures, as PNG files in a folder `figures` next to the notebook; nothing in the repository is changed. The long computations (making the data, training, the full MD checks) are not repeated in the notebooks; their results are read from `data/`, `models/` and `results/`. Each notebook also runs a short live version, such as a simulation, a training run or an OpenMM run with the network, so every step can be seen working. See [Making everything again](#making-everything-again) to redo the long runs.

### Getting a newer version

The downloaded copy is not updated automatically. To get the latest version, delete the folder `polymer-neural-nets` in your home folder and run cell c.0 again, or use `git clone` and `git pull` (next section).

### Working offline

Run any notebook once while online, or clone the repository while online (`git clone https://github.com/lossofgenerality/polymer-neural-nets`), and install the libraries with `pip install -r requirements.txt`. After that no internet connection is needed.

### If something goes wrong

| problem | what to do |
|---|---|
| c.0 says *This Jupyter runs Python inside the web browser* (JupyterLite, Pyodide; paths such as `/home/pyodide/…`) | a Jupyter that runs Python inside the web page cannot run PyTorch or OpenMM. Use Jupyter with a real Python on your computer, which also opens in the browser: install [uv](https://docs.astral.sh/uv/), then `git clone https://github.com/lossofgenerality/polymer-neural-nets`, `cd polymer-neural-nets`, `uv sync`, `uv run jupyter notebook`. The saved outputs can still be read in the notebook and in the `.md` and `.pdf` editions |
| `ModuleNotFoundError` right after the libraries were installed | restart the kernel (Kernel → Restart Kernel) and run again: a library installed during a session is sometimes visible only after a restart |
| the libraries are installed in the wrong Python | choose the kernel of the Python where you want them (Kernel → Change Kernel); c.0 always installs into the Python of the running kernel |
| `No module named pip` | run `python -m ensurepip` in that Python, or install uv, which c.0 then uses |
| c.0 stops with *Could not install … into the Python of this Jupyter kernel* | the kernel's Python cannot install packages (no pip, or locked by the system or Homebrew). The message shows that Python's path, the exact error and three fixes. The surest fix is to run Jupyter from the repository's own environment: in a terminal, in the repository folder, `uv sync` and then `uv run jupyter notebook`, or make a venv: `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install -r requirements.txt`, `jupyter notebook` |
| OpenMM does not install with pip (older systems) | install it with conda: `conda install -c conda-forge openmm`, then run c.0 again |
| the download is blocked by a firewall or proxy | download the zip above in a browser, unzip it, rename the folder to `polymer-neural-nets` and put it in your home folder, or put the notebook inside the unzipped folder |
| an old copy in your home folder no longer matches the notebook | delete `~/polymer-neural-nets` and run c.0 again |
| PyTorch takes too much disk space on Linux | install the CPU-only build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

### Removing it again

Delete the folder `polymer-neural-nets` in your home folder and the `figures` folders next to the notebooks. To remove the libraries as well, run `pip uninstall torch torch-geometric openmm` (and the others in the table above, if nothing else uses them).

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
