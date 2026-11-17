<!-- © Computational Classnotes · CC BY-NC-SA 4.0 -->
# How it works: a map of the code

*Polymer Neural Nets, version 1.0.0 (17 November 2026) · Dara O Shayda · Noor Hoshino*

This page follows one number, a force on one bead, from the physical model to the trained network
and back into a simulation. It names the file and function that does each step and the notebook
section that explains it.

## 1. The physical model makes the data

| step | code | explained in |
|---|---|---|
| Kremer–Grest chain: FENE bonds + WCA repulsion; ideal chain: harmonic bonds | `sim.build_system`, Eqs. (S1)–(S4) | notebook 00, §10.4; notebook 01, §1 |
| random starting walk, energy minimisation | `sim.initial_positions` | notebook 01, §1 |
| Langevin dynamics at k_BT = 1 (OpenMM `LangevinMiddleIntegrator`) | `sim.sample` | notebook 00, §9.2 |
| a frame every `stride` steps: positions, exact forces, energy | `sim.sample` → `data/*.npz` | notebook 01, §2 |
| many independent chains at once; pivot Monte Carlo | `sim.sample_replicas`, `sim.pivot_mc` | notebook 02, §2; notebook 00, Theorem 11 |

The data are made by `scripts/generate_data.py`; nothing is downloaded (see `data/README.md`).

## 2. A frame becomes a graph

| step | code | explained in |
|---|---|---|
| one node per bead, with position x_i and type z_i (0 = end, 1 = interior) | `graph.to_data`, `graph.node_types` | notebook 01, §3–4 |
| bonded edges i ↔ i ± 1 (type 1) | `graph.chain_bonds` | notebook 00, §3 |
| non-bonded edges for pairs closer than r_c = 1.5 σ (type 0) | `graph.chain_graph` | notebook 01, Figures 3–4 |
| energy target minus a constant per bead | `graph.to_data(energy_offset=…)` | notebook 03, §1 |
| several graphs as one batch | `torch_geometric.data.Batch` | notebook 01, Table 8 |

## 3. The network turns the graph into an energy

| step | code | explained in |
|---|---|---|
| distance r_ij of every edge → 32 Gaussians e_k(r_ij) | `models.GaussianRBF`, Eq. (M2) | notebook 00, §6.1 |
| smooth cutoff f(r_ij) | `models.cosine_envelope`, Eq. (M3) | notebook 00, §6.2, Theorem 6 |
| bead type → initial features h_i⁽⁰⁾ | `nn.Embedding` | notebook 01, §5 |
| 3 message-passing layers | `models.CFConv` (SchNet style) or `models.EGNNLayer` (EGNN style) | notebook 00, §4 and §6.3–6.4 |
| energy per bead ε_i, total E = Σ ε_i | `readout`, Eq. (M10) | notebook 00, Eq. (6.9) |
| forces F_i = −∂E/∂x_i by automatic differentiation | `models._EnergyModel.forward` | notebook 00, §7; notebook 01, §6 |

Every MLP inside is the algebra of notebook 00, §1 (`models.mlp`), checked against PyTorch there.

## 4. Training compares forces with OpenMM

| step | code | explained in |
|---|---|---|
| loss = force MSE + 0.05 × energy MSE per bead | `train._loss`, Eq. (T1) | notebook 03, §1; notebook 00, §8 |
| gradient by double backpropagation | `loss.backward()` | notebook 03, §2 |
| Adam, cosine schedule, gradient clipping | `train.fit` | notebook 00, §2.3 |
| errors on held-out data, energy shift split off | `train.evaluate` | notebook 03, §5–6 |

The models are trained by `scripts/train.py`.

## 5. The trained network drives a simulation

| step | code | explained in |
|---|---|---|
| network as a function positions → (E, F) | `md.LearnedPotential` | notebook 04, §1 |
| OpenMM with a `CustomExternalForce` refreshed every step | `md.OpenMMLearnedSimulation` | notebook 04, §1 |
| constant-energy velocity Verlet | `md.velocity_verlet`, Eq. (D1) | notebook 00, §9.1 |
| batched BAOAB Langevin for many chains at once | `md.langevin_batched`, Eq. (D2) | notebook 00, §9.2 |

## 6. The physics checks

| check | code | results |
|---|---|---|
| R_g ∼ N^ν | `analysis.radius_of_gyration`, `analysis.fit_flory_exponent` | notebook 02, §2 (reference); notebook 04, §4 (learned) |
| bond lengths and angles | `analysis.bond_lengths`, `analysis.bond_angles`, `analysis.compare_distributions` | notebook 02, §3; notebook 04, §3 |
| energy conservation | `analysis.energy_drift` | notebook 02, §4; notebook 04, §2 |

They are run on the trained models by `scripts/run_md.py`, which writes `results/`.

## 7. Building the notebooks and their editions

The notebooks are written as lists of cells in `scripts/notebooks/build_*.py`, executed with
nbconvert, and exported to Markdown and PDF:

```
uv run python scripts/notebooks/build_00_theory.py          # and build_01 … build_04
uv run jupyter nbconvert --to notebook --execute --inplace 00-theory/*.ipynb
uv run python scripts/export_editions.py                    # .md and .pdf next to each notebook
uv run python scripts/make_references.py                    # docs/REFERENCES.md, .bib, .pdf
```

`scripts/build_all.sh` does all of this in order.

© Computational Classnotes · CC BY-NC-SA 4.0
