<!-- © Computational Classnotes · CC BY-NC-SA 4.0 -->
# data/

*Polymer Neural Nets, version 1.0.0 (17 November 2026)*

Every file here was made by `scripts/generate_data.py` with OpenMM, from the Kremer–Grest and ideal
bead-spring models of `src/polymer_nn/sim.py`. Nothing was downloaded and nothing is measured. Notebook
[`01-data-and-graph`](../01-data-and-graph/The-Data-and-the-Graph.md) opens every file and explains every array.

| file | contents | made with |
|---|---|---|
| `kg_N32.npz` | training set: 2000 frames of a 32-bead Kremer–Grest chain | `--model kg --n-beads 32 --frames 2000 --out data/kg_N32.npz` |
| `kg_N64.npz` | test set: 300 frames of a 64-bead chain (never used for training) | `--model kg --n-beads 64 --frames 300 --seed 7 --out data/kg_N64.npz` |
| `scaling_kg.npz` | R_g at N = 16, 32, 64, 128 from 8 independent chains each, with a pivot-MC check | `--scaling --models kg --replicas 8 --frames 200 --stride 1000 --equilibrate 100000 --out data/scaling_kg.npz` |
| `scaling_ideal.npz` | the same for the ideal chain, 16 chains each | `--scaling --models ideal --replicas 16 --frames 1000 --stride 1000 --equilibrate 100000 --out data/scaling_ideal.npz` |

Each command is run as `uv run python scripts/generate_data.py …`.

## Layout of a single-chain file

| array | shape | type | unit | meaning |
|---|---|---|---|---|
| `positions` | (F, N, 3) | float32 | σ | x_i of every bead in every frame: axes (frame, bead, x/y/z) |
| `forces` | (F, N, 3) | float32 | ε/σ | exact force F_i = −∂E/∂x_i from OpenMM |
| `energy` | (F,) | float64 | ε | total potential energy E of every frame |
| `n_beads` | scalar | int64 | – | N |
| `model` | scalar | string | – | `kg` (Kremer–Grest) or `ideal` |
| `kT` | scalar | float64 | ε | k_BT (1) |

F is the number of frames and N the number of beads. Units are reduced Lennard-Jones units: σ = bead
diameter, ε = energy scale, τ = σ√(m/ε) the time unit (in OpenMM: σ = 1 nm, ε = 1 kJ/mol, m = 1 amu, τ = 1 ps).

## Layout of a scaling file

| array | shape | meaning |
|---|---|---|
| `sizes` | (4,) | the chain lengths N |
| `<model>_rg` | (4,) | ⟨R_g²⟩^½ from MD at each N, in σ |
| `<model>_rg_sem` | (4,) | its standard error, from the spread of the independent chains |
| `<model>_final_N<n>` | (R, n, 3) | the last configuration of each of the R chains |
| `kg_pivot_rg` | (4,) | ⟨R_g²⟩^½ from pivot Monte Carlo started at those configurations (Kremer–Grest only) |

© Computational Classnotes · CC BY-NC-SA 4.0
