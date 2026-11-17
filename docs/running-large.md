<!-- © Computational Classnotes · CC BY-NC-SA 4.0 -->
# Running large: Mac, Codespaces, GPU machines

*Polymer Neural Nets, version 1.0.0 (17 November 2026) · Dara O Shayda · Noor Hoshino*

Everything in the repository runs on a laptop CPU at N = 16–128. The settings below
scale it up.

## Install

```bash
uv sync                        # Python 3.14 and the exact library versions of uv.lock
uv run pytest                  # 24 tests, about 10 s
```

`uv.lock` pins the standard PyTorch wheel from PyPI, which runs on CPU everywhere and uses an
NVIDIA GPU when one is present. On Apple silicon, PyTorch uses the GPU through `--device mps`.

## Where to run what

| Machine | Good for | Flags |
|---|---|---|
| GitHub Codespaces (4–8 cores) | tests, notebooks, data up to N ≈ 256 | `--platform CPU --device cpu` |
| Mac (Apple silicon) | training medium models | `--device mps`, and OpenMM `--platform OpenCL` |
| NVIDIA GPU | training, N = 10⁴–10⁶, melts | `--device cuda`, OpenMM `--platform CUDA` |

## Production recipe

1. **Reference data at scale.** Generate a Kremer–Grest melt with periodic boundaries
   (switch `CustomNonbondedForce` to `CutoffPeriodic` and add a box), or use LAMMPS
   for very large melts [thompson2022]. Save frames in the same `.npz` layout.
2. **Neighbour lists.** Replace the dense `torch.cdist` in `graph.chain_graph` with
   `torch_cluster.radius_graph`, so graph construction stays O(N).
3. **Train** with `scripts/train.py --device cuda --batch-size 64 --epochs 200`.
   A model trained on short chains transfers to longer ones, because its energy is a
   sum over beads (see `--test` and the R_g check).
4. **Run MD on the GPU** through `openmm-torch` (conda-forge) [eastman2024]:

   ```python
   import torch
   from openmmtorch import TorchForce
   # wrap the model in a TorchScript module mapping positions (nm) -> energy (kJ/mol)
   force = TorchForce(torch.jit.script(wrapper))
   system.addForce(force)
   ```

   The pip-only bridge in `md.OpenMMLearnedSimulation` gives the same physics but
   calls Python every step, so it is meant for testing and small chains.
5. **Coarse-grain** with `polymer_nn.coarse_grain` to reach 10⁵–10⁶ effective beads.

© Computational Classnotes · CC BY-NC-SA 4.0
