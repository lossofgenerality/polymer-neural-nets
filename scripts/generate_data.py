# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Make the reference data: OpenMM simulations of bead-spring chains.

Every number the networks learn from is made by this script; nothing is
downloaded. It calls ``polymer_nn.sim`` (read its module docstring for the
physical model, the units and the procedure). Two modes:

**Single chain** (default): one chain of ``--n-beads`` beads, Langevin dynamics
at k_BT = 1, ``--equilibrate`` steps thrown away, then ``--frames`` frames
recorded every ``--stride`` steps. The output ``.npz`` file holds

    positions (F, N, 3) σ · forces (F, N, 3) ε/σ · energy (F,) ε · n_beads · model · kT

These are the training and test sets (notebook 01, section 2).

**Scaling** (``--scaling``): for every N in ``--sizes``, ``--replicas``
independent chains in one OpenMM system (friction 0.1/τ by default, starting
walks relaxed by ``--pivot-moves`` pivot Monte Carlo moves). Writes the
root-mean-square R_g per N with a standard error from the spread of the
replicas, the final configuration of every replica, and an independent
pivot-MC estimate of R_g (``--pivot-check`` moves per replica). These are
the reference values of the R_g ∼ N^ν check (notebook 02, section 2).

The files in ``data/`` were made with::

    uv run python scripts/generate_data.py --model kg --n-beads 32 --frames 2000 --out data/kg_N32.npz
    uv run python scripts/generate_data.py --model kg --n-beads 64 --frames 300 --seed 7 --out data/kg_N64.npz
    uv run python scripts/generate_data.py --scaling --models kg --replicas 8 --frames 200 --stride 1000 \
        --equilibrate 100000 --out data/scaling_kg.npz
    uv run python scripts/generate_data.py --scaling --models ideal --replicas 16 --frames 1000 --stride 1000 \
        --equilibrate 100000 --out data/scaling_ideal.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from polymer_nn.analysis import radius_of_gyration
from polymer_nn.sim import ChainSpec, pivot_mc, sample, sample_replicas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="kg", choices=["kg", "ideal"])
    p.add_argument("--n-beads", type=int, default=32)
    p.add_argument("--frames", type=int, default=2000)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--equilibrate", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--platform", default="CPU")
    p.add_argument("--scaling", action="store_true", help="R_g vs N for both chain models")
    p.add_argument("--sizes", default="16,32,64,128")
    p.add_argument("--replicas", type=int, default=16, help="independent chains per size (--scaling)")
    p.add_argument("--friction", type=float, default=None,
                   help="Langevin friction in 1/tau (default 1; --scaling uses 0.1 for faster sampling)")
    p.add_argument("--models", default="ideal,kg")
    p.add_argument("--pivot-moves", type=int, default=3000,
                   help="pivot-MC pre-equilibration of each starting walk (--scaling)")
    p.add_argument("--pivot-check", type=int, default=4000,
                   help="pivot-MC moves per replica for the independent R_g check (0 = off)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.scaling:
        sizes = [int(s) for s in a.sizes.split(",")]
        res = {"sizes": np.array(sizes)}
        fr = 0.1 if a.friction is None else a.friction
        for model in a.models.split(","):
            rg_mean, rg_sem = [], []
            for n in sizes:
                t = time.time()
                d = sample_replicas(ChainSpec(n_beads=n, model=model, seed=a.seed, platform=a.platform, friction=fr),
                                    n_replicas=a.replicas, n_frames=a.frames, stride=a.stride,
                                    equilibrate=a.equilibrate, pivot_moves=a.pivot_moves)
                rg = radius_of_gyration(d["positions"].reshape(-1, n, 3)).reshape(a.frames, a.replicas)
                rg_mean.append(np.sqrt(np.mean(rg**2)))
                # error bar from the spread between independent replicas
                rg_sem.append(np.std(np.sqrt(np.mean(rg**2, axis=0))) / np.sqrt(a.replicas))
                res[f"{model}_final_N{n}"] = d["final"]
                if a.pivot_check:
                    rng = np.random.default_rng(a.seed + n)
                    tr = np.concatenate([pivot_mc(x, a.pivot_check, rng, excluded_volume=model == "kg",
                                                  record_every=10)[1][len(range(0, a.pivot_check, 10)) // 4:]
                                         for x in d["final"]])
                    res[f"{model}_pivot_rg"] = np.append(res.get(f"{model}_pivot_rg", np.array([])),
                                                         np.sqrt(np.mean(tr**2)))
                piv = f"  pivot MC {res[f'{model}_pivot_rg'][-1]:.3f}" if a.pivot_check else ""
                print(f"{model:5s} N={n:4d}  MD <Rg^2>^1/2 = {rg_mean[-1]:.3f} ± {rg_sem[-1]:.3f}{piv}"
                      f"  [{time.time() - t:.0f}s]", flush=True)
            res[f"{model}_rg"] = np.array(rg_mean)
            res[f"{model}_rg_sem"] = np.array(rg_sem)
        np.savez(a.out, **res)
    else:
        t = time.time()
        d = sample(ChainSpec(n_beads=a.n_beads, model=a.model, seed=a.seed, platform=a.platform,
                             friction=1.0 if a.friction is None else a.friction),
                   n_frames=a.frames, stride=a.stride, equilibrate=a.equilibrate)
        np.savez_compressed(a.out, **d)
        print(f"wrote {a.out}  ({a.frames} frames, N={a.n_beads}, {time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
