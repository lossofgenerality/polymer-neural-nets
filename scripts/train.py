# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Train a graph-network force field on OpenMM data by force matching.

Steps:

1. read the data file and turn every frame into a graph (``polymer_nn.graph``);
   subtract the mean energy per bead e₀ from the energy targets;
2. shuffle with a fixed seed and keep ``--val-frac`` of the frames for
   validation (never used to change the weights);
3. build the network (``--model invariant`` or ``egnn``, ``--hidden`` channels,
   ``--layers`` message-passing layers, cutoff ``--cutoff``);
4. minimise the loss of ``polymer_nn.train`` (forces + 0.05 × energy per bead)
   with Adam, a cosine learning-rate schedule and gradient clipping, for
   ``--epochs`` passes over the training frames in batches of ``--batch-size``;
5. save the checkpoint (``--out``) and the training history next to it as JSON;
   if ``--test`` is given, evaluate the trained model on that file too.

The models in ``models/`` were trained with::

    uv run python scripts/train.py --data data/kg_N32.npz --test data/kg_N64.npz --model invariant --epochs 25 --out models/invariant.pt
    uv run python scripts/train.py --data data/kg_N32.npz --test data/kg_N64.npz --model egnn --epochs 20 --out models/egnn.pt

Notebook 03 explains the loss, the gradient and the results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from polymer_nn.graph import dataset_from_npz
from polymer_nn.models import build_model, save_model
from polymer_nn.train import TrainConfig, evaluate, fit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--test", type=Path, help="optional held-out file, e.g. a longer chain")
    p.add_argument("--model", default="invariant", choices=["invariant", "egnn"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--cutoff", type=float, default=1.5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    data, offset = dataset_from_npz(a.data, cutoff=a.cutoff)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(data))
    n_val = max(1, int(a.val_frac * len(data)))
    val = [data[i] for i in idx[:n_val]]
    train = [data[i] for i in idx[n_val:]]
    config = dict(hidden=a.hidden, layers=a.layers, cutoff=a.cutoff)
    model = build_model(a.model, **config)
    print(f"{a.model}: {sum(p.numel() for p in model.parameters())} parameters, "
          f"{len(train)} train / {len(val)} val frames")
    hist = fit(model, train, val, TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, device=a.device))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model.cpu(), a.out, a.model, config, offset)
    if a.test:
        test, _ = dataset_from_npz(a.test, cutoff=a.cutoff, energy_offset=offset)
        hist["test"] = evaluate(model, test)
        print("held-out test:", hist["test"])
    hist = {"copyright": "© Computational Classnotes · CC BY-NC-SA 4.0", **hist}
    a.out.with_suffix(".json").write_text(json.dumps(hist, indent=2, ensure_ascii=False))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
