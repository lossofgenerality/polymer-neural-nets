# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Training by force matching: how the weights are learned.

The loss
========

For a batch of B graphs with N_g beads each, predicted energies Ê_g and
forces F̂_i, and reference energies E_g and forces F_i from OpenMM,

    L(θ) = w_F · (1/3N_tot) Σ_i |F̂_i - F_i|²  +  w_E · (1/B) Σ_g ((Ê_g - E_g)/N_g)²,    (T1)

with w_F = 1 and w_E = 0.05. The first term is **force matching**
[izvekov2005, noid2008]: the mean squared error of all 3 N_tot force
components. The second keeps the energy right as well; dividing by N_g makes
it an error per bead, so long and short chains count alike. Forces carry far
more information than energies (3N numbers per frame instead of one), which
is why they dominate the loss (theory notebook, section 8).

The forces F̂ = -∂Ê/∂x are themselves derivatives of the network, so
∂L/∂θ needs second derivatives of Ê (∂²Ê/∂θ∂x). PyTorch computes them because
the model runs with ``create_graph=True`` in training mode.

The optimiser
=============

Adam [kingma2015] with learning rate η (``lr`` = 2·10⁻³) and a cosine
schedule [loshchilov2017] η_t = ½ η (1 + cos(π t / T)) over T epochs. Adam
keeps running averages of each gradient and of its square and takes steps
of about η in every direction (theory notebook, Eq. (2.6)). The norm of the
whole gradient is clipped at 10, so one unusual batch cannot throw the
weights far away.

One *epoch* is one pass over the training frames in random order, in
batches of ``batch_size`` graphs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .models import _EnergyModel


@dataclass
class TrainConfig:
    """Training settings; the defaults are those used for the models in ``models/``."""

    epochs: int = 30
    batch_size: int = 16
    lr: float = 2e-3
    weight_force: float = 1.0  # w_F of Eq. (T1)
    weight_energy: float = 0.05  # w_E of Eq. (T1); set 0 for coarse-grained data
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1


def _loss(model, batch, cfg: TrainConfig):
    """Eq. (T1) for one batch; also returns the force and energy terms."""
    e, f = model(batch)
    n = torch.bincount(batch.batch).to(e.dtype)  # beads per graph, N_g
    lf = ((f - batch.force) ** 2).mean()  # force MSE over all components
    le = (((e - batch.energy) / n) ** 2).mean()  # energy MSE per bead
    return cfg.weight_force * lf + cfg.weight_energy * le, lf.detach(), le.detach()


def evaluate(model: _EnergyModel, data, batch_size: int = 32, device="cpu") -> dict[str, float]:
    """Errors of a model on a list of graphs.

    Returns

    * ``force_rmse``: root-mean-square error of the force components (ε/σ);
    * ``energy_rmse_per_bead``: RMS of (Ê - E)/N (ε);
    * ``energy_rmse_per_bead_centred`` and ``energy_shift_per_bead``: the same
      error split into a constant shift and the rest. Data at a single chain
      length cannot fix how a constant energy is shared between end and
      interior beads, so on chains of another length the absolute energy may
      carry a constant shift per bead. A constant changes no force, so the
      centred error is the one that measures the energy landscape.
    """
    model.eval()
    sf = se = nf = ne = 0.0
    de = []
    for batch in DataLoader(data, batch_size=batch_size):
        batch = batch.to(device)
        e, f = model(batch)
        n = torch.bincount(batch.batch).to(e.dtype)
        sf += float(((f.detach() - batch.force) ** 2).sum())
        nf += f.numel()
        se += float((((e.detach() - batch.energy) / n) ** 2).sum())
        ne += e.numel()
        de.append(((e.detach() - batch.energy) / n).cpu().numpy())
    de = np.concatenate(de)
    return {"force_rmse": (sf / nf) ** 0.5, "energy_rmse_per_bead": (se / ne) ** 0.5,
            "energy_rmse_per_bead_centred": float(np.std(de)), "energy_shift_per_bead": float(np.mean(de))}


def fit(model: _EnergyModel, train, val, cfg: TrainConfig = TrainConfig(), verbose=True) -> dict:
    """Train ``model`` on the list of graphs ``train``; check on ``val`` after every epoch.

    Returns the history: per epoch, the training and validation force RMSE
    and the validation energy RMSE per bead, together with the settings and
    the RMS size of the reference forces (``force_scale``), so that errors can
    be read as a fraction of the typical force.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
    force_scale = float(np.sqrt(np.mean([float((d.force**2).mean()) for d in train])))
    history = {"epoch": [], "train_force_rmse": [], "val_force_rmse": [], "val_energy_rmse_per_bead": [],
               "config": asdict(cfg), "force_scale": force_scale}
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        acc, cnt = 0.0, 0
        for batch in loader:
            batch = batch.to(cfg.device)
            opt.zero_grad()  # gradients accumulate in PyTorch, so reset them
            loss, lf, _ = _loss(model, batch, cfg)
            loss.backward()  # ∂L/∂θ for every weight, by backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()  # one Adam step
            acc += float(lf) * batch.num_graphs
            cnt += batch.num_graphs
        sched.step()  # next learning rate of the cosine schedule
        v = evaluate(model, val, device=cfg.device)
        history["epoch"].append(epoch)
        history["train_force_rmse"].append((acc / cnt) ** 0.5)
        history["val_force_rmse"].append(v["force_rmse"])
        history["val_energy_rmse_per_bead"].append(v["energy_rmse_per_bead"])
        if verbose and epoch % cfg.log_every == 0:
            print(f"epoch {epoch:3d}  train F-RMSE {history['train_force_rmse'][-1]:.3f}  "
                  f"val F-RMSE {v['force_rmse']:.3f} (|F|~{force_scale:.2f})  "
                  f"val E-RMSE/bead {v['energy_rmse_per_bead']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
    return history
