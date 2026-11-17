# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""From a polymer configuration to a graph: what the network receives as input.

A neural network cannot take "a polymer" as input; it needs numbers in a fixed
format. Here a configuration is encoded as a **graph** G = (V, E) in the
format of PyTorch Geometric [fey2019], a ``torch_geometric.data.Data`` object.
The notebook *The Data and the Graph* (``01-data-and-graph``) draws this graph
for real frames and follows one through the network.

Nodes
-----
One node per bead, i = 0, …, N-1. Each node carries

* its **position** x_i ∈ ℝ³ (``data.pos``, float tensor N × 3, in σ). This is
  the only geometric input. The energy must not depend on where the chain
  is or how it is turned, so the networks use positions only through
  differences x_i - x_j and distances r_ij = |x_i - x_j|.
* its **type** z_i ∈ {0, 1} (``data.z``, integer tensor N): 0 for the two end
  beads, 1 for interior beads. Ends have one bond and interior beads two,
  so they sit in different environments. The type is turned into the initial
  feature vector h_i⁽⁰⁾ by a learned lookup table (an embedding).

Edges
-----
Messages travel along directed edges j → i, stored as two rows of indices,
``data.edge_index`` (integer tensor 2 × |E|): row 0 holds the sources j, row 1
the targets i. Every edge is stored in both directions. There are two kinds:

* **bonded edges**, (i, i+1) and (i+1, i) for every bond, whatever the
  distance: 2(N - 1) edges with ``edge_type`` = 1;
* **non-bonded edges**, every other ordered pair with r_ij < r_c, where the
  cutoff is r_c = 1.5 σ by default: ``edge_type`` = 0.

So

    E_bond = {(i, i±1)},     E_nb = {(i, j) : |i - j| ≥ 2, r_ij < r_c},     (G1)

and ``data.edge_type`` (integer tensor |E|) tells the two apart. The cutoff
makes the graph sparse: each bead has only a handful of neighbours, so the
cost of the network grows linearly with N.

Targets (training only)
-----------------------
* ``data.force``: the reference forces, float tensor N × 3, in ε/σ;
* ``data.energy``: the reference energy minus a constant per bead (float
  tensor of length 1, in ε), see :func:`to_data`.

Batches
-------
For training, several graphs are joined into one big disconnected graph
(``torch_geometric.data.Batch``). Node indices of graph g are shifted by the
number of nodes of the graphs before it, and the vector ``batch.batch``
(length = total nodes) records which graph each node belongs to, so the
energy of each graph can be summed separately.

Large chains
------------
The neighbour search below is a plain distance matrix per chain, O(N²),
which is fine up to a few thousand beads. For 10⁴–10⁶ beads, replace it with
a cell list (``torch_cluster.radius_graph`` or ``matscipy``); nothing else
needs to change.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch, Data


def chain_bonds(n_beads: int, device=None) -> torch.Tensor:
    """Bonded edges of a linear chain in both directions, shape (2, 2(N-1)).

    For N = 4 this is [[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]]: first the
    forward edges i → i+1, then the backward edges i+1 → i.
    """
    i = torch.arange(n_beads - 1, device=device)
    src = torch.cat([i, i + 1])
    dst = torch.cat([i + 1, i])
    return torch.stack([src, dst])


def node_types(n_beads: int) -> torch.Tensor:
    """Node types z_i: 0 for the two chain ends, 1 for interior beads."""
    t = torch.ones(n_beads, dtype=torch.long)
    t[0] = t[-1] = 0
    return t


def chain_graph(pos: torch.Tensor, bonds: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Edges and edge types of one chain, Eq. (G1). ``pos`` is (N, 3).

    The bonded edges come first (type 1), then every other ordered pair
    closer than ``cutoff`` (type 0). The search runs without gradients:
    *which* edges exist is a discrete choice. The model's smooth cutoff
    envelope (``models.cosine_envelope``) makes sure the energy changes
    continuously when an edge appears or disappears.
    """
    with torch.no_grad():
        d = torch.cdist(pos, pos)  # all pairwise distances, N × N
        close = d < cutoff
        close.fill_diagonal_(False)  # no self-loops
        close[bonds[0], bonds[1]] = False  # bonded pairs are added separately, with type 1
        nb = close.nonzero().T  # (2, number of non-bonded edges), rows = (source j, target i)
    edge_index = torch.cat([bonds, nb], dim=1)
    edge_type = torch.cat(
        [torch.ones(bonds.shape[1], dtype=torch.long), torch.zeros(nb.shape[1], dtype=torch.long)]
    ).to(pos.device)
    return edge_index, edge_type


def to_data(positions, forces=None, energy=None, cutoff: float = 1.5, energy_offset: float = 0.0) -> Data:
    """One configuration as a PyG ``Data`` graph (the network input).

    Parameters
    ----------
    positions : (N, 3) array of bead positions in σ.
    forces : optional (N, 3) reference forces in ε/σ (training target).
    energy : optional reference energy in ε (training target).
    cutoff : r_c of Eq. (G1).
    energy_offset : a constant e₀ per bead removed from the energy target,
        E_target = E - N e₀. The raw energies are about 20 ε per bead (the
        FENE springs and WCA shifts), but only differences of energy matter
        physically; removing the mean makes the target O(1), which helps
        training. Forces are unchanged, because the constant does not depend
        on the positions.

    Returns
    -------
    ``Data(pos, z, edge_index, edge_type[, force, energy])``.
    """
    pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32)
    n = pos.shape[0]
    bonds = chain_bonds(n)
    edge_index, edge_type = chain_graph(pos, bonds, cutoff)
    data = Data(pos=pos, z=node_types(n), edge_index=edge_index, edge_type=edge_type)
    if forces is not None:
        data.force = torch.as_tensor(np.asarray(forces), dtype=torch.float32)
    if energy is not None:
        data.energy = torch.tensor([float(energy) - energy_offset * n], dtype=torch.float32)
    return data


def rebuild_edges(batch: Batch, cutoff: float) -> Batch:
    """Recompute the edges of every graph in a batch after the beads moved.

    During MD the neighbours change, so the graph is rebuilt at every step.
    ``batch.ptr`` holds where each graph's nodes start and end.
    """
    out_ei, out_et = [], []
    ptr = batch.ptr.tolist()
    for g in range(batch.num_graphs):
        a, b = ptr[g], ptr[g + 1]
        bonds = chain_bonds(b - a, device=batch.pos.device)
        ei, et = chain_graph(batch.pos[a:b], bonds, cutoff)
        out_ei.append(ei + a)  # shift local indices to batch-wide indices
        out_et.append(et)
    batch.edge_index = torch.cat(out_ei, dim=1)
    batch.edge_type = torch.cat(out_et)
    return batch


def dataset_from_npz(path_or_dict, cutoff: float = 1.5, energy_offset: float | None = None):
    """Turn a data file of ``scripts/generate_data.py`` into a list of graphs.

    Every frame becomes one ``Data`` graph with its force and energy targets.
    If ``energy_offset`` is None it is set to the mean energy per bead of
    this file. Returns ``(data_list, energy_offset)``; pass the same offset when
    building a validation or test set, so that energies are comparable.
    """
    is_path = isinstance(path_or_dict, (str, bytes)) or hasattr(path_or_dict, "__fspath__")
    d = np.load(path_or_dict) if is_path else path_or_dict
    P, F, E = d["positions"], d["forces"], d["energy"]
    n = P.shape[1]
    if energy_offset is None:
        energy_offset = float(np.mean(E) / n)
    data = [to_data(P[k], F[k], E[k], cutoff, energy_offset) for k in range(len(E))]
    return data, energy_offset
