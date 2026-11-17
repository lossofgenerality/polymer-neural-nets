# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Tests of the graph construction (polymer_nn.graph), the network input.

Every bond must appear in both directions with type 1, every other pair
closer than the cutoff with type 0, and nothing else; the energy offset must
make the energy targets of order 1.
"""
import numpy as np
import torch

from polymer_nn.graph import dataset_from_npz, to_data


def test_edges(small_dataset):
    x = small_dataset["positions"][0]
    d = to_data(x, cutoff=1.5)
    src, dst = d.edge_index
    assert (src != dst).all()
    bonded = d.edge_type == 1
    assert int(bonded.sum()) == 2 * (len(x) - 1)
    assert ((src[bonded] - dst[bonded]).abs() == 1).all()
    r = (d.pos[src] - d.pos[dst]).norm(dim=-1)
    assert (r[~bonded] < 1.5).all()
    # every close non-bonded pair is present
    D = torch.cdist(d.pos, d.pos)
    close = int(((D < 1.5) & (D > 0)).sum()) - 2 * (len(x) - 1)
    assert int((~bonded).sum()) == close


def test_dataset_offset(small_dataset):
    data, off = dataset_from_npz(small_dataset)
    e = np.array([float(g.energy) for g in data])
    assert abs(e.mean()) < 1e-2
    assert data[0].force.shape == data[0].pos.shape
