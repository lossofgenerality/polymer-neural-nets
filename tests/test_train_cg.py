# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Tests of training (polymer_nn.train) and coarse-graining (polymer_nn.coarse_grain).

A few epochs of force matching must reduce the force error clearly; a saved
and reloaded model must give the same errors; the CG mapping must average
positions and sum forces.
"""
import numpy as np
import torch

from polymer_nn.coarse_grain import coarse_grain_dataset, map_forces, mapping_matrix
from polymer_nn.graph import dataset_from_npz
from polymer_nn.models import build_model, load_model, save_model
from polymer_nn.train import TrainConfig, evaluate, fit


def test_training_reduces_force_error(small_dataset, tmp_path):
    data, off = dataset_from_npz(small_dataset)
    torch.manual_seed(0)
    model = build_model("invariant", hidden=16, layers=2)
    before = evaluate(model, data[:10])["force_rmse"]
    fit(model, data[10:], data[:10], TrainConfig(epochs=4, batch_size=8, lr=3e-3), verbose=False)
    after = evaluate(model, data[:10])["force_rmse"]
    assert after < 0.7 * before
    save_model(model, tmp_path / "m.pt", "invariant", dict(hidden=16, layers=2, cutoff=1.5), off)
    m2, ck = load_model(tmp_path / "m.pt")
    assert abs(evaluate(m2, data[:10])["force_rmse"] - after) < 1e-5 and ck["energy_offset"] == off


def test_cg_mapping(small_dataset):
    M = mapping_matrix(12, 3)
    assert np.allclose(M.sum(axis=1), 1.0)
    f = small_dataset["forces"]
    assert np.allclose(map_forces(f, M).sum(axis=1), f.sum(axis=1), atol=1e-3)
    cg = coarse_grain_dataset(small_dataset, 3)
    assert cg["positions"].shape[1] == 4
