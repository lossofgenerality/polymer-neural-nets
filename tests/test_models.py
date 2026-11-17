# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Tests of the symmetries and the calculus of both networks (polymer_nn.models).

For random weights, in double precision: the energy is invariant under
rotations and translations and the forces rotate with the chain (theory
notebook, Theorems 4, 5, 7); reading the chain backwards changes nothing
(Theorem 3); the forces equal minus the finite-difference gradient of the
energy (Eq. (7.1)); and the energy is continuous when a neighbour crosses
the cutoff (Theorem 6).
"""
import numpy as np
import pytest
import torch

from polymer_nn.graph import to_data
from polymer_nn.models import build_model
from tests.conftest import random_rotation


@pytest.fixture(params=["invariant", "egnn"])
def model(request):
    torch.manual_seed(0)
    return build_model(request.param, hidden=16, layers=2).double().eval()


def _ef(model, x):
    d = to_data(x, cutoff=model.cutoff)
    d.pos = torch.tensor(x, dtype=torch.float64)
    e, f = model(d)
    return float(e.detach()), f.detach().numpy()


def test_rotation_translation(model, small_dataset, rng):
    x = small_dataset["positions"][5].astype(np.float64)
    e0, f0 = _ef(model, x)
    Q = random_rotation(rng)
    e1, f1 = _ef(model, x @ Q.T + rng.normal(size=3) * 5)
    assert abs(e0 - e1) < 1e-8 * max(1, abs(e0))
    assert np.allclose(f0 @ Q.T, f1, atol=1e-7)


def test_reversal_symmetry(model, small_dataset):
    x = small_dataset["positions"][2].astype(np.float64)
    e0, f0 = _ef(model, x)
    e1, f1 = _ef(model, x[::-1].copy())
    assert abs(e0 - e1) < 1e-8 * max(1, abs(e0))
    assert np.allclose(f0[::-1], f1, atol=1e-7)


def test_forces_are_gradient(model, small_dataset):
    x = small_dataset["positions"][7].astype(np.float64)
    _, f = _ef(model, x)
    h = 1e-6
    for i, k in [(0, 0), (5, 1), (11, 2)]:
        xp, xm = x.copy(), x.copy()
        xp[i, k] += h
        xm[i, k] -= h
        num = -(_ef(model, xp)[0] - _ef(model, xm)[0]) / (2 * h)
        assert abs(num - f[i, k]) < 1e-5


def test_energy_continuous_at_cutoff(model):
    # two short chains far apart along the backbone; move one end through the cutoff
    base = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    es = []
    for gap in np.linspace(model.cutoff - 1e-4, model.cutoff + 1e-4, 5):
        x = base.copy()
        x[5] = [0, 0, gap]  # bead 5 approaches bead 0 (non-bonded)
        x[4] = [1, 0, gap]
        x[3] = [2, 0, gap]
        es.append(_ef(model, x)[0])
    assert np.max(np.abs(np.diff(es))) < 1e-5
