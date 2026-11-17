# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Tests of the physics checks (polymer_nn.analysis) on cases with known answers.

Random walks must give nu = 1/2; a straight chain has known bond lengths,
angles, R_g and end-to-end distance; a constant energy has no drift; and a
sample compared with itself has zero distance.
"""
import numpy as np

from polymer_nn import analysis as A


def test_random_walk_exponent(rng):
    sizes = [16, 32, 64, 128, 256]
    rg = []
    for n in sizes:
        steps = rng.normal(size=(2000, n - 1, 3)) / np.sqrt(3)
        x = np.concatenate([np.zeros((2000, 1, 3)), np.cumsum(steps, axis=1)], axis=1)
        rg.append(np.sqrt(np.mean(A.radius_of_gyration(x) ** 2)))
    fit = A.fit_flory_exponent(sizes, rg)
    assert abs(fit.nu - A.NU_IDEAL) < 0.03


def test_straight_chain_geometry():
    x = np.stack([np.arange(5.0), np.zeros(5), np.zeros(5)], axis=1)
    assert np.allclose(A.bond_lengths(x), 1.0)
    assert np.allclose(A.bond_angles(x), np.pi)
    assert np.isclose(A.radius_of_gyration(x)[0], np.sqrt(2.0))
    assert np.isclose(A.end_to_end(x)[0], 4.0)


def test_energy_drift_of_constant_is_zero():
    t = np.linspace(0, 10, 100)
    d = A.energy_drift(t, np.full_like(t, 3.0), 10)
    assert abs(d["drift_per_bead_per_tau"]) < 1e-12 and d["std_per_bead"] == 0


def test_identical_distributions(rng):
    s = rng.normal(size=5000)
    c = A.compare_distributions(s, s)
    assert c["wasserstein"] == 0 and c["ks_stat"] == 0
