# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Tests of the reference simulations (polymer_nn.sim).

The data are only as good as the model that made them, so these tests check
the OpenMM system itself: the right forces are present, the Kremer-Grest bond
length comes out at its known value of about 0.97 sigma, the stored forces are
minus the gradient of the energy (checked by finite differences), and pivot
Monte Carlo keeps bond lengths fixed and swells a chain with excluded volume.
"""
import numpy as np

from polymer_nn.analysis import bond_lengths
from polymer_nn.sim import build_system, initial_positions, reference_energy_forces


def test_system_has_expected_forces():
    assert build_system(10, "kg").getNumForces() == 2
    assert build_system(10, "ideal").getNumForces() == 1


def test_kremer_grest_bond_length(small_dataset):
    # Kremer-Grest chains have a mean bond length close to 0.97 sigma
    assert 0.94 < bond_lengths(small_dataset["positions"]).mean() < 1.0


def test_forces_are_minus_gradient(rng):
    x = initial_positions(8, rng)
    e0, f = reference_energy_forces(x)
    h = 1e-5
    for i, d in [(0, 0), (3, 1), (7, 2)]:
        xp, xm = x.copy(), x.copy()
        xp[i, d] += h
        xm[i, d] -= h
        num = -(reference_energy_forces(xp)[0] - reference_energy_forces(xm)[0]) / (2 * h)
        assert abs(num - f[i, d]) < 1e-3 * max(1.0, abs(f[i, d]))


def test_pivot_mc_keeps_bonds_and_swells(rng):
    from polymer_nn.analysis import radius_of_gyration
    from polymer_nn.sim import pivot_mc

    x = initial_positions(24, rng)
    b0 = bond_lengths(x)
    y, tr_saw = pivot_mc(x, 600, rng, record_every=5)
    assert np.allclose(bond_lengths(y), b0)
    _, tr_ideal = pivot_mc(x, 600, rng, excluded_volume=False, record_every=5)
    assert np.mean(tr_saw[30:] ** 2) > np.mean(tr_ideal[30:] ** 2)
    assert np.isclose(radius_of_gyration(y)[0], tr_saw[-1], rtol=0.5)
