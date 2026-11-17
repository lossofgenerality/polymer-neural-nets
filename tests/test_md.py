# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Tests of the molecular-dynamics drivers (polymer_nn.md).

Velocity Verlet with the exact forces conserves the total energy; the OpenMM
bridge (CustomExternalForce refreshed every step), driven by exact forces,
produces a stable, energy-conserving trajectory with the right bond lengths.
"""
import numpy as np

from polymer_nn import analysis as A
from polymer_nn.md import OpenMMLearnedSimulation, reference_potential, velocity_verlet


class RefPot:
    def __init__(self, n):
        self.n = n
        self.f = reference_potential(n)

    def __call__(self, x):
        return self.f(x)


def test_reference_nve_conserves_energy(small_dataset):
    x0 = small_dataset["positions"][-1].astype(np.float64)
    r = velocity_verlet(reference_potential(12), x0, n_steps=500, dt=0.002)
    d = A.energy_drift(r["time"], r["total"], 12)
    assert d["max_abs_dev_per_bead"] < 5e-3


def test_openmm_bridge_matches_reference_dynamics(small_dataset):
    """The CustomExternalForce bridge driven by exact forces must reproduce exact Verlet."""
    x0 = small_dataset["positions"][-1].astype(np.float64)
    sim = OpenMMLearnedSimulation(RefPot(12), x0, integrator="verlet", dt=0.002)
    out = sim.run(20, stride=5)
    assert np.isfinite(out["positions"]).all()
    assert 0.9 < A.bond_lengths(out["positions"]).mean() < 1.05
    tot = out["potential"] + out["kinetic"]
    assert np.std(tot) / 12 < 0.05
