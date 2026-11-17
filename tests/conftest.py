# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Shared fixtures: a small Kremer-Grest dataset (12 beads, 60 frames) made
once per test session with OpenMM, and a seeded random generator.
"""
import numpy as np
import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture(scope="session")
def small_dataset():
    from polymer_nn.sim import ChainSpec, sample

    return sample(ChainSpec(n_beads=12, model="kg", seed=3), n_frames=60, stride=50, equilibrate=2000)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q
