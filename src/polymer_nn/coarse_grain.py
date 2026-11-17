# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Coarse-graining: fewer, bigger beads, for very long polymers.

To reach chains of 10⁴–10⁶ beads, several fine beads are merged into one
coarse-grained (CG) site, and a network learns the forces between the sites.
This module holds the bookkeeping: the mapping and the mapped forces.

The mapping
===========

With m consecutive fine beads per site and equal masses, site I sits at the
centre of mass of its beads. Written as a matrix M of size n_CG × n,

    X_I = Σ_j M_Ij x_j,     M_Ij = 1/m if bead j belongs to site I, else 0.     (C1)

Mapped forces
=============

The force-matching target on site I is the total fine-grained force on its
beads (for equal masses),

    F_I = Σ_{j ∈ I} f_j.                                                     (C2)

The multiscale coarse-graining theorem [izvekov2005, noid2008] says that the
function of X that minimises the mean squared distance to these mapped
forces is the *mean force* -∇W(X), the gradient of the potential of mean
force W (a free energy). So the same networks and the same loss
(``train.fit``) learn W from mapped data, with ``weight_energy=0`` because
the fine-grained energy is not the CG free energy (theory notebook,
Theorem 9). This is the CGnet idea [wang2019, husic2020].
"""

from __future__ import annotations

import numpy as np


def mapping_matrix(n_fine: int, beads_per_site: int) -> np.ndarray:
    """The matrix M of Eq. (C1), shape (n_CG, n_fine), for equal masses."""
    if n_fine % beads_per_site:
        raise ValueError("n_fine must be a multiple of beads_per_site")
    n_cg = n_fine // beads_per_site
    M = np.zeros((n_cg, n_fine))
    for I in range(n_cg):
        M[I, I * beads_per_site : (I + 1) * beads_per_site] = 1.0 / beads_per_site
    return M


def map_positions(positions: np.ndarray, M: np.ndarray) -> np.ndarray:
    """CG site positions X = M x, Eq. (C1), for (N, 3) or (F, N, 3) input."""
    return np.einsum("In,...nd->...Id", M, positions)


def map_forces(forces: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Force-matching targets F_I = Σ_{j∈I} f_j, Eq. (C2)."""
    S = (M > 0).astype(float)  # 1 where bead j is in site I
    return np.einsum("In,...nd->...Id", S, forces)


def coarse_grain_dataset(d: dict, beads_per_site: int) -> dict:
    """Map a dataset of ``sim.sample`` to CG resolution, in the same file format.

    The energy entry is kept only for bookkeeping: train CG models on forces
    alone (``weight_energy=0``).
    """
    n = int(d["positions"].shape[1])
    M = mapping_matrix(n, beads_per_site)
    return {
        "positions": map_positions(d["positions"], M).astype(np.float32),
        "forces": map_forces(d["forces"], M).astype(np.float32),
        "energy": np.asarray(d["energy"]),
        "n_beads": np.int64(n // beads_per_site),
        "model": np.str_(f"cg{beads_per_site}-" + str(d["model"])),
        "kT": d["kT"],
    }
