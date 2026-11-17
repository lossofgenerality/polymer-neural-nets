# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""The physics checks, used on the reference data and on the learned models alike.

A learned force field is only useful if a simulation driven by it reproduces
the physics of the system it imitates. Small errors in the forces could add
up over a long simulation and make the chain too compact, too swollen,
or unstable. So the networks are judged not only by their force error on
held-out frames (``train.evaluate``) but by the three checks of the project
brief, computed here:

1. **Scaling of the chain size.** The radius of gyration

       R_g² = (1/N) Σ_i |x_i - x_cm|²,     x_cm = (1/N) Σ_i x_i,          (A1)

   measures how spread out the chain is. For long chains it follows a power
   law R_g ∝ N^ν (theory notebook, section 10): ν = 1/2 for an ideal chain and
   ν ≈ 0.588 for a self-avoiding chain [clisby2010]. :func:`fit_flory_exponent`
   fits ν as the slope of log R_g against log N.

2. **Local structure.** The distributions of bond lengths |x_{i+1} - x_i|
   and bond angles θ_i (the angle at bead i between the bonds to i-1 and
   i+1; θ = π for a straight chain) must match the reference.
   :func:`compare_distributions` measures the difference with the
   Wasserstein-1 distance (the area between the two cumulative
   distributions, in the units of the variable) and the Kolmogorov–Smirnov
   statistic (the largest vertical gap between them).

3. **Energy conservation.** Without a thermostat (NVE dynamics) the total
   energy H = K + E must stay constant. :func:`energy_drift` fits a straight
   line to H(t) and reports its slope per bead (the *drift*, which should be
   ≈ 0) and the size of the fluctuations (which shrink as Δt² for a good
   integrator, theory notebook, section 9).

All functions accept a single configuration (N, 3) or a stack of frames
(F, N, 3) and work in reduced units (σ, ε, τ).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

#: Flory exponent of an ideal (random-walk) chain.
NU_IDEAL = 0.5
#: Flory exponent of a 3D self-avoiding walk, best estimate of Clisby (2010).
NU_SAW = 0.587597


def _frames(positions) -> np.ndarray:
    """Make sure the array is (F, N, 3), in float64."""
    x = np.asarray(positions, dtype=np.float64)
    return x[None] if x.ndim == 2 else x


def radius_of_gyration(positions: np.ndarray) -> np.ndarray:
    """R_g of each frame, Eq. (A1). Input (N, 3) or (F, N, 3); output (F,)."""
    x = _frames(positions)
    com = x.mean(axis=1, keepdims=True)  # centre of mass (unit masses)
    return np.sqrt(((x - com) ** 2).sum(axis=-1).mean(axis=-1))


def end_to_end(positions: np.ndarray) -> np.ndarray:
    """End-to-end distance R_e = |x_N - x_1| of each frame."""
    x = _frames(positions)
    return np.linalg.norm(x[:, -1] - x[:, 0], axis=-1)


@dataclass
class ScalingFit:
    """Result of fitting R_g = b N^ν: the exponent ν, its standard error and b."""

    nu: float
    nu_err: float
    prefactor: float

    def __str__(self) -> str:
        return f"nu = {self.nu:.3f} ± {self.nu_err:.3f}"


def fit_flory_exponent(n_beads, rg_mean) -> ScalingFit:
    """Fit log R_g = log b + ν log N by least squares and return ν ± error and b.

    ``rg_mean`` should be the root-mean-square ⟨R_g²⟩^{1/2} at each N. With
    only a few short chains the fitted ν is an *effective* exponent: it
    includes corrections to scaling, so compare models and reference at the
    same N rather than with the asymptotic value alone.
    """
    logn = np.log(np.asarray(n_beads, dtype=float))
    logr = np.log(np.asarray(rg_mean, dtype=float))
    res = stats.linregress(logn, logr)
    return ScalingFit(nu=float(res.slope), nu_err=float(res.stderr), prefactor=float(np.exp(res.intercept)))


def bond_lengths(positions: np.ndarray) -> np.ndarray:
    """All bond lengths |x_{i+1} - x_i| of all frames, as one flat array."""
    x = _frames(positions)
    return np.linalg.norm(np.diff(x, axis=1), axis=-1).ravel()


def bond_angles(positions: np.ndarray) -> np.ndarray:
    """All bond angles θ_i in radians, as one flat array (π = straight).

    θ_i is the angle between u = x_{i-1} - x_i and v = x_{i+1} - x_i:
    cos θ_i = u·v / (|u| |v|). In a Kremer–Grest chain the WCA repulsion
    between beads i-1 and i+1 forbids θ below about 60°.
    """
    x = _frames(positions)
    b = np.diff(x, axis=1)  # bond vectors b_i = x_{i+1} - x_i
    u, v = -b[:, :-1], b[:, 1:]
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.arccos(np.clip(cos, -1.0, 1.0)).ravel()


def compare_distributions(sample: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """How different two samples of a scalar quantity are.

    Returns the Wasserstein-1 distance W₁ = ∫ |C_sample(t) - C_ref(t)| dt
    (C the cumulative distributions; same unit as the data), the
    Kolmogorov–Smirnov statistic sup_t |C_sample - C_ref| (between 0 and 1),
    the difference of the means and the ratio of the standard deviations.
    """
    ks = stats.ks_2samp(sample, reference)
    return {
        "wasserstein": float(stats.wasserstein_distance(sample, reference)),
        "ks_stat": float(ks.statistic),
        "mean_diff": float(np.mean(sample) - np.mean(reference)),
        "std_ratio": float(np.std(sample) / np.std(reference)),
    }


def energy_drift(time: np.ndarray, total_energy: np.ndarray, n_beads: int) -> dict[str, float]:
    """Drift and fluctuation of the total energy H(t) of an NVE run, per bead.

    * ``drift_per_bead_per_tau``: slope of the least-squares line through H(t),
      divided by N. A conservative force field integrated with velocity
      Verlet has no systematic drift; a non-conservative one heats up or
      cools down.
    * ``std_per_bead``: standard deviation of H, divided by N; for Verlet it is
      proportional to Δt² (theory notebook, Eq. (9.2)).
    * ``max_abs_dev_per_bead``: largest |H(t) - H(0)|, divided by N.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(total_energy, dtype=float)
    slope = stats.linregress(t, e).slope
    return {
        "drift_per_bead_per_tau": float(slope / n_beads),
        "std_per_bead": float(np.std(e) / n_beads),
        "max_abs_dev_per_bead": float(np.max(np.abs(e - e[0])) / n_beads),
    }


def kinetic_energy(velocities: np.ndarray, masses: float | np.ndarray = 1.0) -> np.ndarray:
    """K = Σ_i ½ m_i |v_i|² of each frame. At equilibrium ⟨K⟩ = (3/2) N k_B T."""
    v = _frames(velocities)
    m = np.asarray(masses, dtype=float)
    m = m[..., None] if m.ndim else m
    return 0.5 * (m * v**2).sum(axis=(-1, -2))
