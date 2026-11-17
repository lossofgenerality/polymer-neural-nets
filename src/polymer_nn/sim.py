# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Reference simulations of bead-spring polymers with OpenMM.

This module produces **all** the data used in this repository. Nothing is
downloaded: every configuration, force and energy that the neural networks
learn from is computed here, by molecular dynamics (MD) with the open-source
engine OpenMM [eastman2017, eastman2024], from a physical model whose
equations are written below. The notebook *The Data and the Graph* (folder
``01-data-and-graph``) walks through the files that come out of it.

The physical model
==================

A polymer is a long chain of repeating units. In a *bead-spring* model each
bead stands for a few chemical units, and only two things matter: beads next
to each other along the chain are held together by a spring, and no two beads
may overlap. Two chain models are provided.

**Kremer–Grest chain** (``model="kg"``) [kremer1990]. Bonded neighbours
i, i+1 interact through the FENE (finitely extensible nonlinear elastic)
spring

    U_FENE(r) = -½ k R₀² ln(1 - (r/R₀)²),     k = 30 ε/σ²,  R₀ = 1.5 σ,     (S1)

which is harmonic for small r and diverges as r → R₀, so a bond can never
stretch beyond 1.5 σ and two chains can never pass through each other. Every
pair of beads, bonded or not, repels through the Weeks–Chandler–Andersen
(WCA) potential [weeks1971], the Lennard-Jones potential cut at its minimum
and shifted up so that it is zero there:

    U_WCA(r) = 4ε[(σ/r)¹² - (σ/r)⁶] + ε   for r < 2^{1/6} σ,   0 otherwise.   (S2)

FENE and WCA together give a mean bond length of 0.97 σ. Because beads
cannot overlap, the chain is a *self-avoiding walk*: it swells, and its size
grows as R_g ∝ N^ν with ν ≈ 0.588 [clisby2010] (theory notebook, section 10).

**Ideal chain** (``model="ideal"``). Harmonic bonds and nothing else:

    U_bond(r) = ½ k (r - r₀)²,     k = 100 ε/σ²,  r₀ = 1 σ.                 (S3)

Beads far apart along the chain do not see each other, so the chain is a
random walk with ν = 1/2 exactly. It is the control case for the R_g check.

The total potential energy of a configuration x = (x₁, …, x_N) ∈ ℝ^{3N} is

    E(x) = Σ_{bonds (i,i+1)} U_bond(|x_{i+1} - x_i|) + Σ_{i<j} U_WCA(|x_j - x_i|),   (S4)

and the force on bead i is F_i = -∂E/∂x_i. These two quantities, E and F,
are the *targets* that the networks learn.

Units
=====

Everything is in reduced Lennard-Jones units: length σ (the bead diameter),
energy ε, mass m. OpenMM expects nm, kJ/mol and amu, so we set σ = 1 nm,
ε = 1 kJ/mol and m = 1 amu. The time unit is then τ = σ √(m/ε) = 1 ps, and the
thermal energy k_B T = 1 ε corresponds to T = 1/k_B ≈ 120.27 K in OpenMM's
input. All numbers written by this module are therefore directly in σ, ε, τ.

How the data is made
====================

1. Build the OpenMM ``System`` (particles, bonds, non-bonded force) with
   :func:`build_system`.
2. Start from a random walk (:func:`initial_positions`), optionally relaxed
   by pivot Monte Carlo (:func:`pivot_mc`), and remove overlaps with an
   energy minimisation.
3. Run Langevin dynamics, ``LangevinMiddleIntegrator`` [zhang2019], at
   k_B T = 1 ε with friction γ and time step Δt. The thermostat makes the
   positions sample the Boltzmann distribution p(x) ∝ exp(-E(x)/k_B T).
4. After an equilibration period, record a *frame* every ``stride`` steps:
   the positions, the forces and the potential energy (:func:`sample`).
   Frames are separated by many steps so that they are nearly independent.

One frame is one training example: the input is the N × 3 array of positions,
the targets are the N × 3 forces and the scalar energy.

Sources (citation keys of ``polymer_nn.references``): kremer1990, weeks1971,
clisby2010, zhang2019, eastman2017, eastman2024, rubinstein2003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openmm
from openmm import unit

#: Boltzmann constant in kJ/(mol K). k_B T = 1 kJ/mol means T = 1/K_B kelvin.
K_B = 0.0083144626181532
#: WCA cutoff 2^(1/6) sigma, the minimum of the Lennard-Jones potential, Eq. (S2).
WCA_CUTOFF = 2.0 ** (1.0 / 6.0)

#: FENE spring constant k (epsilon/sigma^2) and maximum extension R0 (sigma), Eq. (S1).
FENE_K = 30.0
FENE_R0 = 1.5
#: Harmonic bond of the ideal chain, Eq. (S3).
HARMONIC_K = 100.0
HARMONIC_R0 = 1.0


@dataclass
class ChainSpec:
    """Everything that defines one simulation.

    Attributes
    ----------
    n_beads : number of beads N in the chain.
    model : ``"kg"`` (Kremer–Grest, self-avoiding) or ``"ideal"`` (random walk).
    kT : thermal energy k_B T in epsilon (1 throughout this repository).
    friction : Langevin friction γ in 1/τ. It changes how fast the chain
        moves, not the equilibrium distribution, so a small γ (0.1) is used
        when only static averages such as R_g are wanted.
    dt : integration time step Δt in τ.
    seed : random seed (starting walk, velocities and thermostat noise).
    platform : OpenMM platform, ``"CPU"``, ``"Reference"``, ``"CUDA"``,
        ``"OpenCL"`` or ``"Metal"``.
    """

    n_beads: int = 32
    model: str = "kg"
    kT: float = 1.0
    friction: float = 1.0
    dt: float = 0.005
    seed: int = 0
    platform: str = "CPU"


def _add_chain_forces(system: openmm.System, model: str, chains: list[list[int]]) -> None:
    """Add the bonded and non-bonded forces of Eqs. (S1)-(S3) for the given chains.

    ``chains`` lists the particle indices of each chain, in order along the
    chain. Several chains can live in one System (see :func:`sample_replicas`).
    """
    n_total = system.getNumParticles()
    if model == "kg":
        # Eq. (S1): OpenMM evaluates the algebraic expression for every bond.
        fene = openmm.CustomBondForce("-0.5*k*r0^2*log(1-(r/r0)^2)")
        fene.addGlobalParameter("k", FENE_K)
        fene.addGlobalParameter("r0", FENE_R0)
        for chain in chains:
            for i, j in zip(chain[:-1], chain[1:]):
                fene.addBond(i, j, [])
        fene.setForceGroup(0)
        system.addForce(fene)

        # Eq. (S2) between *every* pair (bonded pairs included, as in
        # Kremer-Grest). step(rc - r) is 1 inside the cutoff and 0 outside.
        wca = openmm.CustomNonbondedForce("step(rc-r)*(4*((1/r)^12-(1/r)^6)+1)")
        wca.addGlobalParameter("rc", WCA_CUTOFF)
        for _ in range(n_total):
            wca.addParticle([])
        # no periodic box: a single chain in implicit good solvent
        wca.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        wca.setCutoffDistance(WCA_CUTOFF)
        wca.setForceGroup(1)
        system.addForce(wca)
    elif model == "ideal":
        # Eq. (S3); OpenMM's HarmonicBondForce uses E = 1/2 k (r - r0)^2
        harm = openmm.HarmonicBondForce()
        for chain in chains:
            for i, j in zip(chain[:-1], chain[1:]):
                harm.addBond(i, j, HARMONIC_R0, HARMONIC_K)
        system.addForce(harm)
    else:
        raise ValueError(f"unknown model {model!r}; use 'kg' or 'ideal'")


def build_system(n_beads: int, model: str = "kg") -> openmm.System:
    """OpenMM ``System`` for one linear chain of ``n_beads`` beads of unit mass.

    Force group 0 holds the bonds, group 1 the WCA repulsion, so that the two
    contributions to the energy can be read separately with
    ``context.getState(getEnergy=True, groups={1})``.
    """
    system = openmm.System()
    for _ in range(n_beads):
        system.addParticle(1.0)  # mass 1 amu = 1 m
    _add_chain_forces(system, model, [list(range(n_beads))])
    return system


def initial_positions(n_beads: int, rng: np.random.Generator, bond: float = 0.97) -> np.ndarray:
    """A random starting walk with steps of length ``bond`` and no overlaps.

    Each new bead is placed in a random direction from the previous one; a
    direction is rejected (and drawn again, up to 1000 times) when the new bead
    would come closer than 0.9 σ to any earlier bead. The walk is centred at
    the origin. It is only a starting point: the dynamics (or pivot Monte
    Carlo) then relax it to the equilibrium shape.
    """
    pos = np.zeros((n_beads, 3))
    for i in range(1, n_beads):
        for _ in range(1000):
            step = rng.normal(size=3)
            step *= bond / np.linalg.norm(step)  # random unit vector times the bond length
            trial = pos[i - 1] + step
            if i < 2 or np.min(np.linalg.norm(pos[: i - 1] - trial, axis=1)) > 0.9:
                break
        pos[i] = trial
    return pos - pos.mean(axis=0)


def _integrator(spec: ChainSpec) -> openmm.LangevinMiddleIntegrator:
    """Langevin thermostat at k_B T = spec.kT, Eq. (9.3) of the theory notebook (00-theory)."""
    integ = openmm.LangevinMiddleIntegrator(
        spec.kT / K_B * unit.kelvin, spec.friction / unit.picosecond, spec.dt * unit.picosecond
    )
    integ.setRandomNumberSeed(spec.seed + 1)
    return integ


def _context(system, integrator, platform: str) -> openmm.Context:
    props = {"Threads": "1"} if platform == "CPU" else {}
    return openmm.Context(system, integrator, openmm.Platform.getPlatformByName(platform), props)


def make_context(spec: ChainSpec, positions: np.ndarray | None = None):
    """An OpenMM ``Context`` (system + integrator + state) ready to run.

    Returns ``(context, integrator)``. Positions default to
    :func:`initial_positions`; velocities are drawn from the Maxwell–Boltzmann
    distribution at k_B T.
    """
    system = build_system(spec.n_beads, spec.model)
    integrator = _integrator(spec)
    context = _context(system, integrator, spec.platform)
    if positions is None:
        positions = initial_positions(spec.n_beads, np.random.default_rng(spec.seed))
    context.setPositions(positions * unit.nanometer)
    context.setVelocitiesToTemperature(spec.kT / K_B * unit.kelvin, spec.seed + 2)
    return context, integrator


def evaluate(context) -> tuple[np.ndarray, np.ndarray, float]:
    """Read positions (σ), forces (ε/σ) and potential energy (ε) from a Context.

    OpenMM returns quantities with units attached; ``value_in_unit`` strips
    them, and because σ = 1 nm and ε = 1 kJ/mol the numbers are already in
    reduced units.
    """
    state = context.getState(getPositions=True, getForces=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    frc = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return np.asarray(pos), np.asarray(frc), float(e)


def sample(spec: ChainSpec, n_frames: int = 1000, stride: int = 200, equilibrate: int = 20000) -> dict[str, np.ndarray]:
    """Run Langevin dynamics for one chain and record ``n_frames`` frames.

    Procedure: energy minimisation (removes overlaps of the starting walk),
    ``equilibrate`` steps that are thrown away, then ``n_frames`` × ``stride``
    steps with a frame recorded after every ``stride`` steps.

    Returns a dict, ready for ``np.savez``, with

    ===============  ==============  =========  ======================================
    key              shape           unit       meaning
    ===============  ==============  =========  ======================================
    ``positions``    (F, N, 3)       σ          bead coordinates x_i of each frame
    ``forces``       (F, N, 3)       ε/σ        exact forces F_i = -∂E/∂x_i, Eq. (S4)
    ``energy``       (F,)            ε          total potential energy E(x), Eq. (S4)
    ``n_beads``      scalar          –          N
    ``model``        scalar string   –          "kg" or "ideal"
    ``kT``           scalar          ε          k_B T
    ===============  ==============  =========  ======================================

    F is the number of frames and N the number of beads.
    """
    context, integrator = make_context(spec)
    openmm.LocalEnergyMinimizer.minimize(context, 1e-3, 500)
    integrator.step(equilibrate)
    P = np.empty((n_frames, spec.n_beads, 3))
    F = np.empty_like(P)
    E = np.empty(n_frames)
    for k in range(n_frames):
        integrator.step(stride)
        P[k], F[k], E[k] = evaluate(context)
    return {
        "positions": P.astype(np.float32),
        "forces": F.astype(np.float32),
        "energy": E.astype(np.float64),
        "n_beads": np.int64(spec.n_beads),
        "model": np.str_(spec.model),
        "kT": np.float64(spec.kT),
    }


# ----------------------------------------------------------------------------
# Pivot Monte Carlo: an independent sampler for the shape of the chain
# ----------------------------------------------------------------------------


def _wca_energy(x: np.ndarray) -> float:
    """Total WCA energy, Eq. (S2), of the pairs |i - j| >= 2.

    The pairs i, i+1 are left out because a pivot move never changes a bond
    length, so their energy is the same before and after the move and cancels
    in the Metropolis ratio.
    """
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(x), 2)
    r = d[iu]
    r = r[r < WCA_CUTOFF]
    if r.size == 0:
        return 0.0
    rr = (1.0 / r) ** 6
    return float((4 * (rr * rr - rr) + 1).sum())


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random rotation matrix (Haar measure on SO(3)).

    The QR decomposition of a matrix of Gaussian numbers gives a uniformly
    random orthogonal Q once the signs of the diagonal of R are moved into Q;
    flipping one column when det Q = -1 maps it to a rotation.
    """
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def pivot_mc(x: np.ndarray, n_moves: int, rng: np.random.Generator, kT: float = 1.0,
             excluded_volume: bool = True, record_every: int = 0):
    """Pivot Monte Carlo for an off-lattice chain [Lal 1969; Madras & Sokal 1988].

    One move: pick a bead k at random, pick a random rotation R, and rotate
    one end of the chain (the beads after k, or the beads before k) rigidly
    about x_k:

        x_j ← x_k + R (x_j - x_k)   for the beads of that end.

    The move is accepted with the Metropolis probability
    min(1, exp(-ΔE_WCA / k_B T)). A rotation and its inverse are equally
    likely, so the proposal is symmetric and the chain samples the Boltzmann
    distribution exp(-E/k_B T) of Eq. (S4) for its bond lengths (Theory,
    Theorem 11). Bond lengths, and so the FENE energies, never change.

    Why it is useful: a single accepted pivot changes the overall shape of
    the chain, so R_g decorrelates in a few accepted moves, independently of
    how slowly the MD dynamics would relax. It is used here (a) to relax
    starting walks before MD and (b) as an independent check of MD averages.

    Returns the final configuration, and, if ``record_every`` > 0, also the
    R_g after every ``record_every`` moves.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    e = _wca_energy(x) if excluded_volume else 0.0
    trace = []
    for t in range(n_moves):
        k = int(rng.integers(1, n - 1))  # pivot bead (not an end bead)
        R = _random_rotation(rng)
        y = x.copy()
        if rng.random() < 0.5:
            y[k + 1 :] = (x[k + 1 :] - x[k]) @ R.T + x[k]  # rotate the tail
        else:
            y[:k] = (x[:k] - x[k]) @ R.T + x[k]  # rotate the head
        e_new = _wca_energy(y) if excluded_volume else 0.0
        if e_new <= e or rng.random() < np.exp(-(e_new - e) / kT):  # Metropolis
            x, e = y, e_new
        if record_every and t % record_every == 0:
            trace.append(np.sqrt(((x - x.mean(0)) ** 2).sum(1).mean()))
    return (x, np.array(trace)) if record_every else x


def sample_replicas(spec: ChainSpec, n_replicas: int = 16, n_frames: int = 400, stride: int = 1000,
                    equilibrate: int = 200000, spacing: float = 1000.0, pivot_moves: int = 0) -> dict[str, np.ndarray]:
    """Simulate ``n_replicas`` independent copies of the chain in one OpenMM Context.

    The copies sit ``spacing`` σ apart, with no periodic box, so they never
    interact: each is an independent sample. For static averages such as R_g
    this is better than one long run, because averages over independent
    chains converge as 1/√(number of chains) whatever the relaxation time.

    ``pivot_moves`` > 0 first relaxes each starting walk with :func:`pivot_mc`
    (excluded volume for ``"kg"``), which shortens the MD equilibration of
    long chains.

    Returns ``positions`` with shape (F, R, N, 3) (frames, replicas, beads,
    xyz) and ``final``, the last configuration of each replica, (R, N, 3).
    """
    n = spec.n_beads
    system = openmm.System()
    for _ in range(n * n_replicas):
        system.addParticle(1.0)
    _add_chain_forces(system, spec.model, [list(range(r * n, (r + 1) * n)) for r in range(n_replicas)])

    rng = np.random.default_rng(spec.seed)
    side = int(np.ceil(n_replicas ** (1 / 3)))  # replicas on a side × side × side grid
    pos = []
    for r in range(n_replicas):
        offset = spacing * np.array([r % side, (r // side) % side, r // side**2], dtype=float)
        x0 = initial_positions(n, rng)
        if pivot_moves:
            x0 = pivot_mc(x0, pivot_moves, rng, spec.kT, excluded_volume=spec.model == "kg")
        pos.append(x0 + offset)
    pos = np.concatenate(pos)

    integrator = _integrator(spec)
    context = _context(system, integrator, spec.platform)
    context.setPositions(pos * unit.nanometer)
    openmm.LocalEnergyMinimizer.minimize(context, 1e-3, 500)
    context.setVelocitiesToTemperature(spec.kT / K_B * unit.kelvin, spec.seed + 2)
    integrator.step(equilibrate)
    frames = np.empty((n_frames, n_replicas, n, 3), dtype=np.float32)
    for k in range(n_frames):
        integrator.step(stride)
        x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        frames[k] = np.asarray(x).reshape(n_replicas, n, 3)
    return {"positions": frames, "final": frames[-1].copy()}


def reference_energy_forces(positions: np.ndarray, model: str = "kg"):
    """Exact energy (ε) and forces (ε/σ) of one configuration, Eq. (S4).

    Uses OpenMM's double-precision ``Reference`` platform, so it is the
    ground truth against which finite differences and the networks are
    checked.
    """
    n = positions.shape[0]
    ctx = openmm.Context(build_system(n, model), openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions * unit.nanometer)
    _, f, e = evaluate(ctx)
    return e, f
