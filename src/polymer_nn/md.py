# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Molecular dynamics driven by a learned force field: the network as a simulator.

Once trained, the network replaces OpenMM's exact force field. Each MD step:

1. build the graph of the current positions (neighbours change as beads move);
2. run the network to get E(x) and F = -∂E/∂x;
3. move the beads with an integrator, using those forces.

Three integrators are provided (theory notebook, section 9).

* :class:`OpenMMLearnedSimulation`: OpenMM does the dynamics with its own
  integrators (``LangevinMiddleIntegrator`` or ``VerletIntegrator``). The
  network forces enter through a ``CustomExternalForce`` with energy
  -(f_x x + f_y y + f_z z) per bead: its force is exactly (f_x, f_y, f_z),
  and before every step these per-bead parameters are set to the network's
  forces. This needs only the pip ``openmm`` wheel. For GPU production runs
  the model can instead be compiled and loaded with ``openmm-torch``
  [eastman2024] (see ``docs/running-large.md``).
* :func:`velocity_verlet`: plain NVE velocity Verlet [swope1982] in NumPy,

      v ← v + ½ Δt F(x);   x ← x + Δt v;   v ← v + ½ Δt F(x),             (D1)

  used for the energy-conservation check (unit masses).
* :func:`langevin_batched`: BAOAB Langevin dynamics in PyTorch for many
  independent chains at once (one batched graph), used to collect R_g
  statistics with the learned forces:

      B: v ← v + ½ Δt F;  A: x ← x + ½ Δt v;  O: v ← c₁ v + c₂ ξ;
      A: x ← x + ½ Δt v;  B: v ← v + ½ Δt F(x),                           (D2)

  with c₁ = exp(-γ Δt), c₂ = √((1 - c₁²) k_B T) and ξ standard normal noise.
  The O step is the exact solution of the friction-plus-noise part, so the
  scheme samples the Boltzmann distribution exp(-E/k_B T) with a
  configurational error of order Δt².
"""

from __future__ import annotations

import numpy as np
import openmm
import torch
from openmm import unit
from torch_geometric.data import Data

from .analysis import kinetic_energy
from .graph import chain_bonds, chain_graph, node_types
from .sim import K_B


class LearnedPotential:
    """A trained model as a function ``positions (N, 3) -> (energy, forces)``.

    The energy offset that was removed during training (``graph.to_data``) is
    added back, so the returned energy is on the same scale as OpenMM's.
    """

    def __init__(self, model, n_beads: int, energy_offset: float = 0.0, device="cpu"):
        self.model = model.to(device).eval()
        self.n = n_beads
        self.offset = energy_offset
        self.device = device
        self.z = node_types(n_beads).to(device)
        self.bonds = chain_bonds(n_beads, device=device)

    def __call__(self, positions) -> tuple[float, np.ndarray]:
        pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32, device=self.device)
        ei, et = chain_graph(pos, self.bonds, self.model.cutoff)  # step 1: the graph
        data = Data(pos=pos.clone().requires_grad_(True), z=self.z, edge_index=ei, edge_type=et)
        e, f = self.model(data)  # step 2: E and F = -dE/dx
        return float(e.detach().sum()) + self.offset * self.n, f.detach().cpu().numpy().astype(np.float64)


class OpenMMLearnedSimulation:
    """An OpenMM ``Context`` whose only force is the learned one.

    Parameters: the potential, starting positions (N, 3), ``integrator``
    ``"langevin"`` (thermostat at k_B T, friction γ) or ``"verlet"`` (NVE),
    time step Δt (τ), seed and OpenMM platform.
    """

    def __init__(self, potential: LearnedPotential, positions, integrator: str = "langevin",
                 kT: float = 1.0, dt: float = 0.005, friction: float = 1.0, seed: int = 0, platform: str = "CPU"):
        self.potential = potential
        n = potential.n
        system = openmm.System()
        for _ in range(n):
            system.addParticle(1.0)
        # E = -F·x with F held fixed during the step gives force F on each bead
        self.force = openmm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        for name in ("fx", "fy", "fz"):
            self.force.addPerParticleParameter(name)
        for i in range(n):
            self.force.addParticle(i, [0.0, 0.0, 0.0])
        system.addForce(self.force)
        if integrator == "langevin":
            integ = openmm.LangevinMiddleIntegrator(kT / K_B * unit.kelvin, friction / unit.picosecond, dt * unit.picosecond)
            integ.setRandomNumberSeed(seed + 1)
        elif integrator == "verlet":
            integ = openmm.VerletIntegrator(dt * unit.picosecond)
        else:
            raise ValueError(integrator)
        self.integrator = integ
        props = {"Threads": "1"} if platform == "CPU" else {}
        self.context = openmm.Context(system, integ, openmm.Platform.getPlatformByName(platform), props)
        self.context.setPositions(np.asarray(positions) * unit.nanometer)
        self.context.setVelocitiesToTemperature(kT / K_B * unit.kelvin, seed + 2)
        self.dt = dt
        self.last_energy = None

    def positions(self) -> np.ndarray:
        s = self.context.getState(getPositions=True)
        return np.asarray(s.getPositions(asNumpy=True).value_in_unit(unit.nanometer))

    def velocities(self) -> np.ndarray:
        s = self.context.getState(getVelocities=True)
        return np.asarray(s.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond))

    def _refresh(self):
        """Evaluate the network at the current positions and hand its forces to OpenMM."""
        e, f = self.potential(self.positions())
        for i in range(self.potential.n):
            self.force.setParticleParameters(i, i, f[i].tolist())
        self.force.updateParametersInContext(self.context)
        self.last_energy = e

    def step(self, n_steps: int = 1):
        """Advance ``n_steps`` steps, with fresh network forces before each one."""
        for _ in range(n_steps):
            self._refresh()
            self.integrator.step(1)

    def run(self, n_frames: int, stride: int = 10) -> dict[str, np.ndarray]:
        """Record ``n_frames`` frames, one every ``stride`` steps.

        Returns positions (F, N, 3), potential energy (learned, ε), kinetic
        energy (ε) and time (τ) of each frame.
        """
        P, E, K, T = [], [], [], []
        for k in range(n_frames):
            self.step(stride)
            x = self.positions()
            e, _ = self.potential(x)
            P.append(x)
            E.append(e)
            K.append(float(kinetic_energy(self.velocities())[0]))
            T.append((k + 1) * stride * self.dt)
        return {"positions": np.array(P), "potential": np.array(E), "kinetic": np.array(K), "time": np.array(T)}


def velocity_verlet(potential, positions, n_steps: int = 2000, dt: float = 0.002, kT: float = 1.0,
                    seed: int = 0, record_every: int = 10) -> dict[str, np.ndarray]:
    """NVE velocity Verlet, Eq. (D1), with unit masses.

    Velocities start from the Maxwell–Boltzmann distribution at k_B T with
    the centre-of-mass motion removed. Returns time, potential, kinetic and
    total energy, and positions, every ``record_every`` steps.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(positions, dtype=np.float64).copy()
    n = x.shape[0]
    v = rng.normal(scale=np.sqrt(kT), size=x.shape)
    v -= v.mean(axis=0)
    e, f = potential(x)
    out = {"time": [], "potential": [], "kinetic": [], "positions": []}
    for s in range(n_steps + 1):
        if s % record_every == 0:
            out["time"].append(s * dt)
            out["potential"].append(e)
            out["kinetic"].append(0.5 * float((v**2).sum()))
            out["positions"].append(x.copy())
        if s == n_steps:
            break
        v += 0.5 * dt * f  # half kick
        x += dt * v  # drift
        e, f = potential(x)  # new forces
        v += 0.5 * dt * f  # half kick
    res = {k: np.array(val) for k, val in out.items()}
    res["total"] = res["potential"] + res["kinetic"]
    res["n_beads"] = n
    return res


def langevin_batched(model, positions, n_steps: int = 20000, dt: float = 0.005, kT: float = 1.0,
                     friction: float = 1.0, record_every: int = 100, seed: int = 0, device="cpu"):
    """BAOAB Langevin dynamics, Eq. (D2), for R independent chains at once.

    ``positions`` is (R, N, 3). All replicas go through the network as one
    batched graph, so one network call gives the forces on every chain.
    Returns the recorded positions, (T, R, N, 3).
    """
    from torch_geometric.data import Batch

    from .graph import rebuild_edges

    g = torch.Generator(device=device).manual_seed(seed)
    model = model.to(device).eval()
    x = torch.as_tensor(np.asarray(positions), dtype=torch.float32, device=device)
    R, N, _ = x.shape
    empty = dict(edge_index=torch.zeros(2, 0, dtype=torch.long), edge_type=torch.zeros(0, dtype=torch.long))
    batch = Batch.from_data_list([Data(pos=x[r], z=node_types(N).to(device), **empty) for r in range(R)]).to(device)
    pos = x.reshape(R * N, 3).clone()
    v = torch.randn(pos.shape, generator=g, device=device) * kT**0.5
    c1 = float(np.exp(-friction * dt))
    c2 = float(np.sqrt((1 - c1**2) * kT))

    def force(p):
        batch.pos = p.detach().requires_grad_(True)
        rebuild_edges(batch, model.cutoff)
        _, f = model(batch)
        return f.detach()

    f = force(pos)
    frames = []
    for s in range(1, n_steps + 1):
        v = v + 0.5 * dt * f  # B
        pos = pos + 0.5 * dt * v  # A
        v = c1 * v + c2 * torch.randn(pos.shape, generator=g, device=device)  # O
        pos = pos + 0.5 * dt * v  # A
        f = force(pos)
        v = v + 0.5 * dt * f  # B
        if s % record_every == 0:
            frames.append(pos.reshape(R, N, 3).cpu().numpy().copy())
    return np.array(frames)


def reference_potential(n_beads: int, model: str = "kg"):
    """The exact OpenMM potential with the same call signature as :class:`LearnedPotential`."""
    from .sim import build_system, evaluate

    ctx = openmm.Context(build_system(n_beads, model), openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName("Reference"))

    def pot(x):
        ctx.setPositions(np.asarray(x) * unit.nanometer)
        _, f, e = evaluate(ctx)
        return e, f

    return pot
