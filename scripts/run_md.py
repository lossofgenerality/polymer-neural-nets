# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Run the three physics checks with a trained model and write ``results/``.

1. **Energy conservation.** NVE velocity Verlet with the learned forces and
   with the exact OpenMM forces, from the same frame and velocities, for 4 τ
   at Δt = 0.004, 0.002 and 0.001 τ (``polymer_nn.md.velocity_verlet``).
2. **Local structure.** OpenMM's Langevin integrator driven by the learned
   forces (``polymer_nn.md.OpenMMLearnedSimulation``); bond-length and
   bond-angle distributions compared with the reference data.
3. **Global structure.** R_g at the chain lengths ``--sizes`` (by default
   16, 32 and 64; the model was trained on 32), from ``--replicas``
   independent chains run together with batched BAOAB Langevin dynamics
   (``polymer_nn.md.langevin_batched``), compared with the reference
   R_g of ``--scaling``.

Writes ``results/checks_<model>.json`` (the numbers) and
``results/checks_<model>.npz`` (the arrays behind the figures of notebook 04).

    uv run python scripts/run_md.py --model models/invariant.pt --ref data/kg_N32.npz --scaling data/scaling_kg.npz --replicas 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from polymer_nn import analysis as A
from polymer_nn.md import LearnedPotential, OpenMMLearnedSimulation, langevin_batched, reference_potential, velocity_verlet
from polymer_nn.models import load_model
from polymer_nn.sim import initial_positions, pivot_mc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True, help="reference data for the trained chain length")
    p.add_argument("--scaling", type=Path, help="reference R_g(N) from generate_data.py --scaling")
    p.add_argument("--sizes", default="16,32,64")
    p.add_argument("--replicas", type=int, default=16)
    p.add_argument("--langevin-steps", type=int, default=20000)
    p.add_argument("--pivot-moves", type=int, default=2000)
    p.add_argument("--friction", type=float, default=0.1, help="1/tau; low friction decorrelates R_g faster")
    p.add_argument("--omm-frames", type=int, default=200)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.model.stem
    model, ckpt = load_model(a.model)
    ref = np.load(a.ref)
    n = int(ref["n_beads"])
    pot = LearnedPotential(model, n, ckpt["energy_offset"], device=a.device)
    x0 = ref["positions"][-1]
    out: dict = {"model": tag}
    s_pre = dict(np.load(a.scaling)) if a.scaling and a.scaling.exists() else None
    arrays: dict = {}

    # 1. energy conservation ------------------------------------------------
    t = time.time()
    out["nve"] = {}
    for dt in (0.001, 0.002, 0.004):
        steps = int(4.0 / dt)  # 4 tau
        lr = velocity_verlet(pot, x0, n_steps=steps, dt=dt, record_every=max(1, steps // 400))
        rr = velocity_verlet(reference_potential(n), x0, n_steps=steps, dt=dt, record_every=max(1, steps // 400))
        out["nve"][str(dt)] = {"learned": A.energy_drift(lr["time"], lr["total"], n),
                               "reference": A.energy_drift(rr["time"], rr["total"], n)}
        if dt == 0.002:
            arrays.update(nve_time=lr["time"], nve_learned_total=lr["total"], nve_learned_pot=lr["potential"],
                          nve_ref_total=rr["total"])
    print(f"[1] NVE done in {time.time() - t:.0f}s:", json.dumps(out["nve"]["0.002"]), flush=True)

    # 2. structure via OpenMM ---------------------------------------------
    t = time.time()
    sim = OpenMMLearnedSimulation(pot, x0, integrator="langevin", seed=3)
    sim.step(2000)
    o = sim.run(a.omm_frames, stride=50)
    bl, ba = A.bond_lengths(o["positions"]), A.bond_angles(o["positions"])
    bl_ref, ba_ref = A.bond_lengths(ref["positions"]), A.bond_angles(ref["positions"])
    out["structure"] = {"bond_length": A.compare_distributions(bl, bl_ref),
                        "bond_angle": A.compare_distributions(ba, ba_ref),
                        "kinetic_per_bead_over_kT": float(np.mean(o["kinetic"]) / n)}
    arrays.update(omm_bond=bl, omm_angle=ba, ref_bond=bl_ref, ref_angle=ba_ref)
    print(f"[2] OpenMM structure done in {time.time() - t:.0f}s:", json.dumps(out["structure"]), flush=True)

    # 3. R_g scaling on unseen lengths --------------------------------------
    sizes = [int(s) for s in a.sizes.split(",")]
    rg = []
    rng = np.random.default_rng(1)
    s_ref = s_pre
    for N in sizes:
        t = time.time()
        # Fresh random walks, pre-relaxed by excluded-volume pivot MC (bonds stay at the
        # generic 0.97 sigma, so the learned model has to equilibrate bonds and local
        # structure itself). A wrong model drifts away from the reference R_g within
        # the run, because N <= 64 relaxes in << 100 tau at friction 0.1.
        starts = np.stack([pivot_mc(initial_positions(N, rng), a.pivot_moves, rng) for _ in range(a.replicas)])
        frames = langevin_batched(model, starts, n_steps=a.langevin_steps, record_every=100, seed=N,
                                  friction=a.friction, device=a.device)
        frames = frames[len(frames) // 4:]  # drop first quarter as equilibration
        r = A.radius_of_gyration(frames.reshape(-1, N, 3))
        rg.append(float(np.sqrt(np.mean(r**2))))
        arrays[f"rg_N{N}"] = r
        print(f"[3] N={N}: <Rg^2>^1/2 = {rg[-1]:.3f} ({time.time() - t:.0f}s)", flush=True)
    fit = A.fit_flory_exponent(sizes, rg)
    out["scaling"] = {"sizes": sizes, "rg": rg, "nu": fit.nu, "nu_err": fit.nu_err}
    if s_ref is not None:
        s = s_ref
        keep = [i for i, N in enumerate(s["sizes"]) if N in sizes]
        out["scaling"]["reference_rg"] = [float(s["kg_rg"][i]) for i in keep]
        if "kg_pivot_rg" in s:
            out["scaling"]["pivot_rg"] = [float(s["kg_pivot_rg"][i]) for i in keep]
        rf = A.fit_flory_exponent(s["sizes"][keep], s["kg_rg"][keep])
        out["scaling"]["reference_nu"] = rf.nu
    print("[3]", json.dumps(out["scaling"]), flush=True)

    out = {"copyright": "© Computational Classnotes · CC BY-NC-SA 4.0", **out}
    (a.out / f"checks_{tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    np.savez_compressed(a.out / f"checks_{tag}.npz", **arrays)
    print(f"wrote {a.out}/checks_{tag}.json")


if __name__ == "__main__":
    main()
