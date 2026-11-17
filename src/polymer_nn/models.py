# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""Graph neural network force fields: what happens inside the network.

Input and output
================

Input: one graph per configuration (``polymer_nn.graph``): bead positions
x_i, bead types z_i, directed edges j → i and edge types τ_ij (1 = bond,
0 = non-bonded neighbour within r_c).

Output: a single number per graph, the potential energy E(x), and, by
automatic differentiation, the force on every bead,

    F_i = -∂E/∂x_i,                                                     (M1)

an N × 3 array. Predicting E and differentiating it, instead of predicting
forces directly, has three consequences (theory notebook, sections 5 and 7):

* the forces are **conservative**: a closed path in configuration space does
  no work, so an NVE simulation conserves the total energy (Theorem 8);
* the forces are **equivariant**: turn the chain by a rotation R and the
  forces turn with it, F(Rx) = R F(x), because E depends only on distances
  (Theorems 4 and 5);
* E is **extensive**: it is a sum of per-bead terms, so a model trained on
  32-bead chains applies to chains of any length.

The layers
==========

Both networks are message-passing networks [gilmer2017]: each bead holds a
feature vector h_i ∈ ℝ^H (H = ``hidden`` = 64 channels), and a *layer* updates
every h_i from the messages sent by its neighbours. Notation (theory notebook,
section 6): r_ij = |x_i - x_j|; e(r) ∈ ℝ^K the radial basis of Eq. (M2);
onehot(τ_ij) ∈ {0,1}² the edge type; MLP a small multilayer perceptron with
SiLU activations (theory notebook, section 1).

Radial basis (K = ``n_rbf`` = 32 Gaussians on [0, r_max])::

    e_k(r) = exp(-γ (r - μ_k)²),  μ_k = k r_max / (K - 1),  γ = ½ (K / r_max)².   (M2)

Smooth cutoff envelope, 1 on bonds and a cosine on non-bonded edges::

    f(r) = ½ [cos(π r / r_c) + 1] for r < r_c, 0 otherwise;  f(r_c) = f'(r_c) = 0.   (M3)

**InvariantGNN** (continuous-filter convolution, SchNet style [schutt2017]),
for each of L = ``layers`` = 3 layers::

    W_ij = MLP_filter([e(r_ij), onehot(τ_ij)]) · f_ij            ∈ ℝ^H     (M4)
    m_ij = W_ij ⊙ (A h_j)                                          ∈ ℝ^H     (M5)
    h_i  ← h_i + MLP_update(Σ_{j→i} m_ij)                                  (M6)

**EquivariantGNN** (E(n)-equivariant layers, EGNN style [satorras2021]),
which also moves auxiliary coordinates y_i (y_i = x_i before the first layer)::

    m_ij = MLP_e([h_i, h_j, e(|y_i - y_j|), onehot(τ_ij)]) · f_ij           (M7)
    y_i  ← y_i + c Σ_{j→i} (y_i - y_j)/(|y_i - y_j| + 1) · MLP_x(m_ij) · f_ij   (M8)
    h_i  ← h_i + MLP_h([h_i, Σ_{j→i} m_ij])                                 (M9)

Readout, the same for both::

    ε_i = MLP_out(h_i),      E = Σ_i ε_i.                                    (M10)

h_i⁽⁰⁾ is a learned vector for each bead type (``nn.Embedding``), "⊙" is the
element-wise product, A a learned matrix, and c = 0.2.

Sources: gilmer2017, schutt2017, schutt2018, satorras2021, behler2007,
chmiela2017 (see ``polymer_nn.references``).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class GaussianRBF(nn.Module):
    """Radial basis e(r) of Eq. (M2): a distance becomes K smooth bumps.

    A single number r is a poor input for a neural network; K Gaussians
    centred on a grid turn it into a K-vector in which nearby distances give
    similar vectors and distant ones different vectors.
    """

    def __init__(self, n_rbf: int = 32, r_max: float = 2.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, r_max, n_rbf))  # μ_k
        self.gamma = 0.5 / (r_max / n_rbf) ** 2  # γ: widths about one grid spacing

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r has shape (|E|,); the result has shape (|E|, K)
        return torch.exp(-self.gamma * (r[:, None] - self.centers) ** 2)


def cosine_envelope(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """f(r) of Eq. (M3): falls smoothly from 1 at r = 0 to 0 at r = r_c.

    Its value and slope are both 0 at the cutoff, so a non-bonded neighbour
    that crosses r_c switches on and off without a jump in the energy or in
    the force (theory notebook, Theorem 6).
    """
    return torch.where(r < cutoff, 0.5 * (torch.cos(math.pi * r / cutoff) + 1.0), torch.zeros_like(r))


def edge_envelope(r: torch.Tensor, edge_type: torch.Tensor, cutoff: float) -> torch.Tensor:
    """1 on bonds (a bond is always present), f(r) on non-bonded edges."""
    return torch.where(edge_type == 1, torch.ones_like(r), cosine_envelope(r, cutoff))


def mlp(*sizes: int, act=nn.SiLU) -> nn.Sequential:
    """A multilayer perceptron: Linear, SiLU, Linear, …, Linear (no activation at the end).

    ``mlp(a, b, c)`` maps ℝ^a → ℝ^c through one hidden layer of b neurons,
    z ↦ W₂ SiLU(W₁ z + b₁) + b₂ (theory notebook, Eq. (1.6)). SiLU(z) =
    z σ(z) is smooth, so the energy is twice differentiable and the forces
    are smooth.
    """
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])


class _EnergyModel(nn.Module):
    """Common part of both networks: energy → forces by autograd, Eq. (M1).

    Subclasses implement ``energy(data)``, returning one energy per graph.
    Any model with this interface (for example a NequIP or MACE wrapper) works
    with ``train.fit``, ``md.LearnedPotential`` and ``md.langevin_batched``.
    """

    cutoff: float

    def energy(self, data, trace: dict | None = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, data, compute_forces: bool = True, trace: dict | None = None):
        """Return ``(E, F)``: energies (n_graphs,) and forces (n_nodes, 3).

        ``torch.autograd.grad`` computes ∂E/∂x for all beads at once by
        backpropagation (theory notebook, section 2), at about the cost of
        one more forward pass. During training ``create_graph=True`` keeps
        the derivative itself differentiable, because the loss contains F and
        we need ∂loss/∂weights through F.

        If ``trace`` is a dict, the intermediate tensors (distances, radial
        basis, features after each layer, per-bead energies) are stored in
        it, for inspection in the notebooks.
        """
        if compute_forces and not data.pos.requires_grad:
            data.pos.requires_grad_(True)
        e = self.energy(data, trace)
        if not compute_forces:
            return e, None
        (grad,) = torch.autograd.grad(e.sum(), data.pos, create_graph=self.training)
        return e, -grad


def _graph_index(data) -> torch.Tensor:
    """Which graph each node belongs to (all zeros for a single graph)."""
    b = getattr(data, "batch", None)
    return b if b is not None else torch.zeros_like(data.z)


# --------------------------------------------------------------------------
# Invariant continuous-filter GNN, Eqs. (M4)-(M6)
# --------------------------------------------------------------------------


class CFConv(MessagePassing):
    """One continuous-filter interaction layer, Eqs. (M4)-(M6).

    PyG's ``MessagePassing`` calls :meth:`message` once per edge (vectorised
    over all edges) and sums the messages at each target node
    (``aggr="add"``), which is the Σ_{j→i} of Eq. (M6).
    """

    def __init__(self, hidden: int, n_rbf: int):
        super().__init__(aggr="add")
        self.filter = mlp(n_rbf + 2, hidden, hidden)  # MLP_filter of Eq. (M4)
        self.lin_in = nn.Linear(hidden, hidden, bias=False)  # A of Eq. (M5)
        self.update_mlp = mlp(hidden, hidden, hidden)  # MLP_update of Eq. (M6)

    def forward(self, h, edge_index, rbf, edge_type, env):
        onehot = nn.functional.one_hot(edge_type, 2).to(rbf.dtype)
        w = self.filter(torch.cat([rbf, onehot], -1)) * env[:, None]  # W_ij, Eq. (M4)
        agg = self.propagate(edge_index, x=self.lin_in(h), w=w)  # Σ_j W_ij ⊙ A h_j
        return h + self.update_mlp(agg)  # residual update, Eq. (M6)

    def message(self, x_j, w):
        # x_j is A h_j gathered at the source of every edge: Eq. (M5)
        return x_j * w


class InvariantGNN(_EnergyModel):
    """SchNet-style energy model: only distances enter, so E is E(3)-invariant.

    Default size: H = 64 channels, K = 32 radial functions, L = 3 layers,
    about 61 000 trainable numbers.
    """

    def __init__(self, hidden: int = 64, layers: int = 3, n_rbf: int = 32, cutoff: float = 1.5, r_max: float = 2.0):
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(2, hidden)  # h_i^(0) for the two bead types
        self.rbf = GaussianRBF(n_rbf, r_max)
        self.convs = nn.ModuleList(CFConv(hidden, n_rbf) for _ in range(layers))
        self.readout = mlp(hidden, hidden, 1)  # MLP_out of Eq. (M10)

    def energy(self, data, trace=None):
        src, dst = data.edge_index  # edge j -> i: src = j, dst = i
        r = (data.pos[src] - data.pos[dst]).norm(dim=-1)  # r_ij for every edge
        rbf = self.rbf(r)  # e(r_ij), (|E|, K)
        env = edge_envelope(r, data.edge_type, self.cutoff)  # f_ij, (|E|,)
        h = self.embed(data.z)  # h^(0), (N, H)
        if trace is not None:
            trace.update(r=r, rbf=rbf, env=env, h0=h)
        for l, conv in enumerate(self.convs):
            h = conv(h, data.edge_index, rbf, data.edge_type, env)
            if trace is not None:
                trace[f"h{l + 1}"] = h
        e_i = self.readout(h).squeeze(-1)  # ε_i, (N,)
        if trace is not None:
            trace["e_i"] = e_i
        return scatter(e_i, _graph_index(data), dim=0, reduce="sum")  # E = Σ_i ε_i per graph


# --------------------------------------------------------------------------
# E(n)-equivariant GNN, Eqs. (M7)-(M9)
# --------------------------------------------------------------------------


class EGNNLayer(MessagePassing):
    """One E(n)-equivariant layer: updates features h and coordinates y."""

    def __init__(self, hidden: int, n_rbf: int, r_max: float):
        super().__init__(aggr="add")
        self.rbf = GaussianRBF(n_rbf, r_max)
        self.phi_e = mlp(2 * hidden + n_rbf + 2, hidden, hidden)  # MLP_e, Eq. (M7)
        self.phi_x = mlp(hidden, hidden, 1)  # MLP_x, Eq. (M8)
        self.phi_h = mlp(2 * hidden, hidden, hidden)  # MLP_h, Eq. (M9)
        self.coord_scale = 0.2  # c of Eq. (M8)

    def forward(self, h, x, edge_index, edge_type, env):
        src, dst = edge_index  # message j = src -> i = dst
        diff = x[dst] - x[src]  # y_i - y_j: a vector, turns with the chain
        r = diff.norm(dim=-1)  # |y_i - y_j|: a scalar, does not
        onehot = nn.functional.one_hot(edge_type, 2).to(h.dtype)
        m = self.phi_e(torch.cat([h[dst], h[src], self.rbf(r), onehot], -1)) * env[:, None]  # Eq. (M7)
        # Eq. (M8): vectors times scalars stay equivariant. A sum (not a mean
        # over neighbours) keeps it smooth when edges enter the cutoff.
        dx = diff / (r[:, None] + 1.0) * self.phi_x(m) * env[:, None]
        x = x + self.coord_scale * scatter(dx, dst, dim=0, dim_size=x.shape[0], reduce="add")
        agg = scatter(m, dst, dim=0, dim_size=h.shape[0], reduce="add")
        h = h + self.phi_h(torch.cat([h, agg], -1))  # Eq. (M9)
        return h, x


class EquivariantGNN(_EnergyModel):
    """EGNN-style energy model. The coordinate updates let later layers see angles.

    Default size: H = 64, K = 32, L = 3, about 98 000 trainable numbers.
    """

    def __init__(self, hidden: int = 64, layers: int = 3, n_rbf: int = 32, cutoff: float = 1.5, r_max: float = 2.0):
        super().__init__()
        self.cutoff = cutoff
        self.embed = nn.Embedding(2, hidden)
        self.layers = nn.ModuleList(EGNNLayer(hidden, n_rbf, r_max) for _ in range(layers))
        self.readout = mlp(hidden, hidden, 1)

    def energy(self, data, trace=None):
        src, dst = data.edge_index
        r0 = (data.pos[src] - data.pos[dst]).norm(dim=-1)
        env = edge_envelope(r0, data.edge_type, self.cutoff)  # from the true distances
        h, x = self.embed(data.z), data.pos  # y^(0) = x
        if trace is not None:
            trace.update(r=r0, env=env, h0=h)
        for l, layer in enumerate(self.layers):
            h, x = layer(h, x, data.edge_index, data.edge_type, env)
            if trace is not None:
                trace[f"h{l + 1}"], trace[f"y{l + 1}"] = h, x
        e_i = self.readout(h).squeeze(-1)
        if trace is not None:
            trace["e_i"] = e_i
        return scatter(e_i, _graph_index(data), dim=0, reduce="sum")


MODELS = {"invariant": InvariantGNN, "egnn": EquivariantGNN}


def build_model(name: str, **kwargs) -> _EnergyModel:
    """``build_model("invariant")`` or ``build_model("egnn", hidden=32)``."""
    return MODELS[name](**kwargs)


def save_model(model: _EnergyModel, path, name: str, config: dict, energy_offset: float) -> None:
    """Save a checkpoint: model name, constructor arguments, weights, energy offset."""
    torch.save({"name": name, "config": config, "state_dict": model.state_dict(), "energy_offset": energy_offset}, path)


def load_model(path, map_location="cpu"):
    """Load a checkpoint written by :func:`save_model`; returns ``(model, checkpoint)``."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ckpt["name"], **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
