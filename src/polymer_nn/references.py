# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
# Authors: Dara, Hoshino
"""Bibliography for the Polymer Neural Nets project.

The sources behind the graph neural networks, equivariant force fields,
coarse-graining, generative models, polymer physics, integrators and open-source
software used in this repository, in one place.

``scripts/make_references.py`` renders this list to ``docs/REFERENCES.md``,
``docs/references.bib`` and ``docs/REFERENCES.pdf``. Edit entries here only, and
the three outputs stay in sync.

    >>> from polymer_nn.references import REFERENCES, cite
    >>> cite("satorras2021")
    'Satorras, Hoogeboom & Welling (2021)'
"""

from __future__ import annotations

from dataclasses import dataclass, field

TITLE = "Polymer Neural Nets: References"
AUTHORS = ("Dara", "Hoshino")
COPYRIGHT = "© Computational Classnotes · CC BY-NC-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


@dataclass(frozen=True)
class Reference:
    key: str
    kind: str  # article, inproceedings, book, misc
    authors: tuple[str, ...]  # "Last, First" ; "others" for et al.
    title: str
    year: int
    venue: str = ""  # journal, proceedings or publisher
    volume: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""
    topic: str = ""
    used_for: str = ""  # where it shows up in this project
    extra: dict = field(default_factory=dict)


TOPICS = (
    "Companion notebooks",
    "Mathematics of neural networks",
    "Graph neural networks",
    "Machine-learned potentials",
    "Equivariant networks",
    "Coarse-graining",
    "Generative models",
    "Polymer property prediction",
    "Polymer physics and models",
    "Integrators and thermostats",
    "Monte Carlo",
    "Open-source software",
    "Optimisation",
)

_R = Reference

REFERENCES: tuple[Reference, ...] = (
    # ------------------------------------------------- companion notebooks
    _R("shayda2026a", "misc", ("O Shayda, Dara", "Hoshino, Noor"), "S Shape Functions Theory", 2026,
       "Computational Classnotes", url="https://github.com/lossofgenerality/neural-networks-things",
       topic="Companion notebooks", used_for="Sigmoids and universal approximation (theory notebook, section 2)."),
    _R("shayda2026b", "misc", ("O Shayda, Dara", "Hoshino, Noor"),
       "Feedforward Neural Networks: The Algebra and the Computing", 2026, "Computational Classnotes",
       url="https://github.com/lossofgenerality/neural-networks-things", topic="Companion notebooks",
       used_for="Neuron, layer, backpropagation and Adam, repeated in the theory notebook, sections 1-2."),
    # ------------------------------------------- mathematics of neural nets
    _R("rumelhart1986", "article", ("Rumelhart, David E.", "Hinton, Geoffrey E.", "Williams, Ronald J."),
       "Learning representations by back-propagating errors", 1986, "Nature", "323", "533--536",
       doi="10.1038/323533a0", topic="Mathematics of neural networks", used_for="Backpropagation (Theorem 1)."),
    _R("cybenko1989", "article", ("Cybenko, George",),
       "Approximation by superpositions of a sigmoidal function", 1989,
       "Mathematics of Control, Signals and Systems", "2", "303--314", doi="10.1007/BF02551274",
       topic="Mathematics of neural networks", used_for="Universal approximation (Theorem 2)."),
    _R("elfwing2018", "article", ("Elfwing, Stefan", "Uchibe, Eiji", "Doya, Kenji"),
       "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning", 2018,
       "Neural Networks", "107", "3--11", doi="10.1016/j.neunet.2017.12.012",
       topic="Mathematics of neural networks", used_for="The SiLU activation used in every MLP."),
    # ---------------------------------------------------------------- GNNs
    _R("scarselli2009", "article",
       ("Scarselli, Franco", "Gori, Marco", "Tsoi, Ah Chung", "Hagenbuchner, Markus", "Monfardini, Gabriele"),
       "The Graph Neural Network Model", 2009, "IEEE Transactions on Neural Networks", "20", "61--80",
       doi="10.1109/TNN.2008.2005605", topic="Graph neural networks",
       used_for="Original GNN formulation: node states updated from neighbours."),
    _R("kipf2017", "inproceedings", ("Kipf, Thomas N.", "Welling, Max"),
       "Semi-Supervised Classification with Graph Convolutional Networks", 2017,
       "International Conference on Learning Representations (ICLR)", url="https://arxiv.org/abs/1609.02907",
       topic="Graph neural networks", used_for="Graph convolutions; background for message passing."),
    _R("gilmer2017", "inproceedings",
       ("Gilmer, Justin", "Schoenholz, Samuel S.", "Riley, Patrick F.", "Vinyals, Oriol", "Dahl, George E."),
       "Neural Message Passing for Quantum Chemistry", 2017,
       "Proceedings of the 34th International Conference on Machine Learning (ICML), PMLR", "70", "1263--1272",
       url="https://arxiv.org/abs/1704.01212", topic="Graph neural networks",
       used_for="Message-passing framework (message, aggregate, update) used by both models."),
    _R("battaglia2018", "misc", ("Battaglia, Peter W.", "others"),
       "Relational inductive biases, deep learning, and graph networks", 2018, "arXiv:1806.01261",
       url="https://arxiv.org/abs/1806.01261", topic="Graph neural networks",
       used_for="General graph-network view of nodes, edges and global readout."),
    # ------------------------------------------------------- ML potentials
    _R("behler2007", "article", ("Behler, Jörg", "Parrinello, Michele"),
       "Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces", 2007,
       "Physical Review Letters", "98", "146401", doi="10.1103/PhysRevLett.98.146401",
       topic="Machine-learned potentials", used_for="Energy as a sum of per-particle contributions (our readout)."),
    _R("schutt2017", "inproceedings",
       ("Schütt, Kristof T.", "Kindermans, Pieter-Jan", "Sauceda, Huziel E.", "Chmiela, Stefan",
        "Tkatchenko, Alexandre", "Müller, Klaus-Robert"),
       "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions", 2017,
       "Advances in Neural Information Processing Systems (NeurIPS)", "30", url="https://arxiv.org/abs/1706.08566",
       topic="Machine-learned potentials", used_for="Continuous-filter convolutions behind InvariantGNN."),
    _R("schutt2018", "article",
       ("Schütt, Kristof T.", "Sauceda, Huziel E.", "Kindermans, Pieter-Jan", "Tkatchenko, Alexandre",
        "Müller, Klaus-Robert"),
       "SchNet -- A deep learning architecture for molecules and materials", 2018,
       "The Journal of Chemical Physics", "148", "241722", doi="10.1063/1.5019779",
       topic="Machine-learned potentials", used_for="Radial basis expansion and cutoff design."),
    _R("chmiela2017", "article",
       ("Chmiela, Stefan", "Tkatchenko, Alexandre", "Sauceda, Huziel E.", "Poltavsky, Igor",
        "Schütt, Kristof T.", "Müller, Klaus-Robert"),
       "Machine learning of accurate energy-conserving molecular force fields", 2017,
       "Science Advances", "3", "e1603015", doi="10.1126/sciadv.1603015",
       topic="Machine-learned potentials", used_for="Forces as the gradient of a learned energy (conservative)."),
    # ------------------------------------------------- equivariant networks
    _R("thomas2018", "misc",
       ("Thomas, Nathaniel", "Smidt, Tess", "Kearnes, Steven", "Yang, Lusann", "Li, Li", "Kohlhoff, Kai",
        "Riley, Patrick"),
       "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds", 2018,
       "arXiv:1802.08219", url="https://arxiv.org/abs/1802.08219", topic="Equivariant networks",
       used_for="Spherical-harmonic tensor messages; basis of NequIP and MACE."),
    _R("satorras2021", "inproceedings", ("Satorras, Víctor Garcia", "Hoogeboom, Emiel", "Welling, Max"),
       "E(n) Equivariant Graph Neural Networks", 2021,
       "Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR", "139", "9323--9332",
       url="https://arxiv.org/abs/2102.09844", topic="Equivariant networks",
       used_for="Coordinate-updating layers in EquivariantGNN."),
    _R("batzner2022", "article",
       ("Batzner, Simon", "Musaelian, Albert", "Sun, Lixin", "Geiger, Mario", "Mailoa, Jonathan P.",
        "Kornbluth, Mordechai", "Molinari, Nicola", "Smidt, Tess E.", "Kozinsky, Boris"),
       "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials", 2022,
       "Nature Communications", "13", "2453", doi="10.1038/s41467-022-29939-5", topic="Equivariant networks",
       used_for="NequIP: the equivariant design named in the brief."),
    _R("batatia2022", "inproceedings",
       ("Batatia, Ilyes", "Kovács, Dávid Péter", "Simm, Gregor N. C.", "Ortner, Christoph", "Csányi, Gábor"),
       "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", 2022,
       "Advances in Neural Information Processing Systems (NeurIPS)", "35", url="https://arxiv.org/abs/2206.07697",
       topic="Equivariant networks", used_for="MACE: higher body-order messages; planned drop-in model."),
    _R("geiger2022", "misc", ("Geiger, Mario", "Smidt, Tess"), "e3nn: Euclidean Neural Networks", 2022,
       "arXiv:2207.09453", doi="10.48550/arXiv.2207.09453", topic="Equivariant networks",
       used_for="Library for tensor-product layers if NequIP/MACE-style models are added."),
    # ------------------------------------------------------ coarse-graining
    _R("izvekov2005", "article", ("Izvekov, Sergei", "Voth, Gregory A."),
       "A Multiscale Coarse-Graining Method for Biomolecular Systems", 2005,
       "The Journal of Physical Chemistry B", "109", "2469--2473", doi="10.1021/jp044629q",
       topic="Coarse-graining", used_for="Force matching from fine-grained MD."),
    _R("noid2008", "article",
       ("Noid, W. G.", "Chu, Jhih-Wei", "Ayton, Gary S.", "Krishna, Vinod", "Izvekov, Sergei",
        "Voth, Gregory A.", "Das, Avisek", "Andersen, Hans C."),
       "The multiscale coarse-graining method. I. A rigorous bridge between atomistic and coarse-grained models",
       2008, "The Journal of Chemical Physics", "128", "244114", doi="10.1063/1.2938860",
       topic="Coarse-graining", used_for="Linear mapping operators and mapped forces (coarse_grain.py)."),
    _R("wang2019", "article",
       ("Wang, Jiang", "Olsson, Simon", "Wehmeyer, Christoph", "Pérez, Adrià", "Charron, Nicholas E.",
        "de Fabritiis, Gianni", "Noé, Frank", "Clementi, Cecilia"),
       "Machine Learning of Coarse-Grained Molecular Dynamics Force Fields", 2019,
       "ACS Central Science", "5", "755--767", doi="10.1021/acscentsci.8b00913", topic="Coarse-graining",
       used_for="CGnets: force matching as supervised learning."),
    _R("husic2020", "article",
       ("Husic, Brooke E.", "Charron, Nicholas E.", "Lemm, Dominik", "Wang, Jiang", "Pérez, Adrià",
        "Majewski, Maciej", "Krämer, Andreas", "Chen, Yaoyi", "Olsson, Simon", "de Fabritiis, Gianni",
        "Noé, Frank", "Clementi, Cecilia"),
       "Coarse graining molecular dynamics with graph neural networks", 2020,
       "The Journal of Chemical Physics", "153", "194101", doi="10.1063/5.0026133", topic="Coarse-graining",
       used_for="GNN (continuous-filter) coarse-grained force fields."),
    # ---------------------------------------------------- generative models
    _R("rezende2015", "inproceedings", ("Rezende, Danilo Jimenez", "Mohamed, Shakir"),
       "Variational Inference with Normalizing Flows", 2015,
       "Proceedings of the 32nd International Conference on Machine Learning (ICML), PMLR", "37", "1530--1538",
       url="https://arxiv.org/abs/1505.05770", topic="Generative models", used_for="Normalizing flows (planned)."),
    _R("noe2019", "article", ("Noé, Frank", "Olsson, Simon", "Köhler, Jonas", "Wu, Hao"),
       "Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning", 2019,
       "Science", "365", "eaaw1147", doi="10.1126/science.aaw1147", topic="Generative models",
       used_for="Sampling equilibrium configurations directly with flows (planned)."),
    _R("ho2020", "inproceedings", ("Ho, Jonathan", "Jain, Ajay", "Abbeel, Pieter"),
       "Denoising Diffusion Probabilistic Models", 2020,
       "Advances in Neural Information Processing Systems (NeurIPS)", "33", url="https://arxiv.org/abs/2006.11239",
       topic="Generative models", used_for="Diffusion models for chain conformations (planned)."),
    _R("hoogeboom2022", "inproceedings",
       ("Hoogeboom, Emiel", "Satorras, Víctor Garcia", "Vignac, Clément", "Welling, Max"),
       "Equivariant Diffusion for Molecule Generation in 3D", 2022,
       "Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR", "162",
       url="https://arxiv.org/abs/2203.17003", topic="Generative models",
       used_for="EGNN-based equivariant diffusion (planned)."),
    # ----------------------------------------------- property prediction
    _R("kuenneth2023", "article", ("Kuenneth, Christopher", "Ramprasad, Rampi"),
       "polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics", 2023,
       "Nature Communications", "14", "4099", doi="10.1038/s41467-023-39868-6",
       topic="Polymer property prediction", used_for="Property prediction (e.g. Tg) from polymer structure (planned)."),
    # ------------------------------------------------------ polymer physics
    _R("kremer1990", "article", ("Kremer, Kurt", "Grest, Gary S."),
       "Dynamics of entangled linear polymer melts: A molecular-dynamics simulation", 1990,
       "The Journal of Chemical Physics", "92", "5057--5086", doi="10.1063/1.458541",
       topic="Polymer physics and models", used_for="FENE + WCA bead-spring chain (sim.py)."),
    _R("weeks1971", "article", ("Weeks, John D.", "Chandler, David", "Andersen, Hans C."),
       "Role of Repulsive Forces in Determining the Equilibrium Structure of Simple Liquids", 1971,
       "The Journal of Chemical Physics", "54", "5237--5247", doi="10.1063/1.1674820",
       topic="Polymer physics and models", used_for="WCA purely repulsive potential."),
    _R("flory1953", "book", ("Flory, Paul J.",), "Principles of Polymer Chemistry", 1953,
       "Cornell University Press", topic="Polymer physics and models",
       used_for="Flory exponent and excluded volume."),
    _R("degennes1979", "book", ("de Gennes, Pierre-Gilles",), "Scaling Concepts in Polymer Physics", 1979,
       "Cornell University Press", topic="Polymer physics and models", used_for="R_g ~ N^nu scaling."),
    _R("rubinstein2003", "book", ("Rubinstein, Michael", "Colby, Ralph H."), "Polymer Physics", 2003,
       "Oxford University Press", topic="Polymer physics and models",
       used_for="Ideal and real chains, Rouse dynamics (sampling times)."),
    _R("clisby2010", "article", ("Clisby, Nathan",),
       "Accurate estimate of the critical exponent nu for self-avoiding walks via a fast implementation of the pivot algorithm",
       2010, "Physical Review Letters", "104", "055702", doi="10.1103/PhysRevLett.104.055702",
       topic="Polymer physics and models", used_for="nu = 0.587597 reference value (analysis.py)."),
    # ------------------------------------------------------- integrators
    _R("swope1982", "article", ("Swope, William C.", "Andersen, Hans C.", "Berens, Peter H.", "Wilson, Kent R."),
       "A computer simulation method for the calculation of equilibrium constants for the formation of physical "
       "clusters of molecules: Application to small water clusters", 1982,
       "The Journal of Chemical Physics", "76", "637--649", doi="10.1063/1.442716",
       topic="Integrators and thermostats", used_for="Velocity Verlet (md.velocity_verlet)."),
    _R("zhang2019", "article",
       ("Zhang, Zhijun", "Liu, Xinzijian", "Yan, Kangyu", "Tuckerman, Mark E.", "Liu, Jian"),
       "Unified Efficient Thermostat Scheme for the Canonical Ensemble with Holonomic or Isokinetic Constraints "
       "via Molecular Dynamics", 2019, "The Journal of Physical Chemistry A", "123", "6056--6079",
       doi="10.1021/acs.jpca.9b02771", topic="Integrators and thermostats",
       used_for="'Middle' Langevin scheme behind OpenMM's LangevinMiddleIntegrator."),
    _R("hairer2003", "article", ("Hairer, Ernst", "Lubich, Christian", "Wanner, Gerhard"),
       "Geometric numerical integration illustrated by the Störmer-Verlet method", 2003, "Acta Numerica", "12",
       "399--450", doi="10.1017/S0962492902000144", topic="Integrators and thermostats",
       used_for="Why Verlet keeps the energy: the modified (shadow) energy, Eq. (9.2)."),
    _R("leimkuhler2013", "article", ("Leimkuhler, Benedict", "Matthews, Charles"),
       "Rational Construction of Stochastic Numerical Methods for Molecular Sampling", 2013,
       "Applied Mathematics Research eXpress", "2013", "34--56", doi="10.1093/amrx/abs010",
       topic="Integrators and thermostats", used_for="BAOAB Langevin splitting (md.langevin_batched)."),
    # ---------------------------------------------------------- Monte Carlo
    _R("lal1969", "article", ("Lal, Moti",),
       "'Monte Carlo' computer simulation of chain molecules. I", 1969, "Molecular Physics", "17", "57--64",
       doi="10.1080/00268976900100781", topic="Monte Carlo", used_for="The pivot move (sim.pivot_mc)."),
    _R("madras1988", "article", ("Madras, Neal", "Sokal, Alan D."),
       "The pivot algorithm: A highly efficient Monte Carlo method for the self-avoiding walk", 1988,
       "Journal of Statistical Physics", "50", "109--186", doi="10.1007/BF01022990", topic="Monte Carlo",
       used_for="Why pivots decorrelate R_g quickly (Theorem 11)."),
    # ---------------------------------------------------- software (all OSS)
    _R("paszke2019", "inproceedings", ("Paszke, Adam", "others"),
       "PyTorch: An Imperative Style, High-Performance Deep Learning Library", 2019,
       "Advances in Neural Information Processing Systems (NeurIPS)", "32", "8024--8035",
       url="https://arxiv.org/abs/1912.01703", topic="Open-source software", used_for="PyTorch (BSD-3-Clause)."),
    _R("fey2019", "misc", ("Fey, Matthias", "Lenssen, Jan Eric"),
       "Fast Graph Representation Learning with PyTorch Geometric", 2019,
       "ICLR Workshop on Representation Learning on Graphs and Manifolds; arXiv:1903.02428",
       url="https://arxiv.org/abs/1903.02428", topic="Open-source software",
       used_for="PyTorch Geometric (MIT): MessagePassing, Data, DataLoader."),
    _R("eastman2017", "article", ("Eastman, Peter", "others"),
       "OpenMM 7: Rapid development of high performance algorithms for molecular dynamics", 2017,
       "PLOS Computational Biology", "13", "e1005659", doi="10.1371/journal.pcbi.1005659",
       topic="Open-source software", used_for="OpenMM (MIT/LGPL) reference simulations."),
    _R("eastman2024", "article", ("Eastman, Peter", "others"),
       "OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials", 2024,
       "The Journal of Physical Chemistry B", "128", "109--116", doi="10.1021/acs.jpcb.3c06662",
       topic="Open-source software", used_for="PyTorch models in OpenMM (openmm-torch route for GPU runs)."),
    _R("thompson2022", "article", ("Thompson, Aidan P.", "others"),
       "LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and "
       "continuum scales", 2022, "Computer Physics Communications", "271", "108171",
       doi="10.1016/j.cpc.2021.108171", topic="Open-source software",
       used_for="LAMMPS (GPL-2.0): alternative engine for large melts."),
    _R("harris2020", "article", ("Harris, Charles R.", "others"), "Array programming with NumPy", 2020,
       "Nature", "585", "357--362", doi="10.1038/s41586-020-2649-2", topic="Open-source software",
       used_for="NumPy (BSD-3-Clause)."),
    _R("virtanen2020", "article", ("Virtanen, Pauli", "others"),
       "SciPy 1.0: fundamental algorithms for scientific computing in Python", 2020, "Nature Methods", "17",
       "261--272", doi="10.1038/s41592-019-0686-2", topic="Open-source software",
       used_for="SciPy (BSD-3-Clause): fits, KS and Wasserstein tests."),
    # ------------------------------------------------------- optimisation
    _R("kingma2015", "inproceedings", ("Kingma, Diederik P.", "Ba, Jimmy"),
       "Adam: A Method for Stochastic Optimization", 2015,
       "International Conference on Learning Representations (ICLR)", url="https://arxiv.org/abs/1412.6980",
       topic="Optimisation", used_for="Optimiser in train.py."),
    _R("loshchilov2017", "inproceedings", ("Loshchilov, Ilya", "Hutter, Frank"),
       "SGDR: Stochastic Gradient Descent with Warm Restarts", 2017,
       "International Conference on Learning Representations (ICLR)", url="https://arxiv.org/abs/1608.03983",
       topic="Optimisation", used_for="Cosine learning-rate schedule in train.py."),
)

BY_KEY = {r.key: r for r in REFERENCES}


# ----------------------------------------------------------------- render


def _last(name: str) -> str:
    return name.split(",")[0].strip()


def short_authors(r: Reference) -> str:
    names = [a for a in r.authors if a != "others"]
    etal = "others" in r.authors or len(names) > 3
    if etal:
        return f"{_last(names[0])} et al."
    if len(names) == 1:
        return _last(names[0])
    return ", ".join(_last(n) for n in names[:-1]) + " & " + _last(names[-1])


def cite(key: str) -> str:
    r = BY_KEY[key]
    return f"{short_authors(r)} ({r.year})"


def _display_name(name: str) -> str:
    if name == "others":
        return "et al."
    last, _, first = name.partition(",")
    return f"{first.strip()} {last.strip()}".strip()


def long_authors(r: Reference) -> str:
    names = [_display_name(a) for a in r.authors]
    if names[-1] == "et al.":
        return ", ".join(names[:-1]) + " et al."
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def link(r: Reference) -> str:
    if r.doi:
        return f"https://doi.org/{r.doi}"
    return r.url


def plain(r: Reference) -> str:
    """One-line citation without markup."""
    bits = [f"{long_authors(r)} ({r.year}). {r.title}."]
    v = r.venue
    if r.volume:
        v += f" {r.volume}"
    if r.pages:
        v += f", {r.pages.replace('--', '–')}"
    if v:
        bits.append(v + ".")
    return " ".join(bits)


def to_markdown() -> str:
    lines = [f"# {TITLE}", "", f"**Authors:** {', '.join(AUTHORS)}  ", f"{COPYRIGHT} ({LICENSE_URL})", "",
             "Sources for the graph neural networks, equivariant force fields, coarse-graining, generative models, "
             "polymer physics, integrators and open-source software used in this project. "
             "Generated from `src/polymer_nn/references.py` by `scripts/make_references.py`; "
             "edit the Python file, not this one.", ""]
    n = 0
    for topic in TOPICS:
        refs = [r for r in REFERENCES if r.topic == topic]
        if not refs:
            continue
        lines += [f"## {topic}", ""]
        for r in refs:
            n += 1
            url = link(r)
            lines.append(f"{n}. {plain(r)}" + (f" <{url}>" if url else "") + f"  \n   *Used for:* {r.used_for} `[{r.key}]`")
        lines.append("")
    lines += ["---", "", f"{COPYRIGHT}. Authors: {', '.join(AUTHORS)}.", ""]
    return "\n".join(lines)


def to_bibtex() -> str:
    out = [f"% {TITLE}", f"% Authors: {', '.join(AUTHORS)}", f"% {COPYRIGHT}", ""]
    field_for_venue = {"article": "journal", "inproceedings": "booktitle", "book": "publisher", "misc": "howpublished"}
    for r in REFERENCES:
        f = {
            "author": " and ".join(r.authors),
            "title": "{" + r.title + "}",
            "year": str(r.year),
        }
        if r.venue:
            f[field_for_venue[r.kind]] = r.venue
        if r.volume:
            f["volume"] = r.volume
        if r.pages:
            f["pages"] = r.pages
        if r.doi:
            f["doi"] = r.doi
        if r.url:
            f["url"] = r.url
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items())
        out.append(f"@{r.kind}{{{r.key},\n{body}\n}}\n")
    return "\n".join(out)
