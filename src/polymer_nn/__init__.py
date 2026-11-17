# © Computational Classnotes · CC BY-NC-SA 4.0
# Polymer Neural Nets, version 1.0.0 (2026-11-17)
"""polymer_nn: graph neural network force fields for bead-spring polymers.

The pipeline, module by module:

* ``sim``          makes the data: OpenMM simulations of Kremer–Grest and ideal
                   chains, and pivot Monte Carlo (positions, forces, energies);
* ``graph``        turns one configuration into the network input, a graph;
* ``models``       the two networks, graph → energy E, and forces F = -∂E/∂x;
* ``train``        learns the weights by force matching;
* ``md``           runs molecular dynamics with the learned forces;
* ``analysis``     the physics checks: R_g scaling, bonds and angles, energy;
* ``coarse_grain`` mappings for coarse-grained force matching;
* ``references``   the bibliography (Dara, Hoshino);
* ``nbtools``      table and figure helpers for the notebooks.

The notebooks in ``00-theory`` … ``04-learned-dynamics`` explain every step.
"""

__version__ = "1.0.0"
COPYRIGHT = "© Computational Classnotes · CC BY-NC-SA 4.0"
