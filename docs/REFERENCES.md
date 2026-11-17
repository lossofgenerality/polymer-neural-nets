# Polymer Neural Nets: References

**Authors:** Dara, Hoshino  
© Computational Classnotes · CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

Sources for the graph neural networks, equivariant force fields, coarse-graining, generative models, polymer physics, integrators and open-source software used in this project. Generated from `src/polymer_nn/references.py` by `scripts/make_references.py`; edit the Python file, not this one.

## Companion notebooks

1. Dara O Shayda and Noor Hoshino (2026). S Shape Functions Theory. Computational Classnotes. <https://github.com/lossofgenerality/neural-networks-things>  
   *Used for:* Sigmoids and universal approximation (theory notebook, section 2). `[shayda2026a]`
2. Dara O Shayda and Noor Hoshino (2026). Feedforward Neural Networks: The Algebra and the Computing. Computational Classnotes. <https://github.com/lossofgenerality/neural-networks-things>  
   *Used for:* Neuron, layer, backpropagation and Adam, repeated in the theory notebook, sections 1-2. `[shayda2026b]`

## Mathematics of neural networks

3. David E. Rumelhart, Geoffrey E. Hinton and Ronald J. Williams (1986). Learning representations by back-propagating errors. Nature 323, 533–536. <https://doi.org/10.1038/323533a0>  
   *Used for:* Backpropagation (Theorem 1). `[rumelhart1986]`
4. George Cybenko (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems 2, 303–314. <https://doi.org/10.1007/BF02551274>  
   *Used for:* Universal approximation (Theorem 2). `[cybenko1989]`
5. Stefan Elfwing, Eiji Uchibe and Kenji Doya (2018). Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks 107, 3–11. <https://doi.org/10.1016/j.neunet.2017.12.012>  
   *Used for:* The SiLU activation used in every MLP. `[elfwing2018]`

## Graph neural networks

6. Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner and Gabriele Monfardini (2009). The Graph Neural Network Model. IEEE Transactions on Neural Networks 20, 61–80. <https://doi.org/10.1109/TNN.2008.2005605>  
   *Used for:* Original GNN formulation: node states updated from neighbours. `[scarselli2009]`
7. Thomas N. Kipf and Max Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. International Conference on Learning Representations (ICLR). <https://arxiv.org/abs/1609.02907>  
   *Used for:* Graph convolutions; background for message passing. `[kipf2017]`
8. Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals and George E. Dahl (2017). Neural Message Passing for Quantum Chemistry. Proceedings of the 34th International Conference on Machine Learning (ICML), PMLR 70, 1263–1272. <https://arxiv.org/abs/1704.01212>  
   *Used for:* Message-passing framework (message, aggregate, update) used by both models. `[gilmer2017]`
9. Peter W. Battaglia et al. (2018). Relational inductive biases, deep learning, and graph networks. arXiv:1806.01261. <https://arxiv.org/abs/1806.01261>  
   *Used for:* General graph-network view of nodes, edges and global readout. `[battaglia2018]`

## Machine-learned potentials

10. Jörg Behler and Michele Parrinello (2007). Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces. Physical Review Letters 98, 146401. <https://doi.org/10.1103/PhysRevLett.98.146401>  
   *Used for:* Energy as a sum of per-particle contributions (our readout). `[behler2007]`
11. Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko and Klaus-Robert Müller (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in Neural Information Processing Systems (NeurIPS) 30. <https://arxiv.org/abs/1706.08566>  
   *Used for:* Continuous-filter convolutions behind InvariantGNN. `[schutt2017]`
12. Kristof T. Schütt, Huziel E. Sauceda, Pieter-Jan Kindermans, Alexandre Tkatchenko and Klaus-Robert Müller (2018). SchNet -- A deep learning architecture for molecules and materials. The Journal of Chemical Physics 148, 241722. <https://doi.org/10.1063/1.5019779>  
   *Used for:* Radial basis expansion and cutoff design. `[schutt2018]`
13. Stefan Chmiela, Alexandre Tkatchenko, Huziel E. Sauceda, Igor Poltavsky, Kristof T. Schütt and Klaus-Robert Müller (2017). Machine learning of accurate energy-conserving molecular force fields. Science Advances 3, e1603015. <https://doi.org/10.1126/sciadv.1603015>  
   *Used for:* Forces as the gradient of a learned energy (conservative). `[chmiela2017]`

## Equivariant networks

14. Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff and Patrick Riley (2018). Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds. arXiv:1802.08219. <https://arxiv.org/abs/1802.08219>  
   *Used for:* Spherical-harmonic tensor messages; basis of NequIP and MACE. `[thomas2018]`
15. Víctor Garcia Satorras, Emiel Hoogeboom and Max Welling (2021). E(n) Equivariant Graph Neural Networks. Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139, 9323–9332. <https://arxiv.org/abs/2102.09844>  
   *Used for:* Coordinate-updating layers in EquivariantGNN. `[satorras2021]`
16. Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E. Smidt and Boris Kozinsky (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature Communications 13, 2453. <https://doi.org/10.1038/s41467-022-29939-5>  
   *Used for:* NequIP: the equivariant design named in the brief. `[batzner2022]`
17. Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner and Gábor Csányi (2022). MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. Advances in Neural Information Processing Systems (NeurIPS) 35. <https://arxiv.org/abs/2206.07697>  
   *Used for:* MACE: higher body-order messages; planned drop-in model. `[batatia2022]`
18. Mario Geiger and Tess Smidt (2022). e3nn: Euclidean Neural Networks. arXiv:2207.09453. <https://doi.org/10.48550/arXiv.2207.09453>  
   *Used for:* Library for tensor-product layers if NequIP/MACE-style models are added. `[geiger2022]`

## Coarse-graining

19. Sergei Izvekov and Gregory A. Voth (2005). A Multiscale Coarse-Graining Method for Biomolecular Systems. The Journal of Physical Chemistry B 109, 2469–2473. <https://doi.org/10.1021/jp044629q>  
   *Used for:* Force matching from fine-grained MD. `[izvekov2005]`
20. W. G. Noid, Jhih-Wei Chu, Gary S. Ayton, Vinod Krishna, Sergei Izvekov, Gregory A. Voth, Avisek Das and Hans C. Andersen (2008). The multiscale coarse-graining method. I. A rigorous bridge between atomistic and coarse-grained models. The Journal of Chemical Physics 128, 244114. <https://doi.org/10.1063/1.2938860>  
   *Used for:* Linear mapping operators and mapped forces (coarse_grain.py). `[noid2008]`
21. Jiang Wang, Simon Olsson, Christoph Wehmeyer, Adrià Pérez, Nicholas E. Charron, Gianni de Fabritiis, Frank Noé and Cecilia Clementi (2019). Machine Learning of Coarse-Grained Molecular Dynamics Force Fields. ACS Central Science 5, 755–767. <https://doi.org/10.1021/acscentsci.8b00913>  
   *Used for:* CGnets: force matching as supervised learning. `[wang2019]`
22. Brooke E. Husic, Nicholas E. Charron, Dominik Lemm, Jiang Wang, Adrià Pérez, Maciej Majewski, Andreas Krämer, Yaoyi Chen, Simon Olsson, Gianni de Fabritiis, Frank Noé and Cecilia Clementi (2020). Coarse graining molecular dynamics with graph neural networks. The Journal of Chemical Physics 153, 194101. <https://doi.org/10.1063/5.0026133>  
   *Used for:* GNN (continuous-filter) coarse-grained force fields. `[husic2020]`

## Generative models

23. Danilo Jimenez Rezende and Shakir Mohamed (2015). Variational Inference with Normalizing Flows. Proceedings of the 32nd International Conference on Machine Learning (ICML), PMLR 37, 1530–1538. <https://arxiv.org/abs/1505.05770>  
   *Used for:* Normalizing flows (planned). `[rezende2015]`
24. Frank Noé, Simon Olsson, Jonas Köhler and Hao Wu (2019). Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science 365, eaaw1147. <https://doi.org/10.1126/science.aaw1147>  
   *Used for:* Sampling equilibrium configurations directly with flows (planned). `[noe2019]`
25. Jonathan Ho, Ajay Jain and Pieter Abbeel (2020). Denoising Diffusion Probabilistic Models. Advances in Neural Information Processing Systems (NeurIPS) 33. <https://arxiv.org/abs/2006.11239>  
   *Used for:* Diffusion models for chain conformations (planned). `[ho2020]`
26. Emiel Hoogeboom, Víctor Garcia Satorras, Clément Vignac and Max Welling (2022). Equivariant Diffusion for Molecule Generation in 3D. Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR 162. <https://arxiv.org/abs/2203.17003>  
   *Used for:* EGNN-based equivariant diffusion (planned). `[hoogeboom2022]`

## Polymer property prediction

27. Christopher Kuenneth and Rampi Ramprasad (2023). polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics. Nature Communications 14, 4099. <https://doi.org/10.1038/s41467-023-39868-6>  
   *Used for:* Property prediction (e.g. Tg) from polymer structure (planned). `[kuenneth2023]`

## Polymer physics and models

28. Kurt Kremer and Gary S. Grest (1990). Dynamics of entangled linear polymer melts: A molecular-dynamics simulation. The Journal of Chemical Physics 92, 5057–5086. <https://doi.org/10.1063/1.458541>  
   *Used for:* FENE + WCA bead-spring chain (sim.py). `[kremer1990]`
29. John D. Weeks, David Chandler and Hans C. Andersen (1971). Role of Repulsive Forces in Determining the Equilibrium Structure of Simple Liquids. The Journal of Chemical Physics 54, 5237–5247. <https://doi.org/10.1063/1.1674820>  
   *Used for:* WCA purely repulsive potential. `[weeks1971]`
30. Paul J. Flory (1953). Principles of Polymer Chemistry. Cornell University Press.  
   *Used for:* Flory exponent and excluded volume. `[flory1953]`
31. Pierre-Gilles de Gennes (1979). Scaling Concepts in Polymer Physics. Cornell University Press.  
   *Used for:* R_g ~ N^nu scaling. `[degennes1979]`
32. Michael Rubinstein and Ralph H. Colby (2003). Polymer Physics. Oxford University Press.  
   *Used for:* Ideal and real chains, Rouse dynamics (sampling times). `[rubinstein2003]`
33. Nathan Clisby (2010). Accurate estimate of the critical exponent nu for self-avoiding walks via a fast implementation of the pivot algorithm. Physical Review Letters 104, 055702. <https://doi.org/10.1103/PhysRevLett.104.055702>  
   *Used for:* nu = 0.587597 reference value (analysis.py). `[clisby2010]`

## Integrators and thermostats

34. William C. Swope, Hans C. Andersen, Peter H. Berens and Kent R. Wilson (1982). A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules: Application to small water clusters. The Journal of Chemical Physics 76, 637–649. <https://doi.org/10.1063/1.442716>  
   *Used for:* Velocity Verlet (md.velocity_verlet). `[swope1982]`
35. Zhijun Zhang, Xinzijian Liu, Kangyu Yan, Mark E. Tuckerman and Jian Liu (2019). Unified Efficient Thermostat Scheme for the Canonical Ensemble with Holonomic or Isokinetic Constraints via Molecular Dynamics. The Journal of Physical Chemistry A 123, 6056–6079. <https://doi.org/10.1021/acs.jpca.9b02771>  
   *Used for:* 'Middle' Langevin scheme behind OpenMM's LangevinMiddleIntegrator. `[zhang2019]`
36. Ernst Hairer, Christian Lubich and Gerhard Wanner (2003). Geometric numerical integration illustrated by the Störmer-Verlet method. Acta Numerica 12, 399–450. <https://doi.org/10.1017/S0962492902000144>  
   *Used for:* Why Verlet keeps the energy: the modified (shadow) energy, Eq. (9.2). `[hairer2003]`
37. Benedict Leimkuhler and Charles Matthews (2013). Rational Construction of Stochastic Numerical Methods for Molecular Sampling. Applied Mathematics Research eXpress 2013, 34–56. <https://doi.org/10.1093/amrx/abs010>  
   *Used for:* BAOAB Langevin splitting (md.langevin_batched). `[leimkuhler2013]`

## Monte Carlo

38. Moti Lal (1969). 'Monte Carlo' computer simulation of chain molecules. I. Molecular Physics 17, 57–64. <https://doi.org/10.1080/00268976900100781>  
   *Used for:* The pivot move (sim.pivot_mc). `[lal1969]`
39. Neal Madras and Alan D. Sokal (1988). The pivot algorithm: A highly efficient Monte Carlo method for the self-avoiding walk. Journal of Statistical Physics 50, 109–186. <https://doi.org/10.1007/BF01022990>  
   *Used for:* Why pivots decorrelate R_g quickly (Theorem 11). `[madras1988]`

## Open-source software

40. Adam Paszke et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems (NeurIPS) 32, 8024–8035. <https://arxiv.org/abs/1912.01703>  
   *Used for:* PyTorch (BSD-3-Clause). `[paszke2019]`
41. Matthias Fey and Jan Eric Lenssen (2019). Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs and Manifolds; arXiv:1903.02428. <https://arxiv.org/abs/1903.02428>  
   *Used for:* PyTorch Geometric (MIT): MessagePassing, Data, DataLoader. `[fey2019]`
42. Peter Eastman et al. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. PLOS Computational Biology 13, e1005659. <https://doi.org/10.1371/journal.pcbi.1005659>  
   *Used for:* OpenMM (MIT/LGPL) reference simulations. `[eastman2017]`
43. Peter Eastman et al. (2024). OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials. The Journal of Physical Chemistry B 128, 109–116. <https://doi.org/10.1021/acs.jpcb.3c06662>  
   *Used for:* PyTorch models in OpenMM (openmm-torch route for GPU runs). `[eastman2024]`
44. Aidan P. Thompson et al. (2022). LAMMPS - a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales. Computer Physics Communications 271, 108171. <https://doi.org/10.1016/j.cpc.2021.108171>  
   *Used for:* LAMMPS (GPL-2.0): alternative engine for large melts. `[thompson2022]`
45. Charles R. Harris et al. (2020). Array programming with NumPy. Nature 585, 357–362. <https://doi.org/10.1038/s41586-020-2649-2>  
   *Used for:* NumPy (BSD-3-Clause). `[harris2020]`
46. Pauli Virtanen et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17, 261–272. <https://doi.org/10.1038/s41592-019-0686-2>  
   *Used for:* SciPy (BSD-3-Clause): fits, KS and Wasserstein tests. `[virtanen2020]`

## Optimisation

47. Diederik P. Kingma and Jimmy Ba (2015). Adam: A Method for Stochastic Optimization. International Conference on Learning Representations (ICLR). <https://arxiv.org/abs/1412.6980>  
   *Used for:* Optimiser in train.py. `[kingma2015]`
48. Ilya Loshchilov and Frank Hutter (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. International Conference on Learning Representations (ICLR). <https://arxiv.org/abs/1608.03983>  
   *Used for:* Cosine learning-rate schedule in train.py. `[loshchilov2017]`

---

© Computational Classnotes · CC BY-NC-SA 4.0. Authors: Dara, Hoshino.
