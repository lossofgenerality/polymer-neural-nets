# © 2026 Computational Classnotes · CC BY-NC-SA 4.0
# Builds PolymerNN-Wolfram.nb from the cell list below (plain-text Notebook expression).
cells = []
def C(style, text): cells.append((style, text))
T = lambda t: C("Text", t)
I = lambda t: C("Input", t)

C("Title", "Graph Networks for Polymers in the Wolfram Language")
C("Subtitle", "A learned force field for bead-spring chains, trained by force matching")
C("Text", "© 2026 Computational Classnotes · CC BY-NC-SA 4.0\nAuthors: Dara O Shayda · Noor Hoshino")
T("This notebook is the Wolfram Language edition of polymer-neural-nets. It does in one place what the Python notebooks 01-04 do: it simulates a reference polymer, turns each frame into a graph, trains a graph network to reproduce the forces, and then runs new simulations with the network in place of the true physics, including on chains twice as long as any in the training set.")
T("All code lives in the package PolymerNN.wl, kept next to this notebook. Every function there has a long comment explaining what it does and why. Reduced Lennard-Jones units are used throughout: ε = σ = m = k_B = 1, and T = 1.")

C("Section", "1. Loading the package")
I('SetDirectory[NotebookDirectory[]];\nGet["PolymerNN.wl"];\n$PNN')
T("$PNN holds every parameter. The bonds are harmonic springs (kBond = 100, r0 = 1). Every pair of beads also repels through the WCA potential, a Lennard-Jones potential cut at its minimum 2^(1/6) and shifted to zero there. So the chain is a self-avoiding walk in a good solvent.")

C("Section", "2. The reference model")
T("The exact forces are compiled (to C when a compiler is present). First check that they are minus the gradient of the energy, by central finite differences on one bead.")
I('''SeedRandom[1];
x = LangevinRun[StraightChain[32], 20000, 20000][[-1, 1]];
h = 10.^-6;
e[k_] := Outer[Times, UnitVector[32, 5], UnitVector[3, k]];
fd = Table[-(ReferenceEnergy[x + h e[k]] - ReferenceEnergy[x - h e[k]])/(2 h), {k, 3}];
{ReferenceForces[x][[5]], fd}''')
T("Langevin dynamics uses the BAOAB splitting: half kick, half drift, an exact Ornstein-Uhlenbeck step for the velocities, half drift, half kick. Generate training data: 8 independent chains of 32 beads, each started straight, equilibrated for 20 000 steps and then sampled every 100 steps (500 frames each). Two more chains of 32 and two of 64 beads are kept aside as test sets.")
I('''SeedRandom[2026]; frames = GenerateData[32, 8, 20000, 500, 100];
SeedRandom[2027]; test32 = GenerateData[32, 2, 20000, 200, 100];
SeedRandom[2028]; test64 = GenerateData[64, 2, 40000, 200, 100];
{Length[frames], "RMS force" -> Sqrt[Mean[Flatten[frames[[All, 2]]^2]]]}''')
I('Graphics3D[{Gray, Tube[frames[[-1, 1]], 0.08], RGBColor[0.2, 0.4, 0.8], Sphere[frames[[-1, 1]], 0.3]}, Boxed -> False, PlotLabel -> "A reference chain, N = 32"]')

C("Section", "3. The chain as a graph")
T("Beads are nodes. There are two kinds of edges: the N - 1 bonds, and every non-bonded pair (|i - j| > 1). A non-bonded edge counts only when its beads are closer than rc = 1.5; a smooth cosine cutoff c(r) switches it off continuously at rc.")
I('''topo = ChainTopology[32];
x = frames[[-1, 1]];
near = Select[topo["Pairs"], Norm[x[[#[[1]]]] - x[[#[[2]]]]] < $PNN["rc"] &];
Graph3D[Range[32], Join[UndirectedEdge @@@ topo["Bonds"], UndirectedEdge @@@ near],
 VertexCoordinates -> x, EdgeStyle -> Join[Thread[(UndirectedEdge @@@ topo["Bonds"]) -> Directive[Thick, Black]], Thread[(UndirectedEdge @@@ near) -> Directive[Orange, Dashed]]],
 PlotLabel -> "Bonds (black) and non-bonded edges inside rc (orange)"]''')
T("Each edge p joining beads i and j carries three things: its length r, the unit vector u = (x_i - x_j)/r, and a radial basis ρ_k(r) = exp(-γ (r - μ_k)^2) on 16 centres μ_k between 0.6 and 1.5. The incidence matrix A has, in column p, +1 at bead i and -1 at bead j.")
I('MatrixPlot[topo["Ab"], PlotLabel -> "Bond incidence matrix A (32 × 31)"]')

C("Section", "4. The network")
T("The message on edge p is a force vector m_p = φ(ρ(r_p)) c(r_p) u_p, where φ is a small multilayer perceptron (16 → 32 → 32 → 1, tanh). There is one φ for bonds and one for non-bonded edges. The force on each bead is F = A.m.")
T("Three properties follow from the construction, not from training. (1) Invariance and equivariance: φ sees only distances, so rotating the chain rotates the forces. (2) Newton's third law: each column of A sums to zero, so the total force is exactly zero. (3) Conservative forces: a central pair force f(r) u is minus the gradient of U(r) = ∫_r^∞ f(s) ds, so energy can be conserved in MD.")
T("A repulsive core U0 = (0.8/r)^12 is added to every edge as a fixed prior, and φ learns only the remainder F_ref - F_prior. The networks also see a clamped distance: never below 0.88 (closest approach in the data is about 0.93) and, for bonds, never above 1.45 (the longest bond seen is about 1.44). Without these two guards, a long run eventually visits distances the network never saw, the extrapolated forces are wrong, and the chain either collapses onto itself or breaks a bond. This first happened at N = 32 and N = 64 during development.")
I('net0 = BuildForceNet[topo]')
C("Subsection", "4.1 The computational graph")
T("ForceNetGraphPlot draws the network as a computational graph. It reads the shape on every edge from the network itself, so it is right for any chain length. Colours: blue = input ports (computed from the bead positions), orange = the two learned edge networks (the only trainable weights), grey = frozen arrays (the incidence matrices, learning rate 0), green = fixed operations without weights, red = the output port. Edge labels are tensor shapes for one frame.")
I('ForceNetGraphPlot[net0]')
T("Reading it from left to right. Each branch starts from the radial basis of its edges: bf (31×16) for the 31 bonds, nf (465×16) for the 465 non-bonded pairs. phiB and phiN are NetMapOperators around the same small chain (16→32→tanh→32→tanh→1); the same weights are applied to every edge, giving one scalar pair force per edge (31 and 465 numbers). On the non-bonded branch, cut multiplies it by the cutoff c(r) from nc. repB and repN copy each scalar three times, and mB and mN multiply by the unit vectors bu and nu, giving the message m = f u on every edge (P×3). sumB and sumN are DotLayers with the incidence matrices Ab (32×31) and An (32×465), which add each message to its two beads with opposite signs. total adds the two branches into F, the 32×3 forces.")
T("Parameter count: each edge network has 16·32 + 32 + 32·32 + 32 + 32 + 1 = 1633 weights, so the whole model has 3266. The shapes for N beads are (N − 1) bonds and (N − 1)(N − 2)/2 pairs; try ForceNetGraphPlot[BuildForceNet[ChainTopology[64]]].")
T("To save the figure at print resolution next to this notebook:")
I('Export[FileNameJoin[{NotebookDirectory[], "force-net-graph.png"}], ForceNetGraphPlot[net0], ImageResolution -> 300]')

C("Section", "5. Training by force matching")
T("The loss is the mean squared error between predicted and reference forces over all beads and frames. NetTrain does the differentiation. The two incidence matrices are frozen (learning-rate multiplier 0), so only the two φ networks learn. Adam is used, with learning rate 0.003, cut by a factor of 10 for the last 30% of training. Allow about 3 to 4 minutes on a laptop CPU.")
I('''SeedRandom[11];
res = TrainForceNet[frames, test32, BatchSize -> 32, MaxTrainingRounds -> 300, TimeGoal -> 240,
   LearningRate -> 0.003, Method -> {"ADAM", "LearningRateSchedule" -> (If[#1 < 0.7 #2, 1., 0.1] &)},
   TrainingProgressReporting -> "Panel"];
net = res["TrainedNet"];
ListLogPlot[{res["RoundLossList"], res["ValidationLossList"]}, Joined -> True, PlotLegends -> {"training", "validation"}, AxesLabel -> {"round", "MSE"}]''')

C("Section", "6. What the network learned")
T("Because each φ is a function of one variable, it can be drawn and compared with the exact pair forces. Positive values push the pair apart.")
I('''pf = PairForceFunctions[net];
trueN[r_] := UnitStep[2^(1/6) - r] 24 (2 r^-13 - r^-7);
trueB[r_] := -$PNN["kBond"] (r - $PNN["r0"]) + trueN[r];
rr = Subdivide[0.85, 1.5, 300];
GraphicsRow[{
  ListLinePlot[{Transpose[{rr, pf["Bond"][rr]}], Transpose[{rr, trueB /@ rr}]}, PlotStyle -> {Thick, Dashed}, PlotLegends -> {"learned", "exact"}, PlotLabel -> "Bond pair force", AxesLabel -> {"r", "f"}],
  ListLinePlot[{Transpose[{rr, pf["NonBond"][rr]}], Transpose[{rr, trueN /@ rr}]}, PlotStyle -> {Thick, Dashed}, PlotLegends -> {"learned", "exact"}, PlotLabel -> "Non-bonded pair force", AxesLabel -> {"r", "f"}]}, ImageSize -> 800]''')
T("Force errors on held-out frames. LearnedForces works for any chain length, because the same φ is applied to every edge. The 64-bead test is a length the network never saw.")
I('''lf = LearnedForces[net];
p32 = lf /@ test32[[All, 1]]; p64 = lf /@ test64[[All, 1]];
<|"RMSE N=32" -> ForceRMSE[p32, test32[[All, 2]]], "RMSE N=64" -> ForceRMSE[p64, test64[[All, 2]]],
  "RMS force" -> Sqrt[Mean[Flatten[test64[[All, 2]]^2]]], "max |total force|" -> Max[Abs[Total[p64, {2}]]]|>''')

C("Section", "7. Molecular dynamics with the learned forces")
T("For speed, the two learned pair forces are sampled on 4001 points between 0.6 and 1.6, and a compiled integrator interpolates them, as LAMMPS does with tabulated potentials. The table reproduces the network to about 10^-5. Potentials come from integrating the tabulated force.")
I('''tab = TabulateForceField[net]; tf = TabulatedForces[tab]; te = TabulatedEnergy[tab];
ForceRMSE[tf /@ test64[[All, 1]], p64]''')
T("NVE test: start one frame with Maxwell velocities and integrate with velocity Verlet (dt = 0.002) for 5000 steps. Do it once with exact forces and once with learned forces. The constant in the learned energy differs from the exact one, because a force fixes a potential only up to a constant, so the fluctuations are what to compare.")
I('''SeedRandom[5]; x0 = test32[[50, 1]];
v0 = RandomVariate[NormalDistribution[], {32, 3}]; v0 -= ConstantArray[Mean[v0], 32];
nveRef = VelocityVerletNVE[x0, v0, 5000, 0.002, ReferenceForces, ReferenceEnergy];
nveNN = VelocityVerletNVE[x0, v0, 5000, 0.002, tf, te];
ListLinePlot[{{#1, #2 - nveRef[[1, 2]]} & @@@ nveRef, {#1, #2 - nveNN[[1, 2]]} & @@@ nveNN},
 PlotLegends -> {"exact forces", "learned forces"}, AxesLabel -> {"t", "E(t) - E(0)"}, PlotRange -> All]''')
T("Langevin runs at N = 16, 32 and 64 with the reference and with the learned field: 4 chains each, 20 000 equilibration steps, 500 frames every 200 steps.")
I('''nnRun = TabulatedLangevin[tab]; SeedRandom[99];
md = Association@Flatten@Table[{{n, "ref"} -> GenerateData[n, 4, 20000, 500, 200],
     {n, "nn"} -> GenerateData[n, 4, 20000, 500, 200, nnRun]}, {n, {16, 32, 64}}];''')
I('''blk[v_] := StandardDeviation[Mean /@ Partition[v, 100]]/Sqrt[Length[v]/100];
stats[n_, k_] := Module[{fr = md[{n, k}][[All, 1]], rg2},
  rg2 = RadiusOfGyration[#]^2 & /@ fr;
  {Sqrt[Mean[rg2]], blk[rg2]/(2 Sqrt[Mean[rg2]]), Mean[Flatten[BondLengths /@ fr]], Mean[Flatten[BondAngles /@ fr]]/Degree}];
Grid[Prepend[Flatten[Table[{n, k, Sequence @@ (NumberForm[#, 4] & /@ stats[n, k])}, {n, {16, 32, 64}}, {k, {"ref", "nn"}}], 1],
  {"N", "forces", Subscript["R", "g"], "±", "<bond>", "<angle>°"}], Frame -> All]''')
I('''GraphicsRow[{
  Histogram[{Flatten[BondLengths /@ md[{64, "ref"}][[All, 1]]], Flatten[BondLengths /@ md[{64, "nn"}][[All, 1]]]}, {0.8, 1.45, 0.01}, "PDF", ChartLegends -> {"reference", "learned"}, PlotLabel -> "Bond lengths, N = 64"],
  Histogram[{Flatten[BondAngles /@ md[{64, "ref"}][[All, 1]]]/Degree, Flatten[BondAngles /@ md[{64, "nn"}][[All, 1]]]/Degree}, {40, 180, 2}, "PDF", ChartLegends -> {"reference", "learned"}, PlotLabel -> "Bond angles, N = 64"]}, ImageSize -> 800]''')
I('''Table[k -> LinearModelFit[Table[{Log[n], Log[stats[n, k][[1]]]}, {n, {16, 32, 64}}], s, s]["BestFitParameters"][[2]], {k, {"ref", "nn"}}]''')
T("The slope of log R_g against log N is the effective exponent ν. For long self-avoiding chains it tends to 0.588. These chains are short, start straight, and the N = 64 runs are short compared with their relaxation time, so both fits come out higher (about 0.72). What matters here is that the learned and reference forces give the same R_g at each N, within the error bars.")

C("Section", "8. Limits")
T("• The networks are pair networks: m_p depends only on r_p. That is exact for this reference model (bonds + WCA), and it is also the theorem in notebook 01: any invariant network whose messages depend only on one edge gives a sum of pair forces. Bond-angle or torsion potentials would need three- and four-body messages (as in NequIP or MACE).\n• The networks are trusted only on the distances seen in the data; the core prior and the clamps handle the rest. Retrain with a hotter or denser data set to widen that range.\n• Everything runs on CPU. A 32-bead model trains in about 3 minutes; a Langevin step of the 64-bead chain with the tabulated field takes about 25 µs with the C compiler.")

from nbtext import write
write(cells, "PolymerNN-Wolfram.nb")
print(len(cells))
