(* ::Package:: *)

(* ::Title:: *)
(*PolymerNN: a graph-network force field for bead-spring polymers*)

(* ::Text:: *)
(*\[Copyright] 2026 Computational Classnotes \[CenterDot] CC BY-NC-SA 4.0*)
(*Authors: Dara O Shayda \[CenterDot] Noor Hoshino*)

(* ::Text:: *)
(*This package is the Wolfram Language edition of polymer-neural-nets.*)
(*It contains everything needed to*)
(*  1. simulate a reference bead-spring chain (harmonic bonds + WCA repulsion) with Langevin dynamics,*)
(*  2. turn each frame into a graph: beads are nodes, bonds and close non-bonded pairs are edges,*)
(*  3. build an invariant message-passing network whose edge messages are scalar pair forces,*)
(*  4. train it by force matching (the loss compares predicted and reference forces on every bead),*)
(*  5. drive new molecular dynamics with the learned forces, for chains of any length,*)
(*  6. check the physics: force error, NVE energy conservation, bond and angle distributions, R_g ~ N^\[Nu].*)
(**)
(*Units are reduced Lennard-Jones units: \[Epsilon] = \[Sigma] = m = k_B = 1, temperature T = 1.*)

BeginPackage["PolymerNN`"];

(* ---------------- public symbols ---------------- *)

$PNN::usage = "$PNN is an Association of the model and simulation parameters.";
ReferenceForces::usage = "ReferenceForces[x] gives the exact forces (n\[Times]3) on the beads of chain x (n\[Times]3).";
ReferenceEnergy::usage = "ReferenceEnergy[x] gives the exact potential energy of chain x.";
StraightChain::usage = "StraightChain[n] gives n beads on a line, spaced by the bond length.";
LangevinRun::usage = "LangevinRun[x0, steps, every, forceFn] runs BAOAB Langevin dynamics and returns {{x, F}, ...} every 'every' steps. forceFn defaults to the compiled reference forces.";
GenerateData::usage = "GenerateData[n, chains, equil, frames, every] returns reference frames {x, F} from independent chains.";
ChainTopology::usage = "ChainTopology[n] gives the graph of an n-bead chain: bond list, non-bonded pair list and their incidence matrices.";
EdgeFeatures::usage = "EdgeFeatures[x, pairs] gives {radial basis (P\[Times]K), unit vectors (P\[Times]3), cutoff weights (P)} for the listed pairs.";
FrameToInput::usage = "FrameToInput[x, F, topo] turns one frame into the Association of tensors fed to the network.";
BuildForceNet::usage = "BuildForceNet[topo] gives the untrained force network for the chain length in topo.";
FramesToData::usage = "FramesToData[frames] turns a list of frames {x, F} into network training data.";
TrainForceNet::usage = "TrainForceNet[frames, valFrames, opts] trains a force network on frames {x, F} by force matching; returns the NetTrain results object.";
PairForceFunctions::usage = "PairForceFunctions[net] gives <|\"Bond\"->fB, \"NonBond\"->fN|>, the learned scalar pair forces as functions of r.";
LearnedForces::usage = "LearnedForces[net] gives a function x \[RightTeeArrow] forces that applies the trained edge networks to a chain of any length.";
TabulateForceField::usage = "TabulateForceField[net] samples the learned pair forces and potentials on a fine grid, for fast compiled MD.";
TabulatedForces::usage = "TabulatedForces[tab] gives a compiled force function that interpolates the table.";
TabulatedLangevin::usage = "TabulatedLangevin[tab] gives a compiled Langevin runner driven by the tabulated learned forces; pass it as the last argument of LangevinRun or GenerateData.";
TabulatedEnergy::usage = "TabulatedEnergy[tab] gives the potential-energy function of the learned force field.";
VelocityVerletNVE::usage = "VelocityVerletNVE[x0, v0, steps, dt, forceFn, energyFn] runs constant-energy MD and returns the total energy at each step.";
RadiusOfGyration::usage = "RadiusOfGyration[x] gives the radius of gyration of chain x.";
BondLengths::usage = "BondLengths[x] gives the bond lengths of chain x.";
BondAngles::usage = "BondAngles[x] gives the bond angles (radians) of chain x.";
PriorForces::usage = "PriorForces[x] gives the forces of the fixed repulsive core prior.";
ForceNetGraphPlot::usage = "ForceNetGraphPlot[net] draws the computational graph of a force network built by BuildForceNet, coloured by node kind, with the tensor shape of every edge read from the network.";
ForceRMSE::usage = "ForceRMSE[pred, ref] gives the root-mean-square force error.";

Begin["`Private`"];

(* ---------------- parameters ---------------- *)

(* kBond, r0 : harmonic bond U = kBond/2 (r - r0)^2
   WCA       : U = 4 (r^-12 - r^-6) + 1 for r < 2^(1/6), zero beyond, between every pair of beads
   kT, gamma, dt : Langevin thermostat and time step
   mu, gw    : centres and width of the Gaussian radial basis fed to the edge networks
   rc        : smooth cutoff for non-bonded edges; pairs beyond rc are not edges of the graph
   rClamp    : the networks see max(r, rClamp). Reference data never bring two beads closer than
               about 0.9, so a network asked about smaller r would be extrapolating blindly.
   rBondMax  : likewise the bond network sees min(r, rBondMax); the longest bond in the data is
               about 1.44. Without this clamp the bond network, fed a stretch it never saw, can
               return a repulsive force and break the chain (this happened at N = 64).
   sigmaCore : a fixed repulsive core U = (sigmaCore/r)^12 is added to every edge (the "prior").
               The networks learn only the remainder F_ref - F_prior (delta learning). Inside the
               data range the prior is small and the networks correct it; below the data range the
               prior guarantees that beads cannot pass through each other. *)
$PNN = <|"kBond" -> 100., "r0" -> 1., "kT" -> 1., "gamma" -> 1., "dt" -> 0.005,
   "rc" -> 1.5, "mu" -> N@Subdivide[0.6, 1.5, 15], "hidden" -> 32,
   "rClamp" -> 0.88, "rBondMax" -> 1.45, "sigmaCore" -> 0.8|>;
$PNN["gw"] = 1/($PNN["mu"][[2]] - $PNN["mu"][[1]])^2;

(* ---------------- reference model ---------------- *)

(* Compile to C when a C compiler is installed (much faster), otherwise to the Wolfram VM. *)
Needs["CCompilerDriver`"];
$CompileTarget = If[Length[CCompilerDriver`CCompilers[]] > 0, "C", "WVM"];

(* Compiled reference forces. The double loop visits every pair once (Newton's third law
   gives the partner's force for free); the second loop adds the bonds. *)
cForce = Compile[{{x, _Real, 2}, {kB, _Real}, {r0, _Real}},
   Module[{n = Length[x], ff = Table[0., {Length[x]}, {3}], d = {0., 0., 0.}, r2 = 0.,
     ir2 = 0., ir6 = 0., f = 0., r = 0., rc2 = 2.^(1/3)},
    Do[d = x[[i]] - x[[j]]; r2 = d.d;
     If[r2 < rc2, ir2 = 1./r2; ir6 = ir2^3; f = 24. ir6 (2. ir6 - 1.) ir2;
      ff[[i]] += f d; ff[[j]] -= f d], {i, 1, n - 1}, {j, i + 1, n}];
    Do[d = x[[i + 1]] - x[[i]]; r = Sqrt[d.d]; f = kB (r - r0)/r;
     ff[[i]] += f d; ff[[i + 1]] -= f d, {i, 1, n - 1}];
    ff], CompilationTarget -> $CompileTarget, RuntimeOptions -> "Speed"];

ReferenceForces[x_] := cForce[x, $PNN["kBond"], $PNN["r0"]];

ReferenceEnergy[x_] := Module[{n = Length[x], d, r, wca, rb},
  d = Outer[Subtract, x, x, 1]; r = Sqrt[Total[d^2, {3}]] + IdentityMatrix[n];
  wca = UnitStep[2.^(1/6) - r] (4 (r^-12 - r^-6) + 1) UpperTriangularize[ConstantArray[1., {n, n}], 1];
  rb = Sqrt[Total[Differences[x]^2, {2}]];
  Total[wca, 2] + Total[$PNN["kBond"]/2 (rb - $PNN["r0"])^2]];

StraightChain[n_] := N@Table[{i $PNN["r0"], 0, 0}, {i, 0, n - 1}];

(* BAOAB Langevin integrator (Leimkuhler-Matthews). B: half kick, A: half drift,
   O: exact Ornstein-Uhlenbeck velocity update, then A and B again.
   Gaussian noise by Box-Muller so that the whole loop compiles. *)
cLangevin[force_] := Compile[{{x0, _Real, 2}, {steps, _Integer}, {every, _Integer},
    {dt, _Real}, {gam, _Real}, {kT, _Real}},
   Module[{x = x0, v = 0. x0, ff = force[x0], c1 = Exp[-gam dt], c2 = 0., n = Length[x0],
     nz = Table[0., {Length[x0]}, {3}],
     out = Table[0., {Quotient[steps, every]}, {2}, {Length[x0]}, {3}], k = 0},
    c2 = Sqrt[kT (1 - c1^2)];
    Do[v += 0.5 dt ff; x += 0.5 dt v;
     nz = Sqrt[-2. Log[RandomReal[{10.^-300, 1.}, {n, 3}]]] Cos[2. Pi RandomReal[1., {n, 3}]];
     v = c1 v + c2 nz; x += 0.5 dt v; ff = force[x]; v += 0.5 dt ff;
     If[Mod[s, every] == 0, k++; out[[k, 1]] = x; out[[k, 2]] = ff], {s, steps}];
    out],
   CompilationOptions -> {"InlineCompiledFunctions" -> True, "InlineExternalDefinitions" -> True},
   CompilationTarget -> $CompileTarget, RuntimeOptions -> "Speed"];

cRefForce = With[{kB = $PNN["kBond"], r0 = $PNN["r0"]}, Compile[{{x, _Real, 2}}, cForce[x, kB, r0],
   CompilationOptions -> {"InlineCompiledFunctions" -> True, "InlineExternalDefinitions" -> True}]];
refLangevin := refLangevin = cLangevin[cRefForce];

LangevinRun[x0_, steps_, every_, runner_: Automatic] :=
  If[runner === Automatic, refLangevin, runner][x0, steps, every, $PNN["dt"], $PNN["gamma"], $PNN["kT"]];

(* Each chain starts straight, is equilibrated for 'equil' steps, then sampled. *)
GenerateData[n_, chains_, equil_, frames_, every_, runner_: Automatic] := Flatten[Table[
    Module[{eq = LangevinRun[StraightChain[n], equil, equil, runner]},
     LangevinRun[eq[[-1, 1]], frames every, every, runner]], {chains}], 1];

(* ---------------- the graph ---------------- *)

(* Radial basis of the clamped distance, and the smooth cosine cutoff. *)
rbf[r_, hi_ : Infinity] := Exp[-Clip[$PNN["gw"] Outer[Subtract, Clip[r, {$PNN["rClamp"], hi}], $PNN["mu"]]^2, {0., 600.}]];
cutoff[r_] := UnitStep[$PNN["rc"] - r] 0.5 (Cos[Pi r/$PNN["rc"]] + 1);

(* Prior pair force f0 = -dU0/dr with U0 = (sigmaCore/r)^12; cut off smoothly on non-bonded edges. *)
priorB[r_] := 12 $PNN["sigmaCore"]^12 r^-13;
priorN[r_] := priorB[r] cutoff[r];

(* Forces on all beads from two scalar pair-force functions (bonded fB, non-bonded fN).
   Positive f pushes the pair apart. Works for any chain length; topologies are cached. *)
pairForceField[fB_, fN_] := Module[{cache = <||>},
  Function[x, Module[{n = Length[x], topo, d, r},
    topo = Lookup[cache, n, cache[n] = Module[{t = ChainTopology[n]},
        <|"Bonds" -> t["Bonds"], "Pairs" -> t["Pairs"], "Ab" -> SparseArray[t["Ab"]], "An" -> SparseArray[t["An"]]|>]];
    d = x[[topo["Bonds"][[All, 1]]]] - x[[topo["Bonds"][[All, 2]]]]; r = Sqrt[Total[d^2, {2}]];
    topo["Ab"].(fB[r] d/r) +
     (d = x[[topo["Pairs"][[All, 1]]]] - x[[topo["Pairs"][[All, 2]]]]; r = Sqrt[Total[d^2, {2}]];
      topo["An"].(fN[r] d/r))]]];

priorField := priorField = pairForceField[priorB, priorN];
PriorForces[x_] := priorField[x];

(* Incidence matrix A (n \[Times] P): column p has +1 at the first bead of pair p and -1 at
   the second. If m_p is the force vector carried by edge p (acting on the first bead),
   then A.m adds +m_p to bead i and -m_p to bead j: Newton's third law is built in. *)
incidence[n_, pr_] := Normal@SparseArray[
   Flatten[MapIndexed[{{#1[[1]], #2[[1]]} -> 1., {#1[[2]], #2[[1]]} -> -1.} &, pr], 1], {n, Length[pr]}];

ChainTopology[n_] := Module[{b = Table[{i, i + 1}, {i, n - 1}],
   p = Select[Subsets[Range[n], {2}], #[[2]] - #[[1]] > 1 &]},
  <|"n" -> n, "Bonds" -> b, "Pairs" -> p, "Ab" -> incidence[n, b], "An" -> incidence[n, p]|>];

(* Edge features. Only the distance r enters the networks (through a Gaussian radial basis),
   so the scalar messages are invariant under rotations and translations; multiplying by the
   unit vector u = (x_i - x_j)/r makes the forces equivariant. The cosine cutoff sends
   non-bonded edges smoothly to zero at rc. *)
EdgeFeatures[x_, pr_, hi_ : Infinity] := Module[{d = x[[pr[[All, 1]]]] - x[[pr[[All, 2]]]], r},
  r = Sqrt[Total[d^2, {2}]];
  {rbf[r, hi], d/r, cutoff[r]}];

(* The training target is the residual force F_ref - F_prior. *)
FrameToInput[x_, F_, topo_] := Module[{b = EdgeFeatures[x, topo["Bonds"], $PNN["rBondMax"]], p = EdgeFeatures[x, topo["Pairs"]]},
  <|"bf" -> b[[1]], "bu" -> b[[2]], "nf" -> p[[1]], "nu" -> p[[2]], "nc" -> p[[3]], "F" -> F - PriorForces[x]|>];

(* ---------------- the network ---------------- *)

(* phi: radial basis (K) -> hidden -> hidden -> scalar. One phi for bonds, one for non-bonded edges. *)
edgeMLP[h_] := NetChain[{LinearLayer[h], ElementwiseLayer[Tanh], LinearLayer[h],
    ElementwiseLayer[Tanh], LinearLayer[{}]}];

(* The graph network for one chain length:
     message on edge p:   m_p = phi(r_p) c(r_p) u_p        (a 3-vector)
     force on bead i:     F_i = sum_p A_ip m_p             (A = incidence matrix)
   NetMapOperator applies the same phi to every edge (weight sharing), so the trained phi
   is independent of n and can be reused on chains of any length. *)
BuildForceNet[topo_] := Module[{nb = Length[topo["Bonds"]], np = Length[topo["Pairs"]],
   K = Length[$PNN["mu"]], h = $PNN["hidden"]},
  NetGraph[<|
    "phiB" -> NetMapOperator[edgeMLP[h]], "phiN" -> NetMapOperator[edgeMLP[h]],
    "cut" -> ThreadingLayer[Times],
    "repB" -> ReplicateLayer[3, 2], "repN" -> ReplicateLayer[3, 2],
    "mB" -> ThreadingLayer[Times], "mN" -> ThreadingLayer[Times],
    "Ab" -> NetArrayLayer["Array" -> topo["Ab"], LearningRateMultipliers -> 0],
    "An" -> NetArrayLayer["Array" -> topo["An"], LearningRateMultipliers -> 0],
    "sumB" -> DotLayer[], "sumN" -> DotLayer[], "total" -> ThreadingLayer[Plus]|>,
   {NetPort["bf"] -> "phiB" -> "repB", {"repB", NetPort["bu"]} -> "mB", {"Ab", "mB"} -> "sumB",
    NetPort["nf"] -> "phiN", {"phiN", NetPort["nc"]} -> "cut" -> "repN",
    {"repN", NetPort["nu"]} -> "mN", {"An", "mN"} -> "sumN",
    {"sumB", "sumN"} -> "total" -> NetPort["F"]},
   "bf" -> {nb, K}, "bu" -> {nb, 3}, "nf" -> {np, K}, "nu" -> {np, 3}, "nc" -> np]];

(* Force matching: NetTrain attaches MeanSquaredLoss between the "F" output and the
   reference forces, i.e. L = < |F_pred - F_ref|^2 >. *)
FramesToData[frames_] := With[{topo = ChainTopology[Length[frames[[1, 1]]]]},
  Merge[FrameToInput[#[[1]], #[[2]], topo] & /@ frames, Identity]];

TrainForceNet[frames_, valFrames_List, opts : OptionsPattern[NetTrain]] :=
  NetTrain[BuildForceNet[ChainTopology[Length[frames[[1, 1]]]]], FramesToData[frames], All,
   ValidationSet -> FramesToData[valFrames], opts, TargetDevice -> "CPU"];
TrainForceNet[frames_, opts : OptionsPattern[NetTrain]] :=
  NetTrain[BuildForceNet[ChainTopology[Length[frames[[1, 1]]]]], FramesToData[frames], All, opts, TargetDevice -> "CPU"];

(* ---------------- using the trained network ---------------- *)

(* The full learned pair forces: network residual + prior. *)
PairForceFunctions[net_] := With[{pB = NetExtract[net, {"phiB", "Net"}], pN = NetExtract[net, {"phiN", "Net"}]},
  <|"Bond" -> Function[r, With[{rl = Flatten[{r}]}, pB[rbf[rl, $PNN["rBondMax"]]] + priorB[rl]]],
    "NonBond" -> Function[r, With[{rl = Flatten[{r}]}, (pN[rbf[rl]] + priorB[rl]) cutoff[rl]]]|>];

(* Learned forces for any n: build the edges, run phi on all of them in one batch, and add
   the messages to the beads with sparse incidence matrices. *)
LearnedForces[net_] := With[{pf = PairForceFunctions[net]}, pairForceField[pf["Bond"], pf["NonBond"]]];

(* Tabulation: phi is a smooth function of one variable, so sampling it on a fine grid and
   interpolating linearly reproduces the network to high accuracy and runs as compiled code.
   Potentials come from integrating the force inward from a point where it vanishes:
   U(r) = Integral_r^rmax f(s) ds (trapezoid rule on the grid). *)
TabulateForceField[net_, rmin_ : 0.6, rmax_ : 1.6, m_ : 4001] := Module[{pf = PairForceFunctions[net], r, fB, fN, uB, uN, dr},
  r = N@Subdivide[rmin, rmax, m - 1]; dr = r[[2]] - r[[1]];
  fB = pf["Bond"][r]; fN = pf["NonBond"][r];
  fN = fN UnitStep[$PNN["rc"] - r];
  uB = Reverse@Accumulate[Prepend[Reverse[ListConvolve[{0.5, 0.5}, fB]] dr, 0.]];
  uN = Reverse@Accumulate[Prepend[Reverse[ListConvolve[{0.5, 0.5}, fN]] dr, 0.]];
  <|"rmin" -> rmin, "dr" -> dr, "r" -> r, "fB" -> fB, "fN" -> fN, "uB" -> uB, "uN" -> uN|>];

cTabForce = Compile[{{x, _Real, 2}, {rmin, _Real}, {dr, _Real}, {fB, _Real, 1}, {fN, _Real, 1}, {rc, _Real}},
   Module[{n = Length[x], ff = Table[0., {Length[x]}, {3}], d = {0., 0., 0.}, r = 0., s = 0., k = 1,
     w = 0., f = 0., m = Length[fB], rc2 = rc^2},
    Do[d = x[[i]] - x[[j]]; r = d.d;
     If[j == i + 1 || r < rc2,
      r = Sqrt[r]; s = (r - rmin)/dr; k = Max[1, Min[m - 1, Floor[s] + 1]]; w = s - (k - 1);
      f = If[j == i + 1, (1 - w) fB[[k]] + w fB[[k + 1]], (1 - w) fN[[k]] + w fN[[k + 1]]];
      ff[[i]] += f d/r; ff[[j]] -= f d/r], {i, 1, n - 1}, {j, i + 1, n}];
    ff], CompilationTarget -> $CompileTarget, RuntimeOptions -> "Speed"];

TabulatedForces[tab_] := With[{a = tab["rmin"], b = tab["dr"], fB = tab["fB"], fN = tab["fN"], rc = $PNN["rc"]},
  Function[x, cTabForce[x, a, b, fB, fN, rc]]];

(* A Langevin runner whose force is the tabulated network, compiled as one loop. *)
TabulatedLangevin[tab_] := With[{a = tab["rmin"], b = tab["dr"], fB = tab["fB"], fN = tab["fN"], rc = $PNN["rc"]},
  cLangevin[Compile[{{x, _Real, 2}}, cTabForce[x, a, b, fB, fN, rc],
    CompilationOptions -> {"InlineCompiledFunctions" -> True, "InlineExternalDefinitions" -> True}]]];

TabulatedEnergy[tab_] := With[{uB = Interpolation[Transpose[{tab["r"], tab["uB"]}], InterpolationOrder -> 1],
    uN = Interpolation[Transpose[{tab["r"], tab["uN"]}], InterpolationOrder -> 1], rc = $PNN["rc"]},
  Function[x, Module[{n = Length[x], d, r},
    d = Outer[Subtract, x, x, 1]; r = Sqrt[Total[d^2, {3}]];
    Total[uB[Diagonal[r, 1]]] + Total[uN[Select[Flatten[UpperTriangularize[r, 2]], 0 < # < rc &]]]]]];

(* Velocity Verlet at constant energy: the test that the learned forces are conservative. *)
VelocityVerletNVE[x0_, v0_, steps_, dt_, force_, energy_, every_ : 10] := Module[{x = x0, v = v0, F = force[x0], out = {}},
  Do[v += 0.5 dt F; x += dt v; F = force[x]; v += 0.5 dt F;
   If[Mod[s, every] == 0, AppendTo[out, {s dt, energy[x] + 0.5 Total[v^2, 2]}]], {s, steps}];
  out];

(* ---------------- picture of the network ---------------- *)

(* Drawing of the force network's computational graph. Node colours mark the kind of node
   (input, learned network, frozen array, fixed operation, output); edge labels are the tensor
   shapes for one frame, read from the network itself, so the picture is right for any N. *)
$netColors = <|"input" -> {RGBColor["#DCE8F7"], RGBColor["#2F6DB5"]},
   "learned" -> {RGBColor["#FDE9CF"], RGBColor["#E07B00"]},
   "frozen" -> {RGBColor["#E9E9E9"], RGBColor["#6B6B6B"]},
   "op" -> {RGBColor["#DFF2E7"], RGBColor["#2E8B57"]},
   "output" -> {RGBColor["#F8DAD6"], RGBColor["#C0392B"]}|>;

ForceNetGraphPlot[net_NetGraph, opts : OptionsPattern[Graphics]] := Module[
  {ports, shp, sz, nodes, edges, box, arrow, nb, np, n, npar, mlpText},
  ports = Join[Information[net, "InputPorts"], Information[net, "OutputPorts"]];
  shp[name_] := Lookup[ports, name, Information[NetExtract[net, name], "OutputPorts"]["Output"]];
  sz[d_] := StringRiffle[ToString /@ Flatten[{d}], "\[Times]"];
  {nb, np} = {First[shp["bf"]], First[shp["nf"]]}; n = First[shp["F"]];
  npar = Information[NetExtract[net, "phiB"], "ArraysTotalElementCount"];
  mlpText[w_] := Row[{"MLP 16\[Rule]32\[Rule]tanh\[Rule]32\[Rule]tanh\[Rule]1\n", npar, " parameters, shared by all ", w}];
  (* name -> {kind, position, width, subtitle} *)
  nodes = <|
    "nc" -> {"input", {3.9, 3.1}, 2.0, "pair cutoff c(r)"},
    "nf" -> {"input", {0, 2}, 2.0, "pair radial basis \[Rho](r)"},
    "bf" -> {"input", {0, 1}, 2.0, "bond radial basis \[Rho](r)"},
    "bu" -> {"input", {6.0, -0.1}, 2.0, "bond unit vectors u"},
    "nu" -> {"input", {8.6, 3.1}, 2.0, "pair unit vectors u"},
    "phiN" -> {"learned", {3.3, 2}, 2.9, mlpText["pairs"]},
    "phiB" -> {"learned", {3.3, 1}, 2.9, mlpText["bonds"]},
    "An" -> {"frozen", {11.8, 3.1}, 2.2, "pair incidence matrix\nfrozen (learning rate 0)"},
    "Ab" -> {"frozen", {11.8, -0.1}, 2.2, "bond incidence matrix\nfrozen (learning rate 0)"},
    "cut" -> {"op", {6.6, 2}, 1.4, "\[CurlyPhi] \[Times] c(r)"},
    "repN" -> {"op", {9.0, 2}, 1.6, "replicate \[Times]3"},
    "repB" -> {"op", {6.6, 1}, 1.6, "replicate \[Times]3"},
    "mN" -> {"op", {11.6, 2}, 1.9, "message m = f u"},
    "mB" -> {"op", {9.0, 1}, 1.9, "message m = f u"},
    "sumN" -> {"op", {14.2, 2}, 1.9, "A\[CenterDot]m (sum per bead)"},
    "sumB" -> {"op", {14.2, 1}, 1.9, "A\[CenterDot]m (sum per bead)"},
    "total" -> {"op", {16.7, 1.5}, 1.9, "bond + non-bonded"},
    "F" -> {"output", {19.2, 1.5}, 1.8, "force on every bead"}|>;
  edges = {"bf" -> "phiB", "phiB" -> "repB", "repB" -> "mB", "bu" -> "mB", "mB" -> "sumB",
    "Ab" -> "sumB", "sumB" -> "total", "nf" -> "phiN", "phiN" -> "cut", "nc" -> "cut",
    "cut" -> "repN", "repN" -> "mN", "nu" -> "mN", "mN" -> "sumN", "An" -> "sumN",
    "sumN" -> "total", "total" -> "F"};
  box[name_ -> {kind_, {x_, y_}, w_, sub_}] := With[{c = $netColors[kind], h = 0.62},
    {EdgeForm[{AbsoluteThickness[2], c[[2]]}], c[[1]],
     Rectangle[{x - w/2, y - h/2}, {x + w/2, y + h/2}, RoundingRadius -> 0.12],
     Text[Style[name, Bold, 13, Black], {x, y + 0.14}],
     Text[Style[sub, 9, GrayLevel[0.15], LineSpacing -> {0.9, 0}], {x, y - 0.13}]}];
  arrow[a_ -> b_] := Module[{pa = nodes[a][[2]], pb = nodes[b][[2]], wa = nodes[a][[3]], wb = nodes[b][[3]], p, q, lab},
    (* from the right side of the source to the left side of the target *)
    {p, q} = {pa + {wa/2, 0}, pb - {wb/2, 0}};
    lab = sz[If[KeyExistsQ[ports, a], ports[a], shp[a]]];
    {If[b === "F", {RGBColor["#C0392B"], AbsoluteThickness[2.4]}, {GrayLevel[0.35], AbsoluteThickness[1.5]}],
     Arrowheads[0.009], Arrow[{p, q}],
     Text[Style[lab, 10, GrayLevel[0.2], Background -> White], (p + q)/2 + {0, 0.11}]}];
  Legended[
   Graphics[{arrow /@ edges, box /@ Normal[nodes]},
    opts, ImageSize -> 1600, PlotRange -> {{-1.2, 20.3}, {-0.6, 3.6}},
    PlotLabel -> Column[{Style["PolymerNN force network", Bold, 18],
       Style[Row[{"N = ", n, " beads \[CenterDot] ", nb, " bonds \[CenterDot] ", np, " non-bonded pairs \[CenterDot] ",
          2 npar, " trainable parameters"}], 12]}, Center],
    Epilog -> Text[Style["\[Copyright] 2026 Computational Classnotes \[CenterDot] CC BY-NC-SA 4.0", 9, Gray], Scaled[{0.99, 0.01}], {1, -1}]],
   SwatchLegend[Values[$netColors[[All, 1]]], {"input port", "learned network", "frozen array", "fixed operation", "output port"},
    LegendMarkers -> "Bubble", LegendMarkerSize -> 14]]];

(* ---------------- observables ---------------- *)

RadiusOfGyration[x_] := Sqrt[Mean[Total[(x - ConstantArray[Mean[x], Length[x]])^2, {2}]]];
BondLengths[x_] := Sqrt[Total[Differences[x]^2, {2}]];
BondAngles[x_] := Module[{b = Differences[x]},
  ArcCos[Clip[Total[-Most[b] Rest[b], {2}]/(Sqrt[Total[Most[b]^2, {2}]] Sqrt[Total[Rest[b]^2, {2}]]), {-1, 1}]]];
ForceRMSE[pred_, ref_] := Sqrt[Mean[Flatten[(pred - ref)^2]]];

End[];
EndPackage[];
