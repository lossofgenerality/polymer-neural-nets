# © 2026 Computational Classnotes · CC BY-NC-SA 4.0
# Builds NetLayers-Tutorial.nb: the network layers used in PolymerNN, one at a time, on a 4-bead chain.
cells = []
def C(style, text): cells.append((style, text))
T = lambda t: C("Text", t)
I = lambda t: C("Input", t)
S = lambda t: C("Section", t)

C("Title", "The Layers Behind a Graph Force Network")
C("Subtitle", "Every Wolfram Language layer used in PolymerNN, explained on a four-bead polymer")
C("Text", "© 2026 Computational Classnotes · CC BY-NC-SA 4.0\nAuthors: Dara O Shayda · Noor Hoshino")
T("The force network in PolymerNN.wl is built from a handful of standard layers: LinearLayer, ElementwiseLayer, NetChain, NetMapOperator, ThreadingLayer, ReplicateLayer, NetArrayLayer and DotLayer. They are joined in a NetGraph through NetPorts and trained with NetTrain. This notebook takes them one at a time. Each one is applied to a tiny chain of four beads, small enough that every number can be checked by hand, and each layer is compared with the same computation written in ordinary Wolfram Language.")
T("By the end, the pieces are assembled into a working mini force network, which is trained to learn a spring force from examples. It is the polymer network in miniature. This notebook does not need PolymerNN.wl.")
I('''Grid[Prepend[{
   {"LinearLayer", "W.x + b", "the three linear maps inside phiB and phiN"},
   {"ElementwiseLayer[Tanh]", "tanh of each entry", "the nonlinearity inside phiB and phiN"},
   {"NetChain", "layers one after another", "phiB and phiN themselves"},
   {"NetMapOperator", "the same net on every row", "applies phi to every edge"},
   {"ThreadingLayer[Times]", "entrywise product", "cut (phi times cutoff), mB and mN (f times u)"},
   {"ThreadingLayer[Plus]", "entrywise sum", "total (bond + non-bonded forces)"},
   {"ReplicateLayer[3, 2]", "copy each number 3 times", "repB and repN"},
   {"NetArrayLayer", "a stored array, optionally frozen", "Ab and An, the incidence matrices"},
   {"DotLayer", "matrix product", "sumB and sumN (A.m)"},
   {"NetGraph, NetPort", "wiring with named inputs and outputs", "the whole force network"}},
  Style[#, Bold] & /@ {"layer", "what it computes", "where PolymerNN uses it"}], Frame -> All, Alignment -> Left, Background -> {None, {LightBlue}}]''')

S("1. The example: a chain of four beads")
T("Four beads, three bonds (1-2, 2-3, 3-4) and three non-bonded pairs (1-3, 1-4, 2-4). These are the edges of the graph. For each edge we need its length r and its unit vector u = (x_i - x_j)/r.")
I('''x4 = {{0., 0., 0.}, {1., 0., 0.}, {0.55, 0.88, 0.}, {-0.25, 1.3, 0.5}};
bonds = {{1, 2}, {2, 3}, {3, 4}};
pairs = {{1, 3}, {1, 4}, {2, 4}};
edgeGeometry[x_, pr_] := Module[{d = x[[pr[[All, 1]]]] - x[[pr[[All, 2]]]], r},
   r = Sqrt[Total[d^2, {2}]]; {r, d/r}];
{rB, uB} = edgeGeometry[x4, bonds];
{rN, uN} = edgeGeometry[x4, pairs];
Grid[Prepend[Join[Transpose[{bonds, rB, uB}], Transpose[{pairs, rN, uN}]], {"edge", "r", "u"}], Frame -> All]''')
I('''Graphics3D[{{Gray, Tube[x4, 0.04]}, {Orange, Dashed, Line[x4[[#]]] & /@ pairs},
  {RGBColor[0.2, 0.4, 0.8], Sphere[x4, 0.12]}, MapIndexed[Text[Style[First[#2], Bold, 16], #1 + {0, 0, 0.25}] &, x4]},
 Boxed -> False, PlotLabel -> "bonds (grey) and non-bonded pairs (orange)"]''')
T("The networks never see r directly. They see a radial basis, the vector of Gaussians ρ_k(r) = exp(-γ (r - μ_k)^2). Here there are four centres μ_k, so each edge becomes a 4-vector, and the 3 bonds give a 3×4 matrix. PolymerNN uses 16 centres. The smooth cutoff c(r) is 1 at r = 0 and falls to 0 at r_c. Here r_c = 2, so all three pairs feel it (PolymerNN uses r_c = 1.5).")
I('''mu = {0.8, 1.0, 1.2, 1.4}; gw = 25.;
rbf[r_] := Exp[-gw Outer[Subtract, r, mu]^2];
cutoff[r_] := UnitStep[2. - r] 0.5 (Cos[Pi r/2.] + 1);
bfeat = rbf[rB]; nfeat = rbf[rN]; nc = cutoff[rN];
{MatrixForm[bfeat], MatrixForm[nfeat], nc}''')

S("2. LinearLayer")
T("LinearLayer[n] computes W.x + b with an n×d weight matrix W and a bias vector b of length n. Normally NetInitialize or NetTrain chooses the weights at random. Here we set them by hand, so the result can be checked against W.x + b.")
I('''W1 = {{0.5, -1., 0.2, 0.}, {0.1, 0.3, -0.4, 0.8}}; b1 = {0.05, -0.1};
lin = LinearLayer[2, "Weights" -> W1, "Biases" -> b1, "Input" -> 4]''')
I('''{lin[bfeat[[1]]], W1.bfeat[[1]] + b1}''')
T("Like every layer, a LinearLayer accepts a batch: a list of inputs gives a list of outputs.")
I('''lin[bfeat] - Map[W1.# + b1 &, bfeat] // Chop''')
T("LinearLayer[{}] gives a scalar, not a length-1 vector. It is the last layer of phi, which returns one pair force per edge. Its weights are a 1×d matrix.")
I('''lin0 = LinearLayer[{}, "Weights" -> {{1.5, -0.7}}, "Biases" -> {0.2}, "Input" -> 2];
{lin0[{1., 2.}], 1.5*1. - 0.7*2. + 0.2}''')

S("3. ElementwiseLayer")
T("ElementwiseLayer[f] applies f to every entry. Without it the network would be linear, however many layers it had. PolymerNN uses Tanh, which is smooth, so the learned forces are smooth functions of r.")
I('''tanh = ElementwiseLayer[Tanh];
{tanh[{-2., 0., 0.5, 3.}], Tanh[{-2., 0., 0.5, 3.}]}''')

S("4. NetChain")
T("NetChain puts layers one after another. Here it is a one-hidden-layer version of phi: radial basis (4) → linear (2) → tanh → linear (scalar). PolymerNN's phi has two hidden layers of width 32: 16 → 32 → tanh → 32 → tanh → 1.")
I('''phi = NetChain[{lin, ElementwiseLayer[Tanh], lin0}]''')
I('''phiByHand[v_] := First[{{1.5, -0.7}}.Tanh[W1.v + b1] + {0.2}];
{phi[bfeat[[1]]], phiByHand[bfeat[[1]]]}''')

S("5. NetMapOperator")
T("phi takes one edge, a 4-vector. The chain has several edges, a P×4 matrix. NetMapOperator[phi] applies phi to every row with the same weights, giving one number per edge. This weight sharing is what makes the trained network independent of the chain length: the number of parameters does not depend on P.")
I('''phiMap = NetMapOperator[phi];
{phiMap[bfeat], phiByHand /@ bfeat}''')
I('''{"parameters in phi" -> Information[phi, "ArraysTotalElementCount"],
 "parameters in NetMapOperator[phi], 3 edges" -> Information[NetMapOperator[phi, "Input" -> {3, 4}], "ArraysTotalElementCount"],
 "parameters in NetMapOperator[phi], 465 edges" -> Information[NetMapOperator[phi, "Input" -> {465, 4}], "ArraysTotalElementCount"]}''')
T("The scalar on each edge is a pair force f(r). A positive value pushes the two beads apart.")
I('''fB = phiMap[bfeat]; fN = phiMap[nfeat];
{fB, fN}''')

S("6. ThreadingLayer")
T("ThreadingLayer[g] combines several arrays entry by entry with g. Its inputs are ports Input1, Input2, and so on; given a list {a, b}, the first goes to Input1 and the second to Input2. PolymerNN uses it three times with Times and once with Plus.")
T("Times, first use (cut): the non-bonded pair force is multiplied by the cutoff, so it fades smoothly to zero at r_c.")
I('''times = ThreadingLayer[Times];
{times[{fN, nc}], fN nc}''')
T("Plus (total): the bond forces and the non-bonded forces on each bead are added.")
I('''plus = ThreadingLayer[Plus];
a = {{1., 2., 3.}, {4., 5., 6.}}; b = {{10., 20., 30.}, {40., 50., 60.}};
{plus[{a, b}], a + b}''')

S("7. ReplicateLayer")
T("A force is a 3-vector, but phi gives a scalar per edge. To multiply each scalar by its unit vector u (P×3), the scalar vector (length P) is first copied into a P×3 array. ReplicateLayer[3, 2] inserts a new level 2 of size 3, so row p becomes {f_p, f_p, f_p}.")
I('''rep = ReplicateLayer[3, 2];
{MatrixForm[rep[fB]], MatrixForm[Transpose[{fB, fB, fB}]]}''')
T("Compare level 1, the default: ReplicateLayer[3] makes three copies of the whole vector (3×P), which is not what we want.")
I('''{Dimensions[ReplicateLayer[3][fB]], Dimensions[ReplicateLayer[3, 2][fB]]}''')

S("8. The message on every edge")
T("Times, second use (mB, mN): the replicated scalars are multiplied by the unit vectors, giving the force vector m_p = f_p u_p that edge p exerts on its first bead.")
I('''mB = times[{rep[fB], uB}];
{MatrixForm[mB], MatrixForm[fB uB]}''')

S("9. NetArrayLayer")
T("NetArrayLayer stores an array inside the network. It has no input; its output is the array. If the array is a learnable parameter, it is trained like a weight. Setting LearningRateMultipliers -> 0 freezes it, so it is constant data that training cannot change. PolymerNN stores the incidence matrices this way.")
T("The incidence matrix A of the bonds has one row per bead and one column per edge. Column p has +1 at the first bead of edge p, -1 at the second, and 0 elsewhere.")
I('''incidence[n_, pr_] := Normal@SparseArray[
    Flatten[MapIndexed[{{#1[[1]], #2[[1]]} -> 1., {#1[[2]], #2[[1]]} -> -1.} &, pr], 1], {n, Length[pr]}];
Ab = incidence[4, bonds]; An = incidence[4, pairs];
{MatrixForm[Ab], MatrixForm[An]}''')
I('''arrB = NetArrayLayer["Array" -> Ab, LearningRateMultipliers -> 0];
{arrB[] == Ab, Dimensions[arrB[]]}''')

S("10. DotLayer")
T("DotLayer[] is the matrix product of its two inputs. A.m adds every edge message to its two beads: +m_p to the first and -m_p to the second, which is Newton's third law. The result is the force on each bead, a 4×3 array.")
I('''dot = DotLayer[];
FB = dot[{Ab, mB}];
{MatrixForm[FB], MatrixForm[Ab.mB]}''')
T("Every column of A sums to zero, so the forces on the chain always add up to zero, whatever phi has learned. (Networks compute in 32-bit floats, so zero means about 10^-7, hence the Chop.)")
I('''{Total[Ab], Chop[Total[FB], 10^-6]}''')

S("11. NetGraph and NetPort: assembling the force network")
T("A NetGraph names its layers and lists the connections. NetPort[\"name\"] is an input or output of the whole graph. {a, b} -> c sends a to Input1 and b to Input2 of c. This is the bond branch of PolymerNN's network for our four-bead chain: inputs bf (3×4) and bu (3×3), output F (4×3).")
I('''bondNet = NetGraph[<|
   "phiB" -> NetMapOperator[phi], "repB" -> ReplicateLayer[3, 2], "mB" -> ThreadingLayer[Times],
   "Ab" -> NetArrayLayer["Array" -> Ab, LearningRateMultipliers -> 0], "sumB" -> DotLayer[]|>,
  {NetPort["bf"] -> "phiB" -> "repB", {"repB", NetPort["bu"]} -> "mB", {"Ab", "mB"} -> "sumB" -> NetPort["F"]},
  "bf" -> {3, 4}, "bu" -> {3, 3}]''')
I('''{bondNet[<|"bf" -> bfeat, "bu" -> uB|>] - FB // Chop}''')
T("The full network adds the non-bonded branch, with the cutoff, and a Plus at the end. Apart from the sizes, this is exactly BuildForceNet in PolymerNN.wl.")
I('''forceNet[phiBond_, phiPair_] := NetGraph[<|
    "phiB" -> NetMapOperator[phiBond], "phiN" -> NetMapOperator[phiPair], "cut" -> ThreadingLayer[Times],
    "repB" -> ReplicateLayer[3, 2], "repN" -> ReplicateLayer[3, 2],
    "mB" -> ThreadingLayer[Times], "mN" -> ThreadingLayer[Times],
    "Ab" -> NetArrayLayer["Array" -> Ab, LearningRateMultipliers -> 0],
    "An" -> NetArrayLayer["Array" -> An, LearningRateMultipliers -> 0],
    "sumB" -> DotLayer[], "sumN" -> DotLayer[], "total" -> ThreadingLayer[Plus]|>,
   {NetPort["bf"] -> "phiB" -> "repB", {"repB", NetPort["bu"]} -> "mB", {"Ab", "mB"} -> "sumB",
    NetPort["nf"] -> "phiN", {"phiN", NetPort["nc"]} -> "cut" -> "repN",
    {"repN", NetPort["nu"]} -> "mN", {"An", "mN"} -> "sumN", {"sumB", "sumN"} -> "total" -> NetPort["F"]},
   "bf" -> {3, 4}, "bu" -> {3, 3}, "nf" -> {3, 4}, "nu" -> {3, 3}, "nc" -> 3];
fullNet = forceNet[phi, phi]''')
I('''F4 = fullNet[<|"bf" -> bfeat, "bu" -> uB, "nf" -> nfeat, "nu" -> uN, "nc" -> nc|>];
byHand = Ab.(fB uB) + An.((fN nc) uN);
{MatrixForm[F4], Max[Abs[F4 - byHand]], Chop[Total[F4], 10^-6]}''')
T("Information looks inside a network: its ports, its layers and its parameter count. NetExtract takes out a single layer.")
I('''{Information[fullNet, "InputPorts"], Information[fullNet, "OutputPorts"],
 Information[fullNet, "ArraysTotalElementCount"], NetExtract[fullNet, "cut"]}''')

S("12. NetTrain: learning a spring")
T("Now the bond branch learns a force from examples. The examples are chains of four beads joined by harmonic springs, U = (k/2)(r - 1)^2 with k = 50, so the exact pair force is f(r) = -50 (r - 1). The bond lengths are drawn around 1 and the forces on the beads are computed exactly. NetTrain gets only the bead forces, never f(r) itself; it has to work out the pair force behind them. This is force matching, as in PolymerNN.")
I('''kSpring = 50.;
randomChain[] := Accumulate[Prepend[Table[RandomReal[NormalDistribution[1., 0.12]] Normalize[RandomVariate[NormalDistribution[], 3]], {3}], {0., 0., 0.}]];
springForces[x_] := Module[{r, u}, {r, u} = edgeGeometry[x, bonds]; Ab.((-kSpring (r - 1.)) u)];
mu8 = N@Subdivide[0.6, 1.4, 7]; gw8 = 1/(mu8[[2]] - mu8[[1]])^2;
rbf8[r_] := Exp[-gw8 Outer[Subtract, r, mu8]^2];
SeedRandom[4]; chains = Table[randomChain[], {3000}];
data = <|"bf" -> (rbf8[First[edgeGeometry[#, bonds]]] & /@ chains), "bu" -> (Last[edgeGeometry[#, bonds]] & /@ chains),
   "F" -> (springForces /@ chains)|>;
Dimensions /@ data''')
T("The network to train: the bond branch again, now with 8 basis functions and phi = 8 → 16 → tanh → 1, starting from random weights.")
I('''springNet = NetGraph[<|
   "phiB" -> NetMapOperator[NetChain[{LinearLayer[16], ElementwiseLayer[Tanh], LinearLayer[{}]}]],
   "repB" -> ReplicateLayer[3, 2], "mB" -> ThreadingLayer[Times],
   "Ab" -> NetArrayLayer["Array" -> Ab, LearningRateMultipliers -> 0], "sumB" -> DotLayer[]|>,
  {NetPort["bf"] -> "phiB" -> "repB", {"repB", NetPort["bu"]} -> "mB", {"Ab", "mB"} -> "sumB" -> NetPort["F"]},
  "bf" -> {3, 8}, "bu" -> {3, 3}];
SeedRandom[5];
res = NetTrain[springNet, data, All, ValidationSet -> Scaled[0.1], BatchSize -> 64, LearningRate -> 0.01,
   MaxTrainingRounds -> 400, TimeGoal -> 30, TrainingProgressReporting -> None];
trained = res["TrainedNet"];
{"rounds" -> res["TotalRounds"], "RMS force error" -> Sqrt[res["FinalValidationLoss"]],
 "RMS force" -> Sqrt[Mean[Flatten[data["F"]^2]]]}''')
T("The trained phi is a function of one variable, so it can be drawn and compared with the exact spring force -50 (r - 1). The shaded band is the range of bond lengths in the data; outside it the network has no information.")
I('''phiLearned = NetExtract[trained, {"phiB", "Net"}];
rr = Subdivide[0.6, 1.4, 200]; rTrain = Flatten[First[edgeGeometry[#, bonds]] & /@ chains];
Show[Plot[-kSpring (r - 1), {r, 0.6, 1.4}, PlotStyle -> {Dashed, Black}],
 ListLinePlot[Transpose[{rr, phiLearned[rbf8[rr]]}], PlotStyle -> {Thick, RGBColor["#E07B00"]}],
 Graphics[{Opacity[0.12], Blue, Rectangle[{Quantile[rTrain, 0.005], -25}, {Quantile[rTrain, 0.995], 25}]}],
 PlotRange -> {-25, 25}, Frame -> True, FrameLabel -> {"r", "pair force f(r)"},
 PlotLabel -> "learned (orange) vs exact spring (dashed)", ImageSize -> 500]''')
T("The frozen NetArrayLayer really was frozen: training left the incidence matrix unchanged.")
I('''{Normal[NetExtract[trained, {"Ab", "Array"}]] == Ab, Normal[NetExtract[trained, {"Ab", "Array"}]] // MatrixForm}''')

S("13. From the toy to the polymer")
T("Every layer of PolymerNN's network has now appeared. The polymer version differs only in size and in the data:\n• 16 basis functions instead of 4 or 8, and phi = 16 → 32 → tanh → 32 → tanh → 1.\n• For N = 32 beads: 31 bonds and 465 non-bonded pairs instead of 3 and 3.\n• The reference forces come from Langevin simulations of a bead-spring chain with WCA repulsion, not from random springs.\n• A fixed repulsive core is added, and the input distance is clamped to the range of the data. Section 12 shows why: outside the shaded band the network is guessing.\nIn PolymerNN.wl, BuildForceNet builds the same graph as forceNet above, and ForceNetGraphPlot draws it.")

from nbtext import write
write(cells, "NetLayers-Tutorial.nb")
# Check script: every input cell, with its result collected
tests = "out = {};\n" + "\n".join(f"AppendTo[out, {k} -> Short[InputForm[({t})], 4]];" for k, (s, t) in enumerate(c for c in cells if c[0] == "Input")) + "\nColumn[out]"
open("layers_test.wl", "w").write(tests)  # all input cells in one file, for a quick check in a kernel
print(len(cells))
