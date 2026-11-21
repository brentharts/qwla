# Spectre phason flips as a substrate for the qubit network

What a tiling rearrangement does to the structure the network is built on. This is a sketch with finite computations, not a derivation of quantum gravity. Full write-up: `quantgrav.tex` in `spectre-monotile-py`; engine and checks: `SPECTRE_FLIPS.md` in `wada-coudene`.

Labels: **Exact**, **Counted** (one window class, one footprint), **Hypothesis**, **Negative**.

## 1. The substrate as a model set

A Spectre tiling is a point `z` of a 4-dimensional internal space. A tile is present at `z` iff its internal image lies in a window, and the windows are graph-directed fractals with ratio `1/lambda`, `lambda^2 = 4 + sqrt(15)`. A phason step moves `z` by a small `delta`; a **flip** is a step that changes which tiles are present. **Exact.**

The flip relation is symmetric under reversal, so flips alone carry no arrow of time. An arrow would have to come from an initial condition or from the inflation direction (the latter is not tested). **Exact; Negative for an arrow from flips.**

## 2. What a flip does to the network

All **Counted** (class 0, level 3, step 0.03):

- A flip is a cluster of tiles, not a single swap: about 10 tiles change per phase within radius 14, uniformly in distance; the clusters inspected have 8 to 17 tiles per side.
- The tiles meeting a fixed footprint change in 48% of the valid pairs. In 75% of those the contact graph is not isomorphic before and after. In the qubit network this means a flip can change which qubits are neighbours, not only which tiles carry them.
- The tetrad flux is zero for every flip, so geometry (flux) and connectivity (contact graph) disagree. This is not a magnetic-reconnection analogue: there is no field or non-ideal term on a planar tile. **Negative** for that reading.
- Area `n_O + 2 n_G` is conserved in a closed cluster while the number of Mystic pairs `n_G` changes (0.32 of 163 area-conserving events at `rho = 7`). The area ledger of the Hamiltonian is therefore respected by a flip, and the pair count (the qubit-carrying tiles) is not.

## 3. Parity sectors (correction)

The parity-sector gap in `SPECTRE_QEC_CODES.md` is `sqrt(15)/5 = 0.7746`, not 0.7483; the formula was right and the decimal wrong. The minority orientation class is the Mystic tiles, with frequency `(5 - sqrt(15))/10 = 0.1127`; the ratio of majority to minority frequency is `4 + sqrt(15) = lambda^2`, the same number that fixes the inflation ratio. Tile counts per level, 1, 9, 71, 559, 4401, 34649, agree with `N_k = 8 N_{k-1} - N_{k-2}`; minority counts at levels 1 to 5: 1, 8, 63, 496, 3905. **Exact / Counted.**

## 4. Hypotheses (not tested)

- The fraction of tiles changed per step is a local clock rate, `rho`, giving `Delta tau = Delta t0 / rho` (the paper's equation I). The fraction is constant, 0.056 (a lower bound), so this is a candidate identification only.
- Erasing one tile never leaves the inside ambiguous given the outside (n = 184); so this test gives no Landauer-type cost. **Negative.**
- Whether a closed flip cluster is a composite of Pachner moves is open: no closed cluster was reached inside the validated region (closure needs radius about 26).

## 5. Where this connects to the code here

`spectremodel.py` builds the single tile and `quantumspectremodel.py` the three-tier network; neither implements the window description. The flip engine lives in `wada-coudene/figures/spectre_flip_*.py` and needs `SPECTRE_DIR` pointing at a clone of `brentharts/spectre`. Mapping a flip onto an update of `tri_spectre_qc.py` (re-assigning qubits when the neighbour graph changes) is not written.
