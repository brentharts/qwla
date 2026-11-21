# Erasure-Correcting Quantum Codes and Gauge Sectors on Einstein Spectre Monotiles

## Abstract

Building on our coupled three-tier Spectre quantum simulator (`TriSpectreNetwork` across **9**, **71**, and **559** tiles), we connect the architecture to recent breakthroughs in **aperiodic monotile quantum error-correcting codes (QECCs)**. Following Li and Boyle’s quasicrystal code construction [1] and its extension to the Hat and Spectre monotiles by **Batle and Bednorz (2026)** [2], we examine how strong local indistinguishability, unique hierarchical deflation, orientation-parity superselection sectors under the hexagonal subgroup $G_6 = \mathbb{R}^2 \times C_6$, and gauge mergers under $SE(2)$ provide a rigorous coding-theoretic foundation for our multi-scale Spectre quantum computer.

## 1. Why Aperiodic Monotiles Naturally Form Quantum Error-Correcting Codes

In conventional stabilizer or surface codes, logical qubits are protected against local noise by engineering topological degeneracy on a periodic lattice. In the **Li–Boyle framework** [1, 2], nature supplies erasure protection directly from the geometry of aperiodic tilings through two structural properties:

1. **Strong Local Indistinguishability (H1):** Every finite patch $P$ occurs with a well-defined, uniform frequency across all tilings within a local-indistinguishability (LI) class. Consequently, for group-averaged orbit states $\vert{}\Psi_{[T]}\rangle = \int_G dg\,\vert{}gT\rangle$, the reduced density matrix $\rho_K = \operatorname{Tr}_{K^c}\vert{}\Psi_{[T]}\rangle\langle\Psi_{[T]}\vert{}$ on any bounded region $K \subset \mathbb{R}^2$ is identical across codewords—satisfying the diagonal Knill–Laflamme erasure condition.

2. **Local Recoverability (H2):** Because every Spectre tiling admits an infinite, unique decomposition into higher-order supertiles (the Spectre unique-hierarchy theorem [3]), removing a bounded region $K$ leaves its complement $K^c$ uniquely determining the deleted tiles (unconditionally for nonsingular tilings via the 4:2 cut-and-project torus parametrization) [2]. Distinct orbits cannot agree outside $K$, ensuring off-diagonal Knill–Laflamme matrix elements vanish.

For our **Tri-Spectre Quantum Network**, this means the **9-**, **71-**, and **559-tile** substitution patches (`QC-9`, `QC-71`, and `QC-559`) are not arbitrary graphs: they represent finite deflation/inflation levels of an intrinsic erasure-correcting code where bounded vertex or tile erasures close up at a rate governed by the squared substitution $\sigma^2$ [2]:

$$
r_n \approx a\,(4 + \sqrt{15})^n
$$

where $4 + \sqrt{15} \approx 7.873$ is the Perron–Frobenius eigenvalue of the Spectre cluster substitution matrix:

$$
M_{\text{Spec}} = \begin{pmatrix} 1 & 1 \\ 6 & 7 \end{pmatrix}
$$

## 2. Exact Perron–Frobenius Statistics and Orientation Parity ($LI_1$ vs. $LI_2$)

Batle and Bednorz show that—unlike Penrose, Ammann–Beenker, or Fibonacci tilings, which each have a single local-indistinguishability class—both the Hat and the Spectre split into **two distinct translational LI classes** [2, 4]:

* **Twelve Rotational Orientations:** In a planar Spectre tiling, all tiles appear at integer multiples of $30^\circ$, partitioning into **6 even orientations** ($0^\circ, 60^\circ, \dots, 300^\circ$) and **6 odd orientations** ($30^\circ, 90^\circ, \dots, 330^\circ$).

* **Handedness Reversal Under Single Substitution $\sigma$:** Each Mystic cluster consists of one even and one odd Spectre, while single Spectres carry the reference parity. Because a single substitution step $\sigma$ reverses orientation parity, the chirality-preserving substitution is $\sigma^2$, with matrix:

$$
M_{\text{Spec}}^2 = \begin{pmatrix} 7 & 8 \\ 48 & 55 \end{pmatrix}, \qquad \lambda_+(\sigma^2) = (4 + \sqrt{15})^2 = 31 + 8\sqrt{15}
$$

* **Exact Asymptotic Orientation Frequencies:** Within class $LI_1$, even orientations form the statistical majority and odd orientations the minority [2]:

$$
f_{\text{even}} = \frac{5 + \sqrt{15}}{10} \approx 0.8873, \qquad f_{\text{odd}} = \frac{5 - \sqrt{15}}{10} \approx 0.1127, \qquad \frac{f_{\text{even}}}{f_{\text{odd}}} = 4 + \sqrt{15}
$$

In $LI_2$—obtained by rotating any tiling in $LI_1$ by $R_{30}$ ($30^\circ$)—these frequencies swap.

## 3. Gauge Groups and Superselection: Why Symmetry Choice Matters

A central discovery of Batle and Bednorz (2026) is how the **choice of gauged isometry group** $G$ governs whether the two LI classes act as a **hybrid quantum–classical memory** (carrying a superselected classical bit alongside erasure-protected quantum sectors) or merge into a single quantum code sector [2]:

| Gauge Group $G$ | Hat Monotile Code | Spectre Monotile Code (`Tile(1,1)`) | Physical & Architectural Meaning | 
| ----- | ----- | ----- | ----- | 
| $G_6 = \mathbb{R}^2 \times C_6$ | 2 sectors + 1 classical bit | **2 sectors + 1 classical bit** | Translations and $60^\circ$ rotations gauged; orientation parity ($LI_1$ vs. $LI_2$) is $G_6$-invariant and acts as a **superselected classical bit**. | 
| $SE(2)$ | 2 sectors + 1 classical bit | **1 merged quantum sector** | Continuous rotations gauged; $R_{30} \in SE(2)$ maps $LI_1 \leftrightarrow LI_2$, merging the Spectre sectors into a single erasure-correcting space. | 
| $E(2)$ | 1 merged quantum sector | 1 sector (maps to mirrored tile) | Full Euclidean group including reflections; Hat chirality bit is gauged away, while Spectre reflections leave the chiral prototile family. | 

### Implications for `TriSpectreNetwork`

1. **Operating Under $G_6$ (Discrete $60^\circ$ Substrate Symmetry):**
   In `spectremodel.py` and `quantumspectremodel.py`, edge vectors are constructed from $60^\circ$ trigonometric increments ($\cos 60^\circ = \tfrac{1}{2}$, $\sin 60^\circ = \tfrac{\sqrt{3}}{2}$) and child tiles fan out across $60^\circ$ sectors. Relative to $G_6 = \mathbb{R}^2 \times C_6$, the Spectre code decomposes into a direct sum of two erasure-correcting sectors:

   $$
   \mathcal{C}_{\text{Spec}} = \mathcal{C}_{\text{Spec}}^{(1)} \oplus \mathcal{C}_{\text{Spec}}^{(2)}
   $$

   Here, the local even-orientation counter $N_K^{\text{even}} = \sum_{t \subset K} \mathbf{1}[t \text{ is even}]$ separates the two sectors with macroscopic gap [2]:

   $$
   \Delta = \langle N_K^{\text{even}} \rangle_{(1)} - \langle N_K^{\text{even}} \rangle_{(2)} = \lvert K \rvert_{\text{tiles}} \frac{\sqrt{15}}{5} \approx 0.7746\,\lvert K \rvert_{\text{tiles}}
   $$

2. **Operating Under Continuous $SE(2)$ (Frame-Independent Continuum Limit):**
   When continuous $30^\circ$ rotations $R_{30}$ are treated as gauge redundancies, the even/odd orientation label is frame-dependent and merges ($\mathcal{C}_{\text{Spec}}^{(1)} = \mathcal{C}_{\text{Spec}}^{(2)}$) [2]. In our coupled three-tier computer (`QC-9`, `QC-71`, `QC-559`), enforcing $R_{30}$-covariance across scales corresponds to merging parity sectors so that cross-tier entanglement operates over a unified, rotationally gauged quantum code space.

## 4. Mapping Batle & Bednorz (2026) onto Our Codebase

The theoretical constructs in [2] map directly onto concrete extensions we can build into `spectremodel.py`, `quantumspectremodel.py`, and `tri_spectre_qc.py`:

* **Parity Bookkeeping Across Generations ($k = 1, 2, 3$):**
  Because a single substitution $\sigma$ flips orientation parity while $\sigma^2$ preserves it [2], our three tiers—$k=1$ (**9 tiles**), $k=2$ (**71 tiles**), and $k=3$ (**559 tiles**)—alternate parity between $k=1$ and $k=2$, while $k=1 \to k=3$ represents a full $\sigma^2$ chirality-preserving inflation step. Tracking each tile's $30^\circ$ orientation index $m \in \{0, \dots, 11\}$ in `tile.data["orientation_step"]` allows `TriSpectreNetwork` to compute the exact finite-patch parity observable $N_K^{\text{even}}$ at each scale.

* **Simulated Erasure & Deflation Recovery:**
  Using `Step 1`'s `State` simulator and `Step 2`'s `SubstitutionPatch`, we can simulate local erasure of a bounded cluster of vertices/tiles in `QC-559` (e.g., tracing out or depolarizing a local patch $K$) and verify that cross-tier entanglement with `QC-71` and `QC-9` preserves logical correlations across scales.

* **Rank-4 First Cohomology ($H^1(\Omega, \mathbb{C}) = \mathbb{C}^4$):**
  As highlighted in Section 5 of Batle and Bednorz [2, 4], the Spectre hull carries rank-4 first cohomology $H^1(\Omega, \mathbb{C}) = \mathbb{C}^4$, analogous to the $H^1(T^2) = \mathbb{Z}^2$ topological degeneracy of the toric code. In finite toroidal approximants of our Spectre network, these four cohomological generators provide a natural target for encoding **4 logical flux/topological qubits** across the multi-scale monotile mesh.

## 5. Summary Comparison Across the Theoretical Stack

| Conceptual Layer | Reference | Contribution to the Spectre Quantum Computer | 
| ----- | ----- | ----- | 
| **Chiral Monotile & Substitution** | Smith et al. (2023) [3] | Defines the 14-vertex `Tile(1,1)` Spectre, Mystic clusters, and the $N_k = 8N_{k-1} - N_{k-2}$ ($9, 71, 559$) hierarchy. | 
| **Quantum Universe & QCA** | Lloyd (2013) [5] | Frames local vertex-to-vertex interactions as an aperiodic Quantum Cellular Automaton where quantum fluctuations and metric geometry co-evolve. | 
| **Monotile QEC & Gauge Sectors** | Batle & Bednorz (2026) [2] | Proves erasure-correction conditions (H1/H2) for the Spectre, establishes the $(5 \pm \sqrt{15})/10$ orientation-class frequencies, and distinguishes $G_6$ hybrid memory from $SE(2)$ sector merger. | 

## References

[1] **Zhi Li and Latham Boyle**, *"The Penrose tiling is a quantum error-correcting code"*, `arXiv:2311.13040 [quant-ph]`, 2023. Available at: <https://arxiv.org/abs/2311.13040>

[2] **Josep Batle and Adam Bednorz**, *"Quantum error-correcting codes from aperiodic monotiles: the Hat and the Spectre"*, `arXiv:2607.15326 [quant-ph]`, October 30, 2026. Available at: <https://arxiv.org/abs/2607.15326>

[3] **David Smith, Joseph Samuel Myers, Craig S. Kaplan, and Chaim Goodman-Strauss**, *"A chiral aperiodic monotile"*, *Combinatorial Theory* 4(2) (2024), #13; `arXiv:2305.17743 [math.CO]`. Available at: <https://arxiv.org/abs/2305.17743>

[4] **Michael Baake, Franz Gähler, Jan Mazáč, and Lorenzo Sadun**, *"On the long-range order of the Spectre tilings"*, *Discrete & Computational Geometry* (2025); `arXiv:2411.15503 [math.DS]`. Available at: <https://arxiv.org/abs/2411.15503>

[5] **Seth Lloyd**, *"The Universe as Quantum Computer"*, `arXiv:1312.4455 [quant-ph]`, December 16, 2013. Available at: <https://arxiv.org/abs/1312.4455>
