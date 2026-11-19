# Tri-Spectre Quantum Network

**A Coupled Multi-Scale Quantum Simulator Built on Chiral Aperiodic Einstein Monotiles**

This project connects a matrix-free, functional quantum state simulator (`State`) with an editable geometric object model of the curved **Einstein Spectre monotile** (`spectremodel.py`) to create a network of three entangled quantum computers at substitution scales of **9**, **71**, and **559** tiles.

---

## Why This Project Is Unique & Interesting

1. **Aperiodic Quasi-Crystal Topology Instead of Rigid Square Grids**
   Most quantum computing architectures assume periodic 2D lattices (such as square heavy-hex or surface-code grids). Here, qubits live on the vertices of **Einstein Spectre monotiles**—14-sided chiral tiles that tile the plane aperiodically without translational symmetry. This creates non-repeating local neighborhoods and hierarchical scale self-similarity across generations.

2. **Substitution Recurrence as Multi-Scale Quantum Hardware ($9 \to 71 \to 559$)**
   Under the Spectre substitution system, single Spectres and 2-tile clusters (*Mystics*) inflate according to:
   * $\text{Mystic} \mapsto 1\,\text{Mystic} + 6\,\text{Singles}$ ($8$ tiles)
   * $\text{Single} \mapsto 1\,\text{Mystic} + 7\,\text{Singles}$ ($9$ tiles)
   
   Both sequences obey the exact linear recurrence:
   $$N_k = 8N_{k-1} - N_{k-2}$$
   Starting from a single Spectre ($N_0 = 1$), generations $k = 1, 2, 3$ produce **9**, **71**, and **559** tiles. Instead of treating scale as an abstract parameter, the system instantiates three distinct quantum computers (`QC-9`, `QC-71`, and `QC-559`) corresponding to these exact substitution tiers and entangles them across scales.

3. **Bidirectional Geometry $\leftrightarrow$ Quantum Feedback Loop**
   In `spectremodel.py`, every tile is a live cubic Bezier object with 14 vertices and 28 control handles whose `twist` angles and `curve_strength` dynamically alter the tile's outline and exact enclosed area. 
   * **Geometry $\to$ Quantum:** Vertex twists and curvature drive local parameterized $R_Z(\theta)$ phase rotations and entangling gates on the mapped qubits.
   * **Quantum $\to$ Geometry:** Local Pauli-$Z$ expectation values $\langle Z_q \rangle$ feed back into vertex twists and `curve_strength`, physically deforming the Bezier boundaries of the monotiles and shifting their exact areas (`tile.area()`) in real time.

4. **Purely Functional, Matrix-Free Quantum Simulation**
   Rather than constructing $2^n \times 2^n$ unitary matrices, the quantum engine transforms streams of `(frozenbitarray, amplitude)` pairs using `pyfunctional` (`seq`) pipelines (`smap`, `filter`, `flatten`, `reduce_by_key`). This keeps gate logic transparent, sparse-friendly, and closely aligned with bitwise operations on vertex registers.

---

## Architecture Overview

```text
+-------------------------------------------------------------------------+
|                        TriSpectreNetwork (Step 3)                       |
|                                                                         |
|   +---------------+       +----------------+       +----------------+   |
|   |     QC-9      | <===> |     QC-71      | <===> |     QC-559     |   |
|   |  (k=1, 9 tiles|  CX/  | (k=2, 71 tiles |  CX/  | (k=3, 559 tiles|   |
|   |   & State)    |  CZ   |   & State)     |  CZ   |   & State)     |   |
|   +-------+-------+       +-------+--------+       +-------+--------+   |
|           |                       |                        |            |
|           +-----------+-----------+------------------------+            |
|                       |  Bidirectional Coupling:                        |
|                       |  * Vertex/Handle Twists -> RZ(theta) Gates      |
|                       |  * <Z_q> Expectations -> Twist & Curve Updates  |
|                       v                                                 |
|   +-----------------------------------------------------------------+   |
|   |             Substitution Hierarchy & Node Mapper (Step 2)       |   |
|   |  * Expands Single/Mystic rules (N_k = 8 N_{k-1} - N_{k-2})      |   |
|   |  * Clusters coincident boundary vertices into topological nodes |   |
|   |  * Writes qubit indices into `vertex.data["qubit"]`             |   |
|   +-----------------------------------------------------------------+   |
|                       |                                                 |
|                       v                                                 |
|   +-----------------------------------------------------------------+   |
|   |          Editable Spectre Monotile Geometry (spectremodel.py)   |   |
|   |  * 14 vertices, 28 Bezier handles, live twists & exact area()   |   |
|   +-----------------------------------------------------------------+   |
+-------------------------------------------------------------------------+
```

---

## Module Breakdown

| File / Module | Role | Key Components |
| :--- | :--- | :--- |
| **`spectremodel.py`** | Editable geometric model of the curved 14-vertex Spectre monotile. | `Tile`, `Vertex`, `BezHandle`, `outline()`, `area()`, live `twist` and `curve_strength`. |
| **`step1_state.py`** | Matrix-free functional quantum state simulator. | `State` (`x`, `z`, `s`, `t`, `phase`, `rz`, `h`, `cx`, `cz`, `ccx`, `expectation_z`, safe `measure`). |
| **`spectre_network.py`** | Substitution patch generator and topological vertex-to-qubit mapper. | `SubstitutionPatch`, `expand_single_tile()`, `build_substitution_tier()`, `cluster_shared_vertices()`, `assign_qubits_to_patch()`. |
| **`tri_spectre_qc.py`** | Coupled three-computer quantum network and geometry feedback engine. | `SpectreQuantumComputer`, `TriSpectreNetwork`, `_tensor_product_states()`, `entangle_across_computers()`, `step_coupled_cycle()`. |

---

## Substitution Table ($N_k = 8N_{k-1} - N_{k-2}$)

| Iteration ($k$) | Starting from 1 Single Spectre | Starting from 1 Mystic (2 tiles) | Role in `TriSpectreNetwork` |
| :---: | :---: | :---: | :--- |
| $k = 0$ | $1$ | $2$ | Seed monotile |
| $k = 1$ | **$9$** | $8$ | **`QC-9`** (Tier 1 quantum computer) |
| $k = 2$ | **$71$** | $62$ | **`QC-71`** (Tier 2 quantum computer) |
| $k = 3$ | **$559$** | $488$ | **`QC-559`** (Tier 3 quantum computer) |
| $k = 4$ | $4{,}401$ | $3{,}842$ | — |
| $k = 5$ | $34{,}649$ | $30{,}248$ | — |
| $k = 6$ | $272{,}791$ | $238{,}142$ | — |

---

## Installation & Requirements

Install the Python dependencies required by the simulator and geometry engine:

```bash
pip install numpy bitarray pyfunctional matplotlib
```

* `numpy` & `matplotlib` (optional, for CLI plotting): Used by `spectremodel.py` for Bezier evaluation, Gauss-Legendre exact area quadrature, and rendering.
* `bitarray` & `pyfunctional`: Used by `step1_state.py` for immutable `frozenbitarray` basis keys and functional `seq` state transformations.

---

## Quick Start

### 1. Run Self-Tests on the Spectre Monotile Geometry
Verify that the editable Bezier tile model passes all geometric, twist, and exact-area checks:

```bash
python3 spectremodel.py --tests-only
```

### 2. Inspect the Three Substitution Tiers (9, 71, 559 Tiles)
Generate the $k = 1, 2, 3$ patches, cluster shared boundary vertices into topological nodes, and bind qubit indices to `vertex.data["qubit"]`:

```bash
python3 quantumspectremodel.py
```

### 3. Run a Coupled Three-Computer Quantum-Geometry Cycle
Instantiate `QC-9`, `QC-71`, and `QC-559`, apply geometry-driven local circuits, entangle anchor qubits across all three computers, and feed $\langle Z_q \rangle$ expectations back into tile twists and areas:

```bash
python3 tri_spectre_qc.py
```

### Example Python Usage

```python
from tri_spectre_qc import TriSpectreNetwork

# Initialize the three coupled quantum computers (9, 71, and 559 tiles)
network = TriSpectreNetwork(
    a=1.0,
    b=1.0,
    curve_strength=0.5,
    qubits_per_tier=(4, 4, 4),
)

# Run 3 coupled quantum-geometry feedback cycles
for step in range(3):
    network.step_coupled_cycle(twist_gain=0.25)
    print(f"=== Cycle {step + 1} ===")
    for row in network.summary():
        print(row)
```

---

## Future Directions & Extensions

* **Exact Hat/Spectre Substitution Gluing:** Upgrade `expand_single_tile()` from sector-offset placement to exact edge-matching substitution rules so shared boundaries match to machine precision across all 559 tiles.
* **Handle-Level Couplings:** Map `BezHandle.twist` (`out_handle` and `in_handle`) to asymmetric two-qubit interaction strengths ($J_{ij}$) along shared monotile edges.
* **Interactive Visualization:** Render `QC-9`, `QC-71`, and `QC-559` side-by-side with `matplotlib`, coloring vertices by $\langle Z_q \rangle$ and animating the Bezier boundary deformations across cycles.

# QWLA: Quantum computing without the linear algebra

<div align="center">
  <img src="https://github.com/user-attachments/assets/5b65a96e-4fcc-4af9-b580-a09c509a38cc" width="300" />
</div>
QWLA is a tiny quantum circuit simulator written in Python.
The goal of QWLA, as the title suggests, is to present quantum computing without using linear algebra.
Instead, QWLA represents a quantum state as a dictionary and applies operations in a functional style, using map, filter, reduce, etc.

## technical details
For all technical details and an introduction to quantum computing without the linear algebra, see the [accompanying paper](https://eprint.iacr.org/2025/1091.pdf).

## structure
- The `state.py` file contains the `State` class, which is a quantum state and supports a number of operations (Clifford + T).
- There are example circuits in the `examples/` folder.


## usage
The easiest way to use QWLA is with [uv](https://docs.astral.sh/uv/getting-started/installation/).
To run an example, simply:

```bash
uv run examples/epr.py
```

## citation

```
@misc{cryptoeprint:2025/1091,
      author = {Aws Albarghouthi},
      title = {Quantum Computing without the Linear Algebra},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/1091},
      year = {2025},
      url = {https://eprint.iacr.org/2025/1091}
}
```


