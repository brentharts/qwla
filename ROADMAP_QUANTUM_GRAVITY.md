# Roadmap: from a Spectre qubit network toward a quantum-gravity theory

This is a plan, not a result. Nothing in this repository or in its companion papers is a theory of quantum gravity, and the stages below are ordered by what can be tested next, not by how close the goal is. Each stage has a deliverable and a pass/fail test, so that a stage can be abandoned honestly.

Companion repositories: `spectre-monotile-py` (exact tilings, flips, pair Hamiltonian; paper `quantgravedit.tex`, long draft `quantgrav.tex`), `wada-coudene` (window and flip engine), `spectre` (Hat and Spectre source, `SPECTRE_DIR`).

Labels used throughout: **Exact** (proved or finite exact computation), **Counted/Measured** (numerical, one setting), **Negative** (a natural expectation refuted), **Hypothesis** (untested).

## 1. What a theory has to supply

A discrete approach to quantum gravity needs, at minimum, the following. The right-hand columns say where the Spectre programme stands.

| # | Requirement | Status in this programme | Label |
|---|---|---|---|
| R1 | A background-independent, local move set | Phason flips are clusters of about ten tiles at uniform density, not single moves; whether clusters are valid tilings, and whether they are Pachner-type moves, is undecided | Measured; open |
| R2 | A Hilbert space and a Hamiltonian | One qubit per bump-and-dent pair, `H_eff = -J sum tau^x - sum K_AB tau^z tau^z`, dipolar frustrated couplings; the unique classical ground state is known exactly to 28 pairs | Exact |
| R3 | Geometry that is dynamical, not fixed | The contact graph changes in 75% of tile-changing flips; the pair count changes while the area ledger is kept. No rule yet turns this into a unitary or a quantum channel | Measured; open |
| R4 | A semiclassical limit (continuum, isotropy, diffeomorphism-like symmetry) | The inflation `lambda^2 = 4 + sqrt(15)` gives a scale ladder and the pair-rule table closes at 53 types from level 4; no continuum limit of the Hamiltonian is defined | Exact for the rule; open for the limit |
| R5 | Sources and a Newtonian limit | None: the tile as it stands has no monopole source, no horizon and no light bending of the right sign | Negative |
| R6 | An arrow or an ordering of time | None from flips (the relation is symmetric); Hamiltonians are real so entropy growth is time-symmetric about the initial state | Negative |
| R7 | Observables that look gravitational: area laws, entanglement geometry, error correction | Erasure is unambiguous (hole of 300 tiles has one refilling), which is the local-recoverability half of the Li-Boyle code condition; no entropy or area-law measurement on the network yet | Exact for refilling; open |
| R8 | Matter and gauge fields | A Z2 gauge form on the patch complex and an exact flux law exist; the lifted tetrad flux is zero | Exact; no dynamics |

The programme is strongest at R2 and R7 (substrate, Hamiltonian, recoverability) and weakest at R4 to R6. The roadmap therefore starts by making the simulator match the exact structure, and only then asks the gravitational questions.

## 2. Stages

### Stage 0: make this repository say what it does (done)
Problems found by running the first version, and what was done (all checked by `--tests-only`):

- *Tile counts.* The tiers had 8, 63 and 496 tiles instead of 9, 71, 559, because a Mystic (two tiles) was counted as one object. The exact builder gives 9, 71, 559, 4401 for singles and 8, 62, 488, 3842 for Mystics.
- *Substitution.* Children were placed by sector offsets, so the patches were not tilings. The patches are now the exact edge-matched tilings of `spectre_exact.py` (integer `Z[zeta_12]`), tested for the tile shape and area, for no overlap, and for edge matching.
- *Qubits.* Qubits now sit on bump-and-dent pairs (6, 82, 804 at generations 2, 3, 4), not on 4 hub vertices. The 9-tile patch has no pair, so the tiers are generations 2, 3, 4. The number of qubits per tier is still a budget (the central pairs), because the joint state is dense.
- *Feedback.* It was inert: a Hadamard on every qubit gives `<Z> = 0` for all later RZ and CZ gates, and `curve_strength` cannot change the area of a point-symmetric Bezier tile. The state now starts in `|0..0>`, evolves under the pair Hamiltonian (Trotter steps tested against exact diagonalisation), and `<Z_p>` moves the flat vertex, changing the two areas by `+-a h <Z_p>` with the pair area conserved.
- *Joint state.* The old code projected the joint state onto invented marginal amplitudes; one joint `State` is now kept and read directly.
- *Docs.* The README module table named files that do not exist; it now matches the files.

Not done: `qubits_per_tier` is small (4 by default), so most pairs carry no qubit; exact patches need `a = b = 1` (no Spectre-to-Hat flow yet); the tiers differ only in how many pairs exist, and the three tiers' central pairs are locally alike, so they give similar `<Z>`.

### Stage 1: an exact substrate (mostly done)
Done: exact edge-matched tilings, qubits on pairs, the area ledger in the geometry, and pair counts 6, 82, 804, 7012 reproduced for generations 2 to 4 (the generation 5 count and the 53-type pair-rule table are not yet recovered inside this repository; they are in `spectre-monotile-py`).
Pass: a test recovering the pair-rule table from the patch, and pair counts to generation 5.

### Stage 2: dynamics of the qubits
Load the dipolar pair Hamiltonian into the QWLA `State` simulator for small clusters (6 to 14 pairs, matrix-free) and compare with the exact diagonalisation in `research/toymodel_v6` and `v10`. Measure the ground-state concurrence, CHSH of reduced pairs, and the gap along the Spectre-to-Hat parameter.
Pass: agreement with the exact values (cat ground state at zero tunnelling; largest pairwise concurrence at an intermediate `tau`). A mismatch is a bug in one of the two programs.

### Stage 3: flips as dynamics
A flip changes which tiles touch, so the Hilbert space (the set of pairs) changes. Define one rule for how a flip acts on the qubits: pairs that persist keep their state, pairs that are created or destroyed are initialised or traced out, and the area ledger and the pair-parity flux are conserved. Then test conservation, reversibility (the flip relation is symmetric, so the rule should be) and whether the resulting channel is unitary on a fixed extended space.
Prerequisite from `spectre-monotile-py`: decide whether flip clusters are tilings (exact refilling of the holes of valid flips).
Pass: a flip channel that conserves the ledger and is its own inverse in the reversed step; fail: any rule that needs the pair count to be conserved (it is not).

### Stage 4: geometry from entanglement
Define a distance from the state, for example from mutual information between pairs, and compare it with the contact-graph distance and the physical distance. Measure the entanglement entropy of a region against its boundary length on levels 3 and 4.
Pass: a stable scaling law (area law, or a measured exponent) and a distance that correlates with the contact graph. A volume law or no correlation closes the idea. This stage is a **Hypothesis**: no measurement exists yet.

### Stage 5: renormalisation and the semiclassical limit
Make the three tiers a coarse-graining ladder rather than three unrelated computers: a map from level `k` to `k-1` that preserves the Hamiltonian's low-energy sector, using the inflation `lambda^2` and the closed 53-type pair rule. Examine the spectral dimension, isotropy of the long-wave limit and how the dipole frustration (0.16 to 0.21 for up to 28 pairs; bulk value near 0.29 is a hypothesis) flows with scale.
Pass: a fixed point or a clean scaling of the couplings with level. Fail: couplings that do not close under the map.

### Stage 6: gravitational signatures, or a decision to stop
The earlier models found no monopole source, no horizon and the wrong sign for light bending. Either add an ingredient that can supply a source (a field on the lift lattice, a four-dimensional complex whose moves are Pachner moves, or a rule that gives the bit an energy), or state that the substrate is an error-correcting code and a toy Hamiltonian rather than a gravity model.
Pass: a source that gives a `1/r` potential or a Gauss-law flux with the right sign, tested on a patch. Otherwise record the negative.

## 3. Decisions and kill criteria

| After stage | If this happens | Then |
|---|---|---|
| 1 | Pairs cannot carry qubits without breaking exact edge matching | Return to vertex qubits and drop the ledger |
| 3 | No flip rule conserves the ledger and is reversible | Treat flips as a classical background only |
| 4 | Entropy has a volume law and no relation to the contact graph | Drop the emergent-geometry claim; keep the code-theoretic reading (`SPECTRE_QEC_CODES.md`) |
| 5 | Couplings do not close under inflation | The scale ladder is kinematic only |
| 6 | No source is found | Say so; publish the substrate and the negatives |

## 4. How the documents here fit together

| Document | Role |
|---|---|
| `README.md` | The simulator and how to run it |
| `SPECTRE_QUANTUM_UNIVERSE.md` | The motivation: aperiodic quantum cellular automaton (Lloyd) |
| `SPECTRE_QEC_CODES.md` | Erasure-correcting codes and gauge sectors (Li-Boyle, Batle-Bednorz) |
| `SPECTRE_PHASON_FLIPS.md` | What a tiling rearrangement does, with the measured numbers |
| this file | The ordered plan and its stopping rules |

## 5. Open items inherited from the papers

- Reproduce or refute the 2-tile swap with reflections at `k = 4` (needs J. Lockwood's erased tile list).
- Closed flip clusters need a placement list of radius about 26 (an hour or more with the current code).
- Prove that the pair-rule table gains no new type at any level (needs the seam automaton).
- Converge the dipole annealing for 82 and 804 pairs.
- A flip rate that depends on the level is the only untested candidate for an arrow of time from flips.

## 6. Reading

Li and Boyle, *The Penrose tiling is a quantum error-correcting code* (arXiv:2311.13040); Batle and Bednorz, *Quantum error-correcting codes from aperiodic monotiles: the Hat and the Spectre* (arXiv:2607.15326); Smith, Myers, Kaplan and Goodman-Strauss, *A chiral aperiodic monotile* (arXiv:2305.17743); Lloyd, *The Universe as Quantum Computer* (arXiv:1312.4455); Comisso and Asenjo on gravitational reconnection in tetrad form, cited in `quantgravedit.tex`.
