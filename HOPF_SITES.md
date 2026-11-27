# The 12 cone points as Bloch points: the Moebius dictionary on the Hat Trick ball

Stage 4i of [`ROADMAP_QUANTUM_GRAVITY.md`](ROADMAP_QUANTUM_GRAVITY.md) records a dictionary from the `ray` fork: the Hopf fibres over two Bloch points `a`, `b` have Moebius invariants `s1 = s2 = |cos(theta_ab / 2)| = |<a|b>|`, and the pair of fibres is a "Hopf clasp" exactly when the two states are orthogonal. It was "not implemented as code". This note evaluates it where the repository already has 12 spin-1/2 points, the cone points of the Hat Trick ball (`ring_bits.py`, `HAT_TIME.md`), with `hopf_sites.py`.

Labels as in the roadmap: **Exact** (identity or finite exact computation), **Counted**, **Known** (standard result used, not derived), **Definition**, **Not done**.

## What is a definition here

A cone point is a point of a sphere, not a spin. To read it as a Bloch point I take the coherent spin-1/2 state in the direction of its position vector. That is a choice; the ball has no spin, and the model of `ring_bits.py` (one ring bit per cone point, Heisenberg exchange on the 30 edges) does not use directions at all. Everything below about overlaps and clasps depends on this reading.

## 1. The three pair classes

**Exact** (`python3 hopf_sites.py --tests`; the vertex set is the regular icosahedron, and `--hat` checks that the ball's pentagon centres have the same pair angles, 30 + 30 + 6 pairs). The 66 pairs fall into three classes:

| class | pairs | angle theta | overlap `s = abs(cos(theta/2))` | `cos(theta)/4` |
|---|---|---|---|---|
| edge (neighbours) | 30 | 63.435 degrees | 0.85065 = sqrt((1 + 1/sqrt 5)/2) | +0.1118 |
| second neighbours | 30 | 116.565 degrees | 0.52573 = sqrt((1 - 1/sqrt 5)/2) | -0.1118 |
| antipodal | 6 | 180 degrees | 0 | -0.25 |

**Exact.** The orthogonal pairs (`s = 0`) are exactly the 6 antipodal pairs, and they partition the 12 points. So in this reading the ball has 6 Hopf clasps, and they pair up all the cone points.

**Known** (and not recomputed): all fibres of one Hopf fibration link pairwise once, so the linking numbers of the 12 fibres are the same for every pair and carry no information. The invariants `s` do, which is the point of the dictionary.

## 2. The exact ground state, by class

The Heisenberg model on the 30 edges, `H = sum_edges S_i.S_j`, on the 4096 states of 12 spin-1/2 (`ring_bits.py` finds its lowest level in the singlet sector as -6.1879).

**Exact to rounding** (sparse diagonalisation, `hopf_sites.py`):

- The ground energy is -6.187890, the next two levels are degenerate at -5.654449, so the gap is 0.5334 and the ground state is unique. It is a singlet: total `S(S+1)` is 0 to 1e-15.
- The correlation `<S_i.S_j>` depends only on the pair class (the spread inside each class is below 1e-14):

| class | `<S_i.S_j>` in the ground state | product of coherent states, `cos(theta)/4` | antiferromagnetic product state, `-cos(theta)/4` |
|---|---|---|---|
| edge | -0.20626 | +0.1118 | -0.1118 |
| second neighbours | +0.08408 | -0.1118 | +0.1118 |
| antipodal | -0.13908 | -0.25 | +0.25 |

- The singlet sum rule holds: `30(-0.20626) + 30(0.08408) + 6(-0.13908) = -4.5`, as `sum over all pairs S_i.S_j = (S(S+1) - 9)/2` requires.

So the quantum ground state has the signs of an antiferromagnet on neighbours and second neighbours (negative, then positive), but on the antipodal pairs, the orthogonal ones, it has -0.139 where a classical antiferromagnet would give +0.25. For two spin-1/2 the correlation runs from -0.75 (singlet) to +0.25 (triplet), so the 6 clasp pairs sit 0.39 of the way from triplet to singlet. This is frustration: the icosahedron cannot be two-coloured.

## 3. What this does and does not show

- It gives the Stage 4i dictionary one concrete, finite instance: 12 Bloch points, 3 overlaps, 6 clasps. Nothing in the dictionary was changed.
- It does not show that the clasp structure matters to the ring-bit model. The ground-state correlations above are a property of the Heisenberg graph, and the antipodal class is distinguished only by graph distance 3 on the icosahedron.
- **Not done:** whether the six clasp pairs organise the level-`k` models of `qdeform_punctures.py` (at level 2 the five qubits sit on pairs of neighbours, not antipodes); whether the overlaps `s` relate to any matrix element of `chi4`.

## Reproduce

    python3 hopf_sites.py --tests          # pair classes, orthogonal pairs, ground state, sum rule (about 5 s)
    python3 hopf_sites.py --tests --hat    # also check against the ball's pentagon centres (needs hexasphere)
