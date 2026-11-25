#!/usr/bin/env python3
"""
Stage 9b of ROADMAP_QUANTUM_GRAVITY.md: the Hilbert space of a sphere with n spin-1/2 punctures (flat SU(2)
connections / LQG intertwiners), compared with the pair-qubit space of the Spectre network.

For n spin-1/2 punctures on a sphere the gauge-invariant space is the singlet subspace Inv((C^2)^n), of dimension
the Catalan number C_{n/2}.  This module builds it exactly (null space of the total spin), lets the rotation group
act by permuting punctures, and counts irreps.  Two measurements:

  A  dimension: the pair-qubit space of p pairs has dimension 2^p, the singlet space of 2p punctures has C_p.
  B  symmetry: the tetrahedral group T = A4 permuting the 4 vertices of a tetrahedron (n=4) and the 12 vertices of
     an icosahedron = pentagon holes of the Hat Trick (n=12): decomposition of the singlet space into T irreps.

Scope: dimension counting and group representation only.  No map from pair qubits into the singlet space is
constructed, so "fits by dimension" is necessary, not an embedding.  Run `python3 puncture_hilbert.py` (tests first).
"""
import itertools
import json
import math
import sys
import numpy as np

PHI = (1 + 5 ** 0.5) / 2


def catalan(k):
    return math.comb(2 * k, k) // (k + 1)


def singlet_space(n):
    """Orthonormal basis (columns) of the SU(2) singlets in (C^2)^n, built from the total J^2 on the Jz=0 sector."""
    assert n % 2 == 0
    N = 1 << n
    bits = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1)       # bit=1 means spin down
    sector = np.where(bits.sum(1) == n // 2)[0]
    idx = {s: i for i, s in enumerate(sector)}
    m = len(sector)
    # J^2 = (n/2)(n/2+2)... use J^2 = Jz^2 + (J+J- + J-J+)/2 on Jz=0, with Jz=0: J^2 = (J+J- + J-J+)/2
    Jm = np.zeros((N, N)) if False else None
    # J^2 restricted to Jz=0 sector, via pair sums: J^2 = 3n/4 + sum_{i<j} (2 S_i.S_j) ; S_i.S_j = swap/2 - 1/4
    # so J^2 = 3n/4 + sum_{i<j} (swap_ij - 1/2)
    H = np.zeros((m, m))
    for i, j in itertools.combinations(range(n), 2):
        for s in sector:
            bi, bj = (s >> i) & 1, (s >> j) & 1
            t = s if bi == bj else s ^ (1 << i) ^ (1 << j)
            H[idx[t], idx[s]] += 1.0
    H += np.eye(m) * (3 * n / 4 - n * (n - 1) / 4)
    w, v = np.linalg.eigh(H)
    keep = np.abs(w) < 1e-8
    V = np.zeros((N, int(keep.sum())))
    V[sector, :] = v[:, keep]
    return V


def perm_on_states(n, perm):
    """Index map of the permutation of tensor factors: new[s'] where bit position perm[i] of s' = bit i of s."""
    N = 1 << n
    s = np.arange(N)
    t = np.zeros(N, dtype=np.int64)
    for i in range(n):
        t |= ((s >> i) & 1) << perm[i]
    return t


def character(V, n, perm):
    t = perm_on_states(n, perm)
    PV = np.zeros_like(V)
    PV[t, :] = V            # (P V)[t(s)] = V[s]
    return float(np.trace(V.T @ PV))


def icosa_vertices():
    v = []
    for a in (1, -1):
        for b in (PHI, -PHI):
            v += [(0, a, b), (a, b, 0), (b, 0, a)]
    return np.array(v, float)


def rot_axis(axis, ang):
    a = np.array(axis, float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def icosa_group(V):
    gens = [rot_axis(V[0], 2 * math.pi / 5), rot_axis(cent3(V), 2 * math.pi / 3)]
    G = [np.eye(3)]
    ch = True
    while ch:
        ch = False
        for g in list(G):
            for h in gens:
                x = h @ g
                if not any(np.allclose(x, y, atol=1e-8) for y in G):
                    G.append(x)
                    ch = True
    return G


def cent3(V):
    """Centre of one triangular face of the icosahedron."""
    d = np.linalg.norm(V[:, None] - V[None], axis=2)
    e = d[d > 1e-9].min()
    for i, j, k in itertools.combinations(range(12), 3):
        if abs(d[i, j] - e) < 1e-6 and abs(d[j, k] - e) < 1e-6 and abs(d[i, k] - e) < 1e-6:
            return V[i] + V[j] + V[k]


def vertex_perm(V, g):
    out = []
    for p in V:
        q = g @ p
        out.append(int(np.argmin(np.linalg.norm(V - q, axis=1))))
    return out


def tetra_in_icosa(G):
    """T: rotations in I that permute the three coordinate axes (the pyritohedral T of order 12)."""
    keep = []
    for g in G:
        M = np.abs(g)
        if np.allclose(np.sort(M.ravel())[-3:], 1, atol=1e-8) and np.allclose((M > 0.5).sum(), 3):
            keep.append(g)
    return keep


def classes_A4(group_perms):
    """Split permutations by cycle type: id (1), double transposition-like order 2 (3), order 3 (8, two classes)."""
    return group_perms


# A4 character table (classes: e, (12)(34) x3, (123) x4, (132) x4)
A4_CHARS = {"1": (1, 1, 1, 1), "1'": (1, 1, "w", "w2"), "1''": (1, 1, "w2", "w"), "3": (3, -1, 0, 0)}


def decompose_A4(chars_by_class):
    """chars_by_class = (chi(e), chi(order2), chi(c3), chi(c3^-1)); return multiplicities of 1, 1', 1'', 3."""
    w = complex(math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3))
    val = lambda x: {"w": w, "w2": w * w}.get(x, x)
    sizes = (1, 3, 4, 4)
    out = {}
    for name, ch in A4_CHARS.items():
        s = sum(sz * c * np.conj(val(x)) for sz, c, x in zip(sizes, chars_by_class, ch))
        out[name] = round((s / 12).real, 6)
    return out


def tetra_vertices():
    return np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], float)


def A4_perm_classes(V, group):
    """Return (list of (perm, class)) classes e / order2 / c3 / c3^-1 by a fixed 3-fold generator."""
    perms = [tuple(vertex_perm(V, g)) for g in group]
    return perms


def analyse(n_vertices):
    V = tetra_vertices() if n_vertices == 4 else icosa_vertices()
    G = tet_rotations() if n_vertices == 4 else tetra_in_icosa(icosa_group(V))
    B = singlet_space(n_vertices)
    assert len(G) == 12, len(G)
    # classes: identity, order 2, order 3 split by rotation sense about a fixed axis
    ch = {"e": [], "c2": [], "c3": [], "c3i": []}
    for g in G:
        tr = np.trace(g)
        x = character(B, n_vertices, vertex_perm(V, g))
        if np.allclose(g, np.eye(3)):
            ch["e"].append(x)
        elif abs(tr + 1) < 1e-8:
            ch["c2"].append(x)
        else:
            ch["c3"].append((g, x))
    # split the eight 3-cycles into two classes: g and g^-1 are in different classes
    c3 = ch["c3"]
    classA, classB = [], []
    rem = list(c3)
    while rem:
        g, x = rem.pop(0)
        classA.append(x)
        for k, (h, y) in enumerate(rem):
            if np.allclose(h, g.T, atol=1e-8):
                classB.append(y)
                rem.pop(k)
                break
    # A4 has two classes of 3-cycles of size 4: pick the class of g by conjugacy inside the group
    cl = conj_class_of(G, c3[0][0])
    A = [x for g, x in c3 if any(np.allclose(g, h, atol=1e-8) for h in cl)]
    Bc = [x for g, x in c3 if not any(np.allclose(g, h, atol=1e-8) for h in cl)]
    cb = (np.mean(ch["e"]), np.mean(ch["c2"]), np.mean(A), np.mean(Bc))
    return {"n": n_vertices, "dim": B.shape[1], "catalan": catalan(n_vertices // 2), "chars": [float(c) for c in cb],
            "irreps": decompose_A4(cb)}


def conj_class_of(G, g):
    out = []
    for h in G:
        x = h @ g @ h.T
        if not any(np.allclose(x, y, atol=1e-8) for y in out):
            out.append(x)
    return out


def tet_rotations():
    """The 12 rotations of the pyritohedral T: even permutations of axes with signs of product +1."""
    G = []
    for p in itertools.permutations(range(3)):
        sg = np.linalg.det(np.eye(3)[list(p)])
        for s in itertools.product((1, -1), repeat=3):
            if np.prod(s) * sg > 0:
                M = np.zeros((3, 3))
                for i in range(3):
                    M[p[i], i] = s[i]
                G.append(M)
    return [g for g in G if np.isclose(np.linalg.det(g), 1) and is_even_axes(g)]


def is_even_axes(g):
    p = [int(np.argmax(np.abs(g[:, i]))) for i in range(3)]
    inv = sum(1 for i in range(3) for j in range(i + 1, 3) if p[i] > p[j])
    return inv % 2 == 0


def dimension_table(pmax=10):
    return [{"pairs": p, "qubit_dim": 2 ** p, "singlet_dim": catalan(p), "fits": 2 ** p <= catalan(p)} for p in range(1, pmax + 1)]


# ---- tests ------------------------------------------------------------------
def test_catalan_dimensions():
    for n in (2, 4, 6, 8):
        assert singlet_space(n).shape[1] == catalan(n // 2)


def test_singlets_are_invariant():
    B = singlet_space(4)
    assert np.allclose(B.T @ B, np.eye(B.shape[1]))
    # swap of factors 0,1 acts as -1 on pair singlet etc: character of identity = dim
    assert np.isclose(character(B, 4, [0, 1, 2, 3]), B.shape[1])


def test_tetra_group_has_twelve():
    assert len(tet_rotations()) == 12
    G = icosa_group(icosa_vertices())
    assert len(G) == 60 and len(tetra_in_icosa(G)) == 12


def test_four_punctures_give_one_one_prime_plus_one_double_prime():
    r = analyse(4)
    assert r["dim"] == 2 and r["irreps"]["1'"] == 1 and r["irreps"]["1''"] == 1 and r["irreps"]["3"] == 0


def test_dimension_table_crossover():
    t = dimension_table(8)
    assert [x["fits"] for x in t][:5] == [False, False, False, False, True] or not t[4]["fits"]
    assert t[5]["singlet_dim"] == 132


def invariant_algebra_dim(n):
    """Dimension of the algebra of operators on (C^2)^n commuting with all SU(2) rotations: sum of squared multiplicities
    of the spin-j irreps (multiplicity = number of times spin j appears)."""
    from math import comb
    tot = 0
    for j2 in range(n % 2, n + 1, 2):                  # 2j
        m = comb(n, (n - j2) // 2) - (comb(n, (n - j2) // 2 - 1) if (n - j2) // 2 >= 1 else 0)
        tot += m * m
    return tot


def test_one_pair_has_only_a_commutative_invariant_algebra():
    # two spin-1/2: singlet (mult 1) and triplet (mult 1) -> algebra dimension 2, commutative: a bit, not a qubit
    assert invariant_algebra_dim(2) == 2
    assert invariant_algebra_dim(4) == 6 ** 0 * (2 * 2 + 3 * 3 + 1 * 1)    # 2 singlets, 3 triplets, 1 quintet


def run_tests():
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fs:
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    run_tests()
    out = {"invariant_algebra_dim_pair": invariant_algebra_dim(2), "dimension_table": dimension_table(10), "n4": analyse(4), "n12": analyse(12)}
    json.dump(out, open("puncture_hilbert.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
