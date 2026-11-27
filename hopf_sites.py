#!/usr/bin/env python3
"""
The 12 cone points as Bloch points: the Moebius dictionary of ROADMAP Stage 4i evaluated on the Hat Trick ball.

Stage 4i (the `ray` fork notes) says that the Hopf fibres over two Bloch points a, b have Moebius invariants
s1 = s2 = |cos(theta_ab / 2)| = |<a|b>|, and that the pair is a "Hopf clasp" exactly when the two states are
orthogonal (s = 0). That was not implemented. Here the 12 cone points of the Hat Trick ball are read as 12 Bloch
points (a coherent spin-1/2 state at each), and:

 1  the pair classes of the icosahedron: angles, counts, overlaps s (exact);
 2  the 6 antipodal pairs are exactly the orthogonal pairs, they partition the 12 points;
 3  the exact ground state of the Heisenberg model H = sum over the 30 edges of S_i.S_j on 12 spins (4096 states):
    energy, total spin, and the correlation <S_i.S_j> in each pair class, against the product-state value
    cos(theta_ij)/4 of the coherent states at the same points.

Definitions here: reading a cone point as a Bloch point (the icosahedron's vertex direction). Nothing here selects that
direction; the Hat Trick ball has no spin.  The vertices are the regular icosahedron (0, +-1, +-phi) and cyclic
permutations; with --hat they are checked against the pentagon centres of the ball (needs hexasphere, via ring_bits).
Usage: python3 hopf_sites.py [--tests] [--hat]
"""
import itertools
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

N = 12
PHI = (1 + 5 ** 0.5) / 2


def icosahedron():
    v = []
    for a, b in itertools.product((1, -1), repeat=2):
        v += [(0, a, b * PHI), (a, b * PHI, 0), (b * PHI, 0, a)]
    P = np.array(v, float)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def classes(P):
    """Angle in degrees between every pair, and the class of each pair: 'edge', 'second', 'antipodal'."""
    G = np.clip(P @ P.T, -1, 1)
    th = np.degrees(np.arccos(G))
    cls = {}
    for i, j in itertools.combinations(range(N), 2):
        a = th[i, j]
        cls[(i, j)] = "edge" if abs(a - 63.4349) < 1e-3 else "second" if abs(a - 116.5651) < 1e-3 else "antipodal" if abs(a - 180) < 1e-3 else "?"
    return th, cls


def overlap(theta_deg):
    """|<a|b>| of two spin-1/2 coherent states at angle theta (also the Moebius invariant s1 = s2 of the Hopf fibres)."""
    return abs(np.cos(np.radians(theta_deg) / 2))


def heisenberg(edges):
    """Sparse H = sum_edges S_i.S_j on 12 spin-1/2, basis = bit strings, and the pair operators for any (i, j)."""
    D = 1 << N
    s = np.arange(D)
    bit = lambda i: (s >> i) & 1

    def pair(i, j):                                    # S_i.S_j = swap/2 - 1/4
        t = np.where(bit(i) == bit(j), s, s ^ (1 << i) ^ (1 << j))
        return sp.csr_matrix((np.full(D, .5), (t, s)), shape=(D, D)) - 0.25 * sp.identity(D, format="csr")
    H = sum((pair(i, j) for i, j in edges), sp.csr_matrix((D, D)))
    return H, pair


def analyse():
    P = icosahedron()
    th, cls = classes(P)
    edges = [e for e, c in cls.items() if c == "edge"]
    out = {"counts": {c: sum(1 for v in cls.values() if v == c) for c in ("edge", "second", "antipodal")}}
    out["angle_deg"] = {"edge": float(th[edges[0]]), "second": float(next(th[e] for e, c in cls.items() if c == "second")), "antipodal": 180.0}
    out["overlap_s"] = {k: float(overlap(v)) for k, v in out["angle_deg"].items()}
    out["cos_over_4"] = {k: float(np.cos(np.radians(v)) / 4) for k, v in out["angle_deg"].items()}
    anti = [e for e, c in cls.items() if c == "antipodal"]
    out["antipodal_pairs_partition"] = sorted(sum(anti, ())) == list(range(N))
    H, pair = heisenberg(edges)
    w, V = eigsh(H, k=3, which="SA")
    order = np.argsort(w); w, V = w[order], V[:, order]
    psi = V[:, 0]
    out["ground_energy"] = float(w[0]); out["next_levels"] = [float(x) for x in w[1:]]
    out["gap"] = float(w[1] - w[0])
    corr = {}
    for c in ("edge", "second", "antipodal"):
        pr = [e for e, k in cls.items() if k == c]
        vals = [float(psi @ (pair(i, j) @ psi)) for i, j in pr]
        corr[c] = dict(mean=float(np.mean(vals)), spread=float(np.ptp(vals)))
    out["ground_correlation"] = corr
    S2 = sum((pair(i, j) @ psi for i, j in itertools.combinations(range(N), 2)), np.zeros(1 << N))
    out["total_pair_sum"] = float(psi @ S2)                       # = (S(S+1) - 9) / 2 for total spin S
    out["total_spin_S2"] = float(2 * out["total_pair_sum"] + 9)   # S(S+1)
    return out


def run_tests(out):
    assert out["counts"] == {"edge": 30, "second": 30, "antipodal": 6}
    assert abs(out["angle_deg"]["edge"] - 63.4349) < 1e-3 and abs(out["angle_deg"]["second"] - 116.5651) < 1e-3
    assert abs(out["overlap_s"]["edge"] - ((1 + 5 ** -.5) / 2) ** .5) < 1e-12
    assert abs(out["overlap_s"]["second"] - ((1 - 5 ** -.5) / 2) ** .5) < 1e-12
    assert out["overlap_s"]["antipodal"] < 1e-12 and out["antipodal_pairs_partition"]
    assert abs(out["ground_energy"] + 6.18789) < 1e-4, out["ground_energy"]      # ring_bits.py: -6.1879, unique
    assert out["gap"] > 1e-3
    assert abs(out["total_spin_S2"]) < 1e-6                                       # a singlet
    cc = out["ground_correlation"]
    assert all(v["spread"] < 1e-6 for v in cc.values())                           # the correlation depends on the class only
    assert abs(30 * cc["edge"]["mean"] + 30 * cc["second"]["mean"] + 6 * cc["antipodal"]["mean"] + 4.5) < 1e-6   # singlet sum rule
    print("hopf_sites: all tests passed")


def check_against_hat():
    """The ball's pentagon centres have the same pair angles (needs hexasphere)."""
    sys.argv = [sys.argv[0]]
    import ring_bits as RB
    ball, fl, T0, rings, P, edges = RB.setup()
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    _, cls = classes(P)
    counts = {c: sum(1 for v in cls.values() if v == c) for c in ("edge", "second", "antipodal", "?")}
    assert counts == {"edge": 30, "second": 30, "antipodal": 6, "?": 0}, counts
    print("hopf_sites: the ball's 12 pentagon centres are a regular icosahedron:", counts)


if __name__ == "__main__":
    res = analyse()
    print(json.dumps(res, indent=1))
    if "--tests" in sys.argv:
        run_tests(res)
    if "--hat" in sys.argv:
        check_against_hat()
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hopf_sites.json"), "w"), indent=1)
