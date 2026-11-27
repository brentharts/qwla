#!/usr/bin/env python3
"""
Entanglement of the level-2 ground state of the 12 punctures (NULL_CONE_CONNECTIONS.md, section 3C3).

At level 2 the 12 punctures are a free Majorana model (qdeform_punctures.py, Stage 9c):
    H_2 = -15/2 - (i/2) sum_<pq> s_pq gamma_p gamma_q,   a_pq = s_pq = -a_qp on the 30 icosahedron edges,   a^2 = -5.
Because a^2 = -5, a/sqrt5 is a complex structure, and the ground state is the Gaussian state with covariance
    G_pq = <i gamma_p gamma_q> = a_pq / sqrt5        (p != q):
the two-point function is 1/sqrt5 on edges and exactly zero beyond nearest neighbours.               (Exact)

For a set R of cone points the reduced state on the Majoranas of R is Gaussian with covariance G_R; with singular
values nu_1 >= nu_2 >= ... of G_R (each appears twice, plus one zero when |R| is odd)
    S(R) = sum_pairs h((1+nu)/2)  [+ (1/2) ln 2 when |R| is odd],   h(x) = -x ln x - (1-x) ln(1-x).
The odd case is a convention (the algebra of an odd number of Majoranas has a centre; ln sqrt2 is the quantum
dimension of one Ising anyon), so the fit is reported for even |R| and for all |R|.

Signs: any antisymmetric sign pattern with a^2 = -5 gives the same entropies up to gamma_p -> -gamma_p; one is found
by search here, so the script needs only numpy.   Usage: python3 null_cone_level2.py [--tests]
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def icosahedron():
    g = (1 + 5 ** 0.5) / 2
    P = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            P += [(0, s1, s2 * g), (s1, s2 * g, 0), (s2 * g, 0, s1)]
    P = np.array(P, float)
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if abs(D[i, j] - 2) < 1e-9]
    assert len(edges) == 30
    return P, edges


def signed_adjacency(edges):
    """An antisymmetric sign matrix on the edges with a^2 = -5 (gauge fixed on a spanning tree)."""
    nb = {i: [] for i in range(12)}
    for i, j in edges:
        nb[i].append(j); nb[j].append(i)
    tree, seen, stack = set(), {0}, [0]
    while stack:
        u = stack.pop()
        for v in nb[u]:
            if v not in seen:
                seen.add(v); tree.add((min(u, v), max(u, v))); stack.append(v)
    free = [e for e in edges if e not in tree]
    a = np.zeros((12, 12))
    for i, j in tree:
        a[i, j], a[j, i] = 1, -1
    sols = []

    F = len(free)
    base = a.copy()
    for mask in range(1 << F):
        m = base.copy()
        for k, (i, j) in enumerate(free):
            s = 1 if (mask >> k) & 1 else -1
            m[i, j], m[j, i] = s, -s
        if abs(m[0] @ m[:, 1]) > 1e-9:                    # cheap rejection on one entry of a^2 first
            continue
        if np.allclose(m @ m, -5 * np.eye(12)):
            sols.append(m)
            if len(sols) == 2:
                break
    return sols


def h(x):
    return 0.0 if x <= 1e-15 or x >= 1 - 1e-15 else -x * math.log(x) - (1 - x) * math.log(1 - x)


def entropy(G, R):
    R = list(R)
    sv = np.linalg.svd(G[np.ix_(R, R)], compute_uv=False)
    sv = np.sort(np.clip(sv, 0, 1))[::-1]
    S = sum(h((1 + sv[2 * k]) / 2) for k in range(len(R) // 2))
    return S + (0.5 * math.log(2) if len(R) % 2 else 0.0)


def fit(rows):
    X = np.array([[c, m, 1.0] for m, c, S in rows]); y = np.array([S for *_, S in rows])
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ co) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    X1 = X[:, 1:]; c1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r2s = 1 - ((y - X1 @ c1) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(per_cut_edge=float(co[0]), per_puncture=float(co[1]), const=float(co[2]), R2_cut_and_size=float(r2), R2_size_only=float(r2s), regions=len(rows))


def main():
    tests = "--tests" in sys.argv
    P, edges = icosahedron()
    sols = signed_adjacency(edges)
    assert sols, "no sign pattern with a^2 = -5"
    a = sols[0]
    G = a / 5 ** 0.5
    ev = np.linalg.eigvals(1j * a).real
    assert np.allclose(np.sort(np.abs(ev)), 5 ** 0.5)
    E0 = -7.5 - 0.5 * np.abs(ev).sum() / 2 * 1.0          # -15/2 - (1/2) sum of the six positive mode energies
    assert abs(E0 - (-7.5 - 3 * 5 ** 0.5)) < 1e-9
    # pure state: G^2 = -1; region and complement have the same entropy
    assert np.allclose(G @ G, -np.eye(12))
    rows = []
    for m in range(1, 7):
        for R in itertools.combinations(range(12), m):
            if m == 6 and 0 not in R:
                continue
            cut = sum((i in R) != (j in R) for i, j in edges)
            rows.append((m, cut, entropy(G, R)))
    assert len(rows) == 2047
    for R in ((0, 1, 2), (0, 3, 5, 7), (1, 2, 3, 4, 5)):
        Rc = [p for p in range(12) if p not in R]
        assert abs(entropy(G, R) - entropy(G, Rc)) < 1e-9
    if len(sols) > 1:                                     # a second sign pattern gives the same entropies
        G2 = sols[1] / 5 ** 0.5
        assert all(abs(entropy(G2, R) - entropy(G, R)) < 1e-9 for R in itertools.combinations(range(12), 4))
    by = {}
    for m in range(1, 7):
        sub = [(c, S) for mm, c, S in rows if mm == m]
        c = np.array([x[0] for x in sub], float); S = np.array([x[1] for x in sub])
        by[m] = dict(n=len(sub), S_mean=float(S.mean()), S_min=float(S.min()), S_max=float(S.max()), cut_min=int(c.min()), cut_max=int(c.max()),
                     corr_S_cut=(float(np.corrcoef(c, S)[0, 1]) if c.std() > 0 and S.std() > 1e-12 else None))
    out = dict(ground_energy=float(E0), correlation_on_edges=1 / 5 ** 0.5, fit_all=fit(rows), fit_even=fit([r for r in rows if r[0] % 2 == 0]),
               entropy_by_region_size=by, max_entropy_half=max(S for m, c, S in rows if m == 6), ln_32=math.log(32), six_ln_sqrt2=3 * math.log(2))
    json.dump(out, open(os.path.join(HERE, "null_cone_level2.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    if tests:
        print("null_cone_level2: all tests passed")


if __name__ == "__main__":
    main()
