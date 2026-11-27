#!/usr/bin/env python3
"""
Is the hand of the hat visible in the ring bits of the Hat Trick ball? (ROADMAP Stage 9c)

All 4096 hat tilings use one hand of the hat. A reflection s of the icosahedron permutes the 12 ring bits; the
tiling T(s b) is again a tiling by hats of the same hand, not the mirror image of T(b). Test: an orientation-blind
invariant of a tiling (a Weisfeiler-Lehman hash of the graph of panels, edge weight = number of shared kite sides,
6 rounds) is compared for b and s b. If T(s b) were always the mirror image of T(b) as a cell complex, the hashes would
agree. Also compared: simple counts (histogram of shared sides, vertex types, hat degrees).
Needs brentharts/hexasphere. Usage: python3 hat_chirality.py   (about 2 minutes)
"""
import collections
import hashlib
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import ring_bits as RB  # noqa: E402
sys.argv = _argv


def main():
    ball, fl, T0, rings, P, edges = RB.setup()
    perms = RB.rotation_perms(ball, fl, T0, rings, P)
    g = np.asarray(ball.G[1])
    refl = [int(np.argmax(P @ (-(g @ p)))) for p in P]            # -g is improper

    def act(b, perm):
        b2 = [0] * 12
        for i in range(12):
            b2[perm[i]] = b[i]
        return tuple(b2)

    def data(T):
        of = {k: i for i in T for k in fl.placements[i]}
        w = collections.Counter()
        at = collections.defaultdict(set)
        for k, i in of.items():
            for n in fl.kite_nb[k]:
                if of[n] != i:
                    w[(i, of[n])] += 1
            for c in fl.corners[k]:
                at[c].add(i)
                if c[0] == "v" and c[1] in fl.pent:
                    w[(i, ("p", c[1]))] += 1
                    w[(("p", c[1]), i)] += 1
        nb = collections.defaultdict(list)
        for (a, b), x in w.items():
            nb[a].append((b, x))
        col = {v: ("P" if isinstance(v, tuple) else "H") for v in nb}
        for _ in range(6):
            col = {v: hashlib.md5(repr((col[v], sorted((col[u], x) for u, x in nb[v]))).encode()).hexdigest() for v in nb}
        wl = hashlib.md5(repr(sorted(col.values())).encode()).hexdigest()
        hist = tuple(sorted(collections.Counter(x for (a, b), x in w.items() if not isinstance(a, tuple) and not isinstance(b, tuple)).items()))
        vt = tuple(sorted(collections.Counter((c[0], len(s)) for c, s in at.items() if len(s) >= 3).items()))
        deg = tuple(sorted(collections.Counter(sum(1 for u, _ in nb[v] if not isinstance(u, tuple)) for v in nb if not isinstance(v, tuple)).items()))
        return wl, (hist, vt, deg)

    allb = list(itertools.product([0, 1], repeat=12))
    D = {b: data(RB.tiling_of(b, T0, rings)) for b in allb}
    orb = {}
    for b in allb:
        if b not in orb:
            o = frozenset(act(b, p) for p in perms)
            for x in o:
                orb[x] = o
    chiral = [b for b in allb if act(b, refl) not in orb[b]]
    res = dict(tilings=len(allb), classes_up_to_rotation=len(set(orb.values())), wl_classes=len({d[0] for d in D.values()}),
               chiral_bit_patterns=len(chiral), min_flipped_in_a_chiral_pattern=min(sum(b) for b in chiral),
               wl_changes_under_rotation=sum(D[b][0] != D[act(b, perms[7])][0] for b in allb),
               wl_differs_from_mirror_pattern=sum(D[b][0] != D[act(b, refl)][0] for b in chiral),
               simple_counts_differ_from_mirror_pattern=sum(D[b][1] != D[act(b, refl)][1] for b in chiral))
    assert res["wl_changes_under_rotation"] == 0
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "hat_chirality.json"), "w"), indent=1)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------------------------------------------------
# The 4-ring chiral term
# ------------------------------------------------------------------------------------------------------------
def helical_paths(P, edges):
    """Paths a-b-c-d of three edges from a vertex a of the icosahedron to its antipode d, with handedness +1 if the
    path turns left at b (seen from outside) and -1 if it turns right. 60 vertex sets, 30 of each hand."""
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    out = {}
    for a in range(12):
        d = int(np.argmin(P @ P[a]))
        for b in adj[a]:
            for c in adj[b] & adj[d]:
                # the straight continuation of a->b on the sphere points at d; c is to the left or right of it
                h = 1 if np.linalg.det(np.array([P[a], P[b], P[c]])) > 0 else -1
                out[frozenset((a, b, c, d))] = h
    return out


def chi4(b, paths):
    return sum(h * all(b[i] for i in S) for S, h in paths.items())


def chiral_term():
    """Checks that the hand of the hat first appears, in counts of local environments, as the 4-ring term chi4."""
    ball, fl, T0, rings, P, edges = RB.setup()
    g = np.asarray(ball.G[1])
    refl = [int(np.argmax(P @ (-(g @ p)))) for p in P]
    paths = helical_paths(P, edges)
    assert len(paths) == 60 and sum(paths.values()) == 0

    def act(b, perm):
        b2 = [0] * 12
        for i in range(12):
            b2[perm[i]] = b[i]
        return tuple(b2)

    def counts(T, R=3):
        of = {k: i for i in T for k in fl.placements[i]}
        w = collections.Counter()
        for k, i in of.items():
            for n in fl.kite_nb[k]:
                if of[n] != i:
                    w[(i, of[n])] += 1
            for c in fl.corners[k]:
                if c[0] == "v" and c[1] in fl.pent:
                    w[(i, ("p", c[1]))] += 1
                    w[(("p", c[1]), i)] += 1
        nb = collections.defaultdict(list)
        for (a, b), x in w.items():
            nb[a].append((b, x))
        col = {v: ("P" if isinstance(v, tuple) else "H") for v in nb}
        for _ in range(R):
            col = {v: hashlib.md5(repr((col[v], sorted((col[u], x) for u, x in nb[v]))).encode()).hexdigest()[:10] for v in nb}
        return collections.Counter(col.values())

    allb = list(itertools.product([0, 1], repeat=12))
    idx = {b: n for n, b in enumerate(allb)}
    D = [counts(RB.tiling_of(b, T0, rings)) for b in allb]
    cols = sorted(set().union(*[set(d) for d in D]))
    N = np.array([[d[c] for c in cols] for d in D], float)
    mir = [idx[act(b, refl)] for b in allb]
    O = (N - N[mir]) / 2                                         # the part of each count that is odd under reflection
    X = np.array([chi4(b, paths) for b in allb], float)
    assert all(X[mir[n]] == -X[n] for n in range(len(allb)))

    def moebius(f):
        a = f.reshape([2] * 12).copy()
        for ax in range(12):
            a = np.stack([np.take(a, 0, axis=ax), np.take(a, 1, axis=ax) - np.take(a, 0, axis=ax)], axis=ax)
        return a
    aX = moebius(X)
    order = np.indices([2] * 12).sum(0)
    res = dict(local_environment_classes=len(cols), classes_with_a_chiral_part=0, lowest_order=None, order4_proportional_to_chi4=0, slopes=collections.Counter())
    lows = set()
    for c in range(len(cols)):
        if np.abs(O[:, c]).max() == 0:
            continue
        res["classes_with_a_chiral_part"] += 1
        a = moebius(O[:, c])
        nz = np.abs(a) > 1e-9
        lows.add(int(order[nz].min()))
        assert not nz[order < 4].any()
        a4, x4 = a[order == 4], aX[order == 4]
        lam = (a4 @ x4) / (x4 @ x4)
        if np.abs(a4 - lam * x4).max() < 1e-9:
            res["order4_proportional_to_chi4"] += 1
            res["slopes"][round(float(lam), 6)] += 1
    res["lowest_order"] = sorted(lows)
    res["slopes"] = {str(k): v for k, v in sorted(res["slopes"].items())}
    res["rank_of_chiral_parts"] = int(np.linalg.matrix_rank(O))
    res["chi4_values"] = {str(int(k)): int(v) for k, v in sorted(collections.Counter(X).items())}
    res["patterns_with_chi4_nonzero"] = int((X != 0).sum())
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "hat_chirality_chi4.json"), "w"), indent=1)
    return res


if __name__ == "__main__" and "--chi4" in sys.argv:
    chiral_term()
