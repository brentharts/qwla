#!/usr/bin/env python3
"""
Closed flip clusters of hat tilings on the sphere (the Hat Trick ball, GP(3,0): 480 kites, 60 hats, 12 pentagon holes).

In the plane a phason flip of the Spectre tiling is a ribbon that does not close inside any region tested
(ROADMAP Stage 3, generic_phase.py). A sphere is compact, so every difference of two tilings is a finite set of
closed clusters. This script finds them for the hat: starting from a tiling T, every connected set C of k hats
(k = 1 .. KMAX) is removed and the vacated kites are tiled again by hats in all possible ways (exact cover). A flip
is a re-tiling that changes every hat of C (so it is not a smaller flip plus unchanged hats). For each flip:
k, the Euler characteristic of the region (1 = disc, 0 = annulus, i.e. a closed ribbon), the change dV of the number
of vertices of the hat complex (corners where 3 or more panels meet; dF = 0 and dE = dV by Euler), and the number
of the 12 pentagons it touches. Then a random walk by flips samples tilings and the flip counts on them.

Needs brentharts/hexasphere (HEXASPHERE_DIR or a sibling folder). Usage: python3 hat_flips.py [KMAX=6] [--tests]
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for c in [os.environ.get("HEXASPHERE_DIR"), os.path.join(HERE, "..", "hexasphere"), os.path.join(HERE, "hexasphere")]:
    if c and os.path.exists(os.path.join(c, "src", "hexasphere", "hattrick", "ball.py")):
        sys.path.insert(0, os.path.join(c, "src"))
        break
else:
    raise SystemExit("set HEXASPHERE_DIR to a clone of brentharts/hexasphere")
from hexasphere.hattrick import HatBall  # noqa: E402


class Flips:
    def __init__(self, ball):
        self.S = ball.S
        self.placements = [frozenset(p) for p in ball.placements]
        self.by_kite = collections.defaultdict(list)
        for i, p in enumerate(self.placements):
            for k in p:
                self.by_kite[k].append(i)
        self.pidx = {p: i for i, p in enumerate(self.placements)}
        self.corners = {k: self.S.corners(k) for k in self.S.kites}
        self.kite_nb = {k: [n for n in self.S.nb[k] if n is not None] for k in self.S.kites}
        self.pent = set(ball.pent)

    def tiling_ids(self, hats):
        return frozenset(self.pidx[frozenset(h)] for h in hats)

    def hat_graph(self, T):
        of = {k: i for i in T for k in self.placements[i]}
        adj = collections.defaultdict(set)
        for k, i in of.items():
            for n in self.kite_nb[k]:
                if of[n] != i:
                    adj[i].add(of[n])
        return of, adj

    def covers(self, region, limit=10000):
        """All exact covers of the kite set `region` by placements inside it."""
        region = frozenset(region)
        cand = {i for k in region for i in self.by_kite[k] if self.placements[i] <= region}
        out = []

        def solve(left, chosen):
            if len(out) >= limit:
                return
            if not left:
                out.append(frozenset(chosen))
                return
            k = min(left, key=lambda q: sum(1 for i in self.by_kite[q] if i in cand and self.placements[i] <= left))
            for i in self.by_kite[k]:
                if i in cand and self.placements[i] <= left:
                    solve(left - self.placements[i], chosen + [i])
        solve(region, [])
        return out

    def region_chi(self, region):
        V, E = set(), set()
        for k in region:
            cs = self.corners[k]
            V.update(cs)
            for a in range(4):
                E.add(frozenset((cs[a], cs[(a + 1) % 4])))
        return len(V) - len(E) + len(region)

    def n_vertices(self, T):
        """Corners of the kite complex where 3 or more panels (hats or pentagons) meet."""
        of = {k: i for i in T for k in self.placements[i]}
        at = collections.defaultdict(set)
        for k, i in of.items():
            for c in self.corners[k]:
                at[c].add(i)
        n = 0
        for c, s in at.items():
            m = len(s) + (1 if c[0] == "v" and c[1] in self.pent else 0)
            n += m >= 3
        return n

    def flips(self, T, kmax):
        """Dict cluster (frozenset of placement ids) -> list of new placement sets, all hats changed."""
        _, adj = self.hat_graph(T)
        seen, found = set(), {}
        frontier = {frozenset([i]) for i in T}
        for k in range(1, kmax + 1):
            nxt = set()
            for C in frontier:
                if C in seen:
                    continue
                seen.add(C)
                region = frozenset().union(*[self.placements[i] for i in C])
                alts = [a for a in self.covers(region) if not (a & C)]
                if alts:
                    found[C] = alts
                if k < kmax:
                    for i in C:
                        for j in adj[i]:
                            if j not in C:
                                nxt.add(C | {j})
            frontier = nxt
        # keep only minimal clusters (no proper sub-cluster that is itself a flip of the same re-tiling part)
        return found

    def describe(self, T, C, new):
        region = frozenset().union(*[self.placements[i] for i in C])
        T2 = (T - C) | new
        touched = {c[1] for k in region for c in self.corners[k] if c[0] == "v" and c[1] in self.pent}
        return dict(k=len(C), chi=self.region_chi(region), dV=self.n_vertices(T2) - self.n_vertices(T), pentagons=len(touched))


def irreducible(found):
    """Drop a flip whose cluster is a disjoint union of two smaller flips' clusters with the same new hats."""
    keys = sorted(found, key=len)
    out = {}
    for C in keys:
        alts = []
        for a in found[C]:
            red = any(D < C and any(b <= a for b in found[D]) for D in out)
            if not red:
                alts.append(a)
        if alts:
            out[C] = alts
    return out


def survey(fl, T, kmax):
    f = irreducible(fl.flips(T, kmax))
    rows = [fl.describe(T, C, a) for C, alts in f.items() for a in alts]
    return f, rows


def run_tests(ball, fl, tilings):
    """Exact checks: 12 disjoint ring flips, 2^12 tilings in all, vertex count = 80 - (edges with both rings flipped)."""
    import itertools
    import numpy as np
    T0, T1 = tilings
    f = irreducible(fl.flips(T0, 5))
    rings = []
    for C, alts in f.items():
        assert len(C) == 5 and len(alts) == 1
        region = frozenset().union(*[fl.placements[i] for i in C])
        assert len(region) == 40 and fl.region_chi(region) == 0                     # an annulus of 5 hats
        p = collections.Counter(v for k in region for v in ball.F[k // 3] if v in fl.pent).most_common(1)[0][0]
        rings.append((p, C, alts[0], region))
    rings.sort(key=lambda r: r[0])
    assert len({r[0] for r in rings}) == 12 and len(frozenset().union(*[r[3] for r in rings])) == 480
    print("1. 12 flips, each the ring of 5 hats around one pentagon (annulus, 40 kites); the rings are disjoint and cover the ball")
    T = T0
    for _, C, new, _ in rings:
        T = (T - C) | new
    assert T == T1
    print("2. flipping all 12 rings of the first icosahedral tiling gives the second")
    allc = fl.covers(frozenset(fl.S.kites), limit=20000)
    assert len(allc) == 4096
    P = np.array([ball.V[r[0]] for r in rings])
    adj = (P @ P.T > 0.3) & (P @ P.T < 0.9)
    assert adj.sum() == 60
    made = set()
    for b in itertools.product([0, 1], repeat=12):
        T = T0
        for bi, (_, C, new, _) in zip(b, rings):
            if bi:
                T = (T - C) | new
        made.add(T)
        e11 = sum(b[i] * b[j] for i in range(12) for j in range(i + 1, 12) if adj[i, j])
        assert fl.n_vertices(T) == 80 - e11
    assert made == set(allc)
    print("3. exact cover of all 480 kites: 4096 = 2^12 hat tilings, exactly the 2^12 ring states")
    print("4. vertices of the hat complex = 80 - (icosahedron edges whose two rings are both flipped), all 4096 states")
    print("all tests passed")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kmax = int(args[0]) if args else 6
    ball = HatBall.from_frequency(3)
    fl = Flips(ball)
    res = {}
    tilings = [fl.tiling_ids(sorted(map(sorted, t))) for t in ball.tilings]
    print("placements", len(fl.placements), "icosahedral tilings", len(tilings), "common hats of the two", len(tilings[0] & tilings[1]))
    for name, T in zip(("I0", "I1"), tilings):
        assert len(T) == 60 and len(frozenset().union(*[fl.placements[i] for i in T])) == 480
        f, rows = survey(fl, T, kmax)
        cnt = collections.Counter((r["k"], r["chi"], r["dV"], r["pentagons"]) for r in rows)
        print(name, "vertices", fl.n_vertices(T), "flips up to k=%d:" % kmax, len(rows))
        for key, n in sorted(cnt.items()):
            print("   k=%d chi=%d dV=%+d pentagons=%d : %d" % (*key, n))
        res[name] = [dict(zip(("k", "chi", "dV", "pentagons", "count"), (*key, n))) for key, n in sorted(cnt.items())]
    if "--tests" in sys.argv:
        run_tests(ball, fl, tilings)
        return
    # random walk by flips from I0
    import random
    rng = random.Random(1)
    T = tilings[0]
    walk = []
    kw = min(kmax, 5)
    seen = {T}
    for step in range(int(os.environ.get("STEPS", "12"))):
        f, rows = survey(fl, T, kw)
        cnt = collections.Counter((r["k"], r["chi"], r["dV"]) for r in rows)
        walk.append(dict(step=step, vertices=fl.n_vertices(T), n_flips=len(rows), types={"k%d chi%d dV%+d" % k: n for k, n in sorted(cnt.items())}))
        print("walk step", step, walk[-1])
        if not f:
            break
        C = rng.choice(sorted(f, key=sorted))
        T = (T - C) | rng.choice(f[C])
        assert len(frozenset().union(*[fl.placements[i] for i in T])) == 480
        seen.add(T)
    res["walk"] = walk
    res["distinct_tilings_in_walk"] = len(seen)
    json.dump(res, open(os.path.join(HERE, "hat_flips_k%d.json" % kmax), "w"), indent=1)
    print("->", "hat_flips_k%d.json" % kmax)


if __name__ == "__main__":
    main()
