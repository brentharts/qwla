#!/usr/bin/env python3
"""
Do closed Spectre folds (ROADMAP Stage 7) have closed flips, in particular around their cone points?

A closed fold is a disc patch of Spectres whose boundary is glued to itself (ballgen, spectre/dp.py); the result is a
flat sphere with cone points. A flip is a second tiling of a disc-shaped cluster of tiles by Spectres. The search is
combinatorial and exact: the cluster's boundary is a cyclic word of interior angles (multiples of 30 degrees, all
edges of length 1), and tilings of the word are found by backtracking: the tile that covers a chosen boundary edge
is one of 14 (one per tile edge); placing it replaces the edge by the tile's other 13 edges, and wherever the
remaining angle at a vertex reaches 0 the two edges there are zipped. A cone point of deficit d inside the cluster
is allowed once: a vertex may be zipped when its nominal angle equals d. A word that zips away completely after
exactly k tiles is a tiling of a flat disc with that boundary (and those cones). For a planar cluster without cone
that disc is the cluster itself (an immersed flat disc with an embedded boundary is embedded); with a cone it is a
flat cone disc with the same boundary and cone angle, which can replace the cluster in the closed surface.

Only clusters with at most one cone point inside are counted: for one cone the flat disc is fixed by its boundary (the
cone is the fixed point of the boundary holonomy), so a second tiling is a flip of the same surface; with two or
more cones the search would also find other surfaces (re-foldings).

Limit of the search: a tile that touches a far part of the boundary splits the region, and the word does not see that,
so some tilings are missed (the original tiling itself is not found for about a quarter of the planar clusters).
Every tiling that is found is valid, so the counts are lower bounds: "2 or more" is a result, "1" or "0" is not.

Model: the straight-edged Spectre, Tile(1,1), one hand only (no reflected tiles), edge kinds not enforced.
Needs brentharts/ballgen (BALLGEN_DIR or a sibling folder). Usage: python3 spectre_fold_flips.py [--tests]
"""
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for c in [os.environ.get("BALLGEN_DIR"), os.path.join(HERE, "..", "ballgen"), os.path.join(HERE, "ballgen")]:
    if c and os.path.exists(os.path.join(c, "spectre", "dp.py")):
        sys.path.insert(0, c)
        break
else:
    raise SystemExit("set BALLGEN_DIR to a clone of brentharts/ballgen")
from spectre.patch import SpectrePatch  # noqa: E402
from spectre.search import PatchPool  # noqa: E402
from spectre.fold import PatchGraph  # noqa: E402
from spectre.dp import can_close, fold_with_holes  # noqa: E402

NE = 14
sys.setrecursionlimit(10000)


def count_tilings(word, k, tau, cones=(), limit=50):
    """Number of tilings (up to `limit`) of the flat disc with boundary angle word `word` by exactly k tiles
    with corner angles `tau`; `cones` are deficits of cone points inside, each used once."""
    count = [0]
    nodes = [0]

    def normalise(a, cones):
        """Zip zero angles. Yields (word, cones) states (branching when a cone may be used)."""
        out = []
        stack = [(a, cones)]
        while stack:
            a, cs = stack.pop()
            if min(a, default=0) < 0:
                continue
            n = len(a)
            if n == 0:
                out.append((a, cs))
                continue
            if n == 2:
                # two edges between the same two vertices: they close iff both vertices are full
                opts = [((), cs)] if (a[0] == 0 and a[1] == 0) else []
                for d in set(cs):
                    rest = list(cs); rest.remove(d); rest = tuple(rest)
                    if (a[0] == d and a[1] == 0) or (a[0] == 0 and a[1] == d):
                        opts.append(((), rest))
                    if a[0] == d and a[1] in rest:
                        r2 = list(rest); r2.remove(a[1]); opts.append(((), tuple(r2)))
                if opts:
                    out.extend(opts)
                elif 0 not in a and not any(x in cs for x in a):
                    out.append((a, cs))             # not closable now; cannot be tiled further either (2-gon), dies later
                continue
            zi = [i for i, x in enumerate(a) if x == 0]
            ci = [i for i, x in enumerate(a) if x in cs and x != 0]
            if zi:
                i = zi[0]
                u, w = (i - 1) % n, (i + 1) % n
                m = a[u] + a[w] - 360
                # vertex u carries the merged angle, v and w are removed
                order = [j for j in range(n) if j not in (i, w)]
                b = [m if j == u else a[j] for j in order]
                stack.append((tuple(b), cs))
                continue
            out.append((a, cs))
            for i in ci:                             # optional: zip at a cone vertex
                u, w = (i - 1) % n, (i + 1) % n
                m = a[u] + a[w] - 360
                order = [j for j in range(n) if j not in (i, w)]
                b = [m if j == u else a[j] for j in order]
                rest = list(cs); rest.remove(a[i])
                stack.append((tuple(b), tuple(rest)))
        return out

    seen_done = set()

    def solve(a, cs, left):
        nodes[0] += 1
        if count[0] >= limit or nodes[0] > 2_000_000:
            return
        if len(a) == 0:
            if left == 0 and not cs:
                count[0] += 1
            return
        if left == 0 or len(a) < 3:
            return
        n = len(a)
        i = min(range(n), key=lambda j: (a[j], j))
        j = (i + 1) % n
        for e in range(NE):
            x, y = a[i] - tau[e], a[j] - tau[(e + 1) % NE]
            if x < 0 or y < 0:
                continue
            mid = [360 - tau[(e - s) % NE] for s in range(1, NE - 1)]
            if j == i + 1:
                b = a[:i] + (x,) + tuple(mid) + (y,) + a[j + 1:]
            else:                                   # i is the last index, j = 0
                b = (y,) + a[1:i] + (x,) + tuple(mid)
            for b2, cs2 in normalise(b, cs):
                solve(b2, cs2, left - 1)

    for a0, cs0 in normalise(tuple(word), tuple(sorted(cones))):
        solve(a0, cs0, k)
    return count[0], nodes[0]


class Surface:
    """Tiles with mates for every half-edge: a planar patch (mates only inside) or a closed fold."""

    def __init__(self, graph, fold=None):
        self.g = graph
        self.T = graph.n_tiles
        self.mate = dict(graph.partner)
        self.cls = list(range(len(graph.xy)))
        self.deficit = {}
        if fold is not None:
            for h1, h2 in fold.glued:
                self.mate[h1] = h2
                self.mate[h2] = h1
            self.cls = [fold.find(v) for v in range(len(graph.xy))]
            for c in fold.cones:
                if c.deficit > 0:
                    self.deficit[fold.find(c.vertex)] = c.deficit
        self.corners = collections.defaultdict(list)
        for t in range(self.T):
            for k in range(NE):
                self.corners[self.cls[int(graph.tile_vertex[t, k])]].append((t, k))
        self.nbrs = collections.defaultdict(set)
        for h, o in self.mate.items():
            if h // NE != o // NE:
                self.nbrs[h // NE].add(o // NE)

    def word(self, cluster):
        """(angle word, interior cone deficits) of a cluster that is a disc with a simple boundary, else None."""
        C = set(cluster)
        g = self.g
        bnd = [h for t in C for h in range(NE * t, NE * t + NE) if self.mate.get(h, -1) // NE not in C or h not in self.mate]
        if any(h not in self.mate for h in bnd) and False:
            return None
        if not bnd:
            return None
        bset = set(bnd)
        word, vs, used = [], [], set()
        h = bnd[0]
        while h not in used:
            used.add(h)
            # angle at the END vertex of h, then next boundary half-edge
            t, k = divmod(h, NE)
            k2 = (k + 1) % NE
            ang = int(g.angle[t, k2])
            nh = NE * t + k2
            guard = 0
            while nh not in bset:
                o = self.mate[nh]
                t, k = divmod(o, NE)
                k2 = (k + 1) % NE
                ang += int(g.angle[t, k2])
                nh = NE * t + k2
                guard += 1
                if guard > 50:
                    return None
            word.append(ang)
            vs.append(self.cls[int(g.tile_vertex[nh // NE, nh % NE])])
            h = nh
        if len(used) != len(bnd) or len(set(vs)) != len(vs):
            return None
        on_b = set(vs)
        cones = []
        for c, d in self.deficit.items():
            if c not in on_b and all(t in C for t, _ in self.corners[c]):
                cones.append(d)
        # the word lists the angle at the end of each boundary half-edge of the CLUSTER walked counter-clockwise
        # around each tile, i.e. with the cluster on the left: these are the interior angles of the region to fill.
        return tuple(word), tuple(cones)

    def clusters(self, kmax, seeds=None):
        seen = set()
        frontier = {frozenset([t]) for t in (seeds if seeds is not None else range(self.T))}
        for k in range(1, kmax + 1):
            nxt = set()
            for C in frontier:
                if C in seen:
                    continue
                seen.add(C)
                yield C
                if k < kmax:
                    for t in C:
                        for u in self.nbrs[t]:
                            if u not in C:
                                nxt.add(C | {u})
            frontier = nxt


def scan(surface, kmax, tau, seeds=None, extra=()):
    stats = collections.Counter()
    hits = []
    todo = list(surface.clusters(kmax, seeds)) + [frozenset(c) for c in extra]
    done = set()
    for C in todo:
        if C in done:
            continue
        done.add(C)
        w = surface.word(C)
        if w is None:
            stats["not a simple disc"] += 1
            continue
        word, cones = w
        if len(cones) > 1:
            stats["more than one cone inside (surface not fixed by the boundary; skipped)"] += 1
            continue
        n, nodes = count_tilings(word, len(C), tau, cones)
        stats[("k=%d cones=%d" % (len(C), len(cones)), "tilings=%d" % n)] += 1
        if n != 1:
            hits.append(dict(k=len(C), cones=list(cones), tilings=n, word=list(word), tiles_in_surface=surface.T))
    return stats, hits


def run(per_n=25, seed=5, Ns=range(4, 22), kmax_fold=4, kmax_plane=5):
    patch = SpectrePatch(levels=3)
    pool = PatchPool(patch)
    g0 = PatchGraph(patch)
    tau = [int(x) for x in g0.angle[0]]
    assert all(sorted(int(x) for x in g0.angle[t]) == sorted(tau) for t in range(g0.n_tiles))
    res = {"tile_angles": tau}
    # sanity: every tile has the same cyclic angle sequence
    assert all([int(x) for x in g0.angle[t]] == tau for t in range(g0.n_tiles))
    n1, _ = count_tilings(tuple(tau), 1, tau)
    assert n1 == 1
    print("tile angles", tau, "; one tile fills its own outline in", n1, "way")
    # planar clusters of the substitution patch
    S0 = Surface(g0)
    import numpy as np
    cen = np.array([patch.straight_polygon(t).mean(0) for t in range(g0.n_tiles)])
    seeds = [int(t) for t in np.argsort(np.linalg.norm(cen - cen.mean(0), axis=1))[:40]]
    st, hits = scan(S0, kmax_plane, tau, seeds)
    tot = collections.Counter()
    for (a, b), n in ((k, v) for k, v in st.items() if isinstance(k, tuple)):
        tot[(a.split()[0], b)] += n
    print("planar clusters (level-3 patch, 40 central seed tiles, k <= %d):" % kmax_plane, dict(tot), "not discs:", st["not a simple disc"])
    res["plane"] = {"counts": {"%s %s" % k: v for k, v in tot.items()}, "hits": hits[:20]}
    # closed folds
    rng = random.Random(seed)
    agg = collections.Counter()
    allhits = []
    nfold = 0
    stars = collections.Counter()
    for N in Ns:
        for _ in range(per_n):
            g = pool.graph(pool.random_subset(N, rng, rng.choice([0.0, 0.3, 0.7])))
            if g is None:
                continue
            A = [int(a) for a in g.loops[0][1]]
            if not can_close(A):
                continue
            r = fold_with_holes(g, [])
            if r is None:
                continue
            nfold += 1
            S = Surface(g, r)
            star = [frozenset(t for t, _ in S.corners[c]) for c in S.deficit]
            star = [s for s in star if len(s) < S.T]
            st, hits = scan(S, min(kmax_fold, N - 1), tau, extra=star)
            for C in star:
                w = S.word(C)
                if w is None:
                    stars["star is not a simple disc"] += 1
                elif len(w[1]) > 1:
                    stars["star holds more than one cone"] += 1
                else:
                    n, _ = count_tilings(w[0], len(C), tau, w[1])
                    stars["star of a cone point: tilings=%d" % n] += 1
            for k, v in st.items():
                if isinstance(k, tuple):
                    agg[(k[0], k[1])] += v
                else:
                    agg[k] += v
            for h in hits:
                h["N"] = N
            allhits += hits
    print("closed folds:", nfold)
    for k, v in sorted(agg.items(), key=str):
        print("  ", k, v)
    print("stars of cone points:", dict(stars))
    res["folds"] = {"n": nfold, "counts": {str(k): v for k, v in agg.items()}, "stars": dict(stars), "hits": allhits[:40], "n_hits": len(allhits)}
    json.dump(res, open(os.path.join(HERE, "spectre_fold_flips.json"), "w"), indent=1)
    print("-> spectre_fold_flips.json")
    return res


if __name__ == "__main__":
    if "--tests" in sys.argv:
        run(per_n=4, Ns=range(4, 10), kmax_fold=3, kmax_plane=3)
    else:
        run()
