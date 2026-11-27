#!/usr/bin/env python3
"""
Closed surfaces from two different Spectre patches glued along their boundaries (open problem 4 of quantgravbits.tex;
ROADMAP Stage 9g).

Self-folds of one Spectre patch stop at about 15 tiles, and a patch never closes with a copy of itself (Stage 9e). Here
two different disc patches P, Q of the level-3 patch are glued boundary to boundary: with boundary angle lists A (of P)
and B (of Q), both walked with the patch on the left and of the same length m, edge i of P is glued to edge (s - i) of Q
reversed, so vertex i of P meets vertex (s - i + 1) of Q, and the gluing is allowed when A_i + B_(s-i+1) <= 360 for every
i (a cone point of deficit 360 - A_i - B_(s-i+1) where the sum is below 360; the deficits then add to 720 by itself).
With --kinds the glued edges must also have the same edge kind, as in a planar Spectre tiling.

For each closed surface: number of cone points, deficits, pairs (two flat corners that meet alone: inside P, inside Q,
or across the seam), and what a local swap of one tile of P does (does the surface still close with the same Q, do the
deficits change). Counted, random samples, straight-edged Spectre. Needs brentharts/ballgen.
--small instead pairs every distinct outline of a connected cluster of 1 to 5 tiles around 40 central tiles (3666
outlines, all 670,647 pairs, a pair counted once whatever the shift) and reports the closed surfaces found.

Usage: python3 spectre_complement.py [patches_per_size=400] [--kinds]     or     python3 spectre_complement.py --small
"""
import collections
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for c in [os.environ.get("BALLGEN_DIR"), os.path.join(HERE, "..", "ballgen")]:
    if c and os.path.exists(os.path.join(c, "spectre", "dp.py")):
        sys.path.insert(0, c)
        break
from spectre.patch import SpectrePatch  # noqa: E402
from spectre.search import PatchPool  # noqa: E402

FLAT = 8                     # corner index of the flat (180 degree) vertex of the Spectre


class Disc:
    def __init__(self, pool, tiles):
        g = pool.graph(tiles)
        if g is None:
            raise ValueError
        self.tiles = frozenset(tiles)
        edges, ang = g.loops[0]
        self.A = np.array([int(a) for a in ang])
        self.m = len(self.A)
        self.kind = np.array([str(g.edge_kind[h % 14]) for h in edges])
        corners = collections.defaultdict(list)
        for t in range(g.n_tiles):
            for k in range(14):
                corners[int(g.tile_vertex[t, k])].append(k)
        bverts = [g.start_vertex(h) for h in edges]
        bset = set(bverts)
        self.inner_pairs = sum(1 for v, ks in corners.items() if v not in bset and ks == [FLAT, FLAT])
        self.flat_alone = np.array([corners[v] == [FLAT] for v in bverts])        # boundary vertex = one flat corner


def gluings(P, Q, kinds=False):
    """Shifts s for which P and Q glue shut."""
    if P.m != Q.m:
        return []
    m = P.m
    i = np.arange(m)
    out = []
    for s in range(m):
        j = (s - i + 1) % m
        if (P.A + Q.A[j] <= 360).all():
            if kinds and not (P.kind == Q.kind[(s - i) % m]).all():
                continue
            out.append(s)
    return out


def surface(P, Q, s):
    m = P.m
    i = np.arange(m)
    j = (s - i + 1) % m
    d = 360 - P.A - Q.A[j]
    seam_pairs = int((P.flat_alone & Q.flat_alone[j]).sum())
    return dict(cones=int((d > 0).sum()), deficits=sorted(int(x) for x in d[d > 0]), pairs=P.inner_pairs + Q.inner_pairs + seam_pairs,
                seam_pairs=seam_pairs, tiles=len(P.tiles) + len(Q.tiles))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    per = int(args[0]) if args else 400
    kinds = "--kinds" in sys.argv
    patch = SpectrePatch(levels=3)
    pool = PatchPool(patch)
    rng = random.Random(11)
    t0 = time.time()
    discs = []
    for N in (6, 10, 15, 20, 30, 40, 60, 80, 120):
        for _ in range(per):
            try:
                discs.append(Disc(pool, pool.random_subset(N, rng, rng.choice([0.3, 0.7, 0.9]))))
            except ValueError:
                pass
    by_m = collections.defaultdict(list)
    seen = set()
    for D in discs:
        if D.tiles not in seen:
            seen.add(D.tiles)
            by_m[D.m].append(D)
    print("disc patches", len(seen), "boundary lengths", min(by_m), "to", max(by_m), "%.0fs" % (time.time() - t0), flush=True)
    found, tested = [], 0
    for m, L in sorted(by_m.items()):
        for a in range(len(L)):
            for b in range(a + 1, len(L)):
                tested += 1
                for s in gluings(L[a], L[b], kinds):
                    found.append((L[a], L[b], s))
    print("pairs of patches tested", tested, "closed surfaces", len(found), "%.0fs" % (time.time() - t0), flush=True)
    rows = [surface(P, Q, s) for P, Q, s in found]
    by_size = collections.defaultdict(list)
    for r in rows:
        by_size[r["tiles"] // 10 * 10].append(r)
    summary = {}
    for k in sorted(by_size):
        R = by_size[k]
        summary[k] = dict(surfaces=len(R), max_pairs=max(r["pairs"] for r in R), mean_pairs=round(float(np.mean([r["pairs"] for r in R])), 2),
                          cones=[min(r["cones"] for r in R), max(r["cones"] for r in R)])
        print("  %3d-%3d tiles: %s" % (k, k + 9, summary[k]), flush=True)
    # local swaps of one tile of P, keeping Q
    sw = collections.Counter()
    order = sorted(range(len(found)), key=lambda n: -rows[n]["pairs"])[:40]
    for n in order:
        P, Q, s = found[n]
        T = set(P.tiles)
        cand = [(t, u) for t in sorted(T) for u in sorted(pool.nbrs[t] - T)]
        rng.shuffle(cand)
        for t, u in cand[:30]:
            try:
                P2 = Disc(pool, sorted((T - {t}) | {u}))
            except ValueError:
                continue
            sw["swaps tried"] += 1
            g2 = gluings(P2, Q, kinds)
            if not g2:
                continue
            sw["still closes"] += 1
            new = [surface(P2, Q, s2) for s2 in g2]
            sw["deficits change"] += all(x["deficits"] != rows[n]["deficits"] for x in new)
            sw["pairs change"] += all(x["pairs"] != rows[n]["pairs"] for x in new)
    print("local swaps of P with Q fixed:", dict(sw))
    best = max(rows, key=lambda r: r["pairs"]) if rows else None
    out = dict(kinds=kinds, discs=len(seen), pairs_tested=tested, surfaces=len(rows), by_tiles=summary, swaps=dict(sw), best=best)
    json.dump(out, open(os.path.join(HERE, "spectre_complement%s.json" % ("_kinds" if kinds else "")), "w"), indent=1)
    print("best:", best)


def small():
    """All pairs of distinct outlines of clusters of 1 to 5 tiles (see the module docstring)."""
    import itertools
    sys.path.insert(0, HERE)
    _a, sys.argv[:] = list(sys.argv), [sys.argv[0]]
    import spectre_fold_flips as SF
    sys.argv[:] = _a
    from spectre.fold import PatchGraph
    patch = SpectrePatch(levels=3)
    pool = PatchPool(patch)
    g0 = PatchGraph(patch)
    S0 = SF.Surface(g0)
    cen = np.array([patch.straight_polygon(t).mean(0) for t in range(g0.n_tiles)])
    seeds = [int(t) for t in np.argsort(np.linalg.norm(cen - cen.mean(0), axis=1))[:40]]
    SC = sys.modules[__name__]
    t0 = time.time()

    def canon(w):
        n=len(w); return min(tuple(w[k:]+w[:k]) for k in range(n))
    words={}
    for C in S0.clusters(5,seeds):
        w=S0.word(C)
        if w is None: continue
        key=(len(C),canon(list(w[0])))
        words.setdefault(key,C)
    print('distinct outlines',len(words),'%.0fs'%(time.time()-t0),flush=True)
    by_m=collections.defaultdict(list)
    for (k,cw),C in words.items(): by_m[len(cw)].append((k,np.array(cw),C))
    res=collections.Counter(); found=[]
    for m,L in sorted(by_m.items()):
        i=np.arange(m)
        for a,b in itertools.combinations_with_replacement(range(len(L)),2):
            A,B=L[a][1],L[b][1]
            for s in range(m):
                if (A+B[(s-i+1)%m]<=360).all():
                    found.append((L[a],L[b],a==b)); break
    print('pairs tested',sum(len(L)*(len(L)+1)//2 for L in by_m.values()),'closing pairs of outlines',len(found),'%.0fs'%(time.time()-t0),flush=True)
    out=[]
    for (ka,A,Ca),(kb,B,Cb),same in found:
        P=SC.Disc(pool,[g0.tile_ids[t] for t in Ca]); Q=SC.Disc(pool,[g0.tile_ids[t] for t in Cb])
        gs=SC.gluings(P,Q); gk=SC.gluings(P,Q,kinds=True)
        if not gs: res['word closed but disc did not (orientation)']+=1; continue
        best=max((SC.surface(P,Q,s) for s in gs),key=lambda r:r['pairs'])
        res['self (double)' if same else 'two different patches']+=1
        res['edge kinds also match']+=bool(gk)
        out.append(dict(tiles_PQ=[ka,kb],double=same,kinds_match=bool(gk),**best))
    print(dict(res))
    by=collections.defaultdict(list)
    for r in out: by[r['tiles']].append(r)
    for n in sorted(by): print(' total tiles',n,'surfaces',len(by[n]),'max pairs',max(r['pairs'] for r in by[n]),'cones',sorted(set(r['cones'] for r in by[n])))
    json.dump(dict(outlines=len(words),closing=len(out),counts=dict(res),surfaces=out),open(os.path.join(HERE, 'spectre_complement_small.json'), 'w'),indent=1)


if __name__ == "__main__":
    if "--small" in sys.argv:
        small()
    else:
        main()
