#!/usr/bin/env python3
"""
A closed substrate with movable cone points AND many pair qubits? (open problem 4 of quantgravbits.tex; Stage 9e)

The hat sphere has 30 pair qubits and fixed cone points; closed folds of Spectre patches of up to 15 tiles have movable
cone points and at most 3 pairs (quantgravalt.tex). Here larger folds are sampled: random disc patches of N tiles of the
level-3 Spectre patch that can be glued shut (exact DP of ballgen, first gluing found), for N = 10 ... 80. For each fold:
the cone-point deficits, the number of bump-and-dent pairs on the closed surface (two flat corners meeting alone, across
the seam included), and what a local swap does (one tile exchanged for a neighbouring tile of the patch, when the new
patch also closes): does the multiset of deficits change, does the pair set change.

Counted, on random samples; one gluing per patch. Needs brentharts/ballgen. Usage: python3 spectre_fold_substrate.py [draws=60]
"""
import collections
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_a, sys.argv = sys.argv, [sys.argv[0]]
import spectre_fold_flips as SF  # noqa: E402  (sets the ballgen path)
sys.argv = _a
from spectre.patch import SpectrePatch  # noqa: E402
from spectre.search import PatchPool  # noqa: E402
from spectre.dp import can_close, fold_with_holes  # noqa: E402


def fold(pool, tiles):
    g = pool.graph(tiles)
    if g is None:
        return None
    if not can_close([int(a) for a in g.loops[0][1]]):
        return None
    r = fold_with_holes(g, [])
    if r is None:
        return None
    S = SF.Surface(g, r)
    pairs = set()
    for c, corners in S.corners.items():
        if len(corners) == 2 and all(int(g.angle[t, k]) == 180 for t, k in corners):
            pairs.add(frozenset(g.tile_ids[t] for t, _ in corners))
    planar = 0
    for v in range(len(g.xy)):
        cs = [(t, k) for t in range(g.n_tiles) for k in range(14) if int(g.tile_vertex[t, k]) == v] if False else None
    return dict(deficits=tuple(sorted(c.deficit for c in r.cones if c.deficit > 0)), pairs=frozenset(pairs), tiles=frozenset(tiles))


def main():
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    patch = SpectrePatch(levels=3)
    pool = PatchPool(patch)
    rng = random.Random(7)
    rows, best = [], None
    t0 = time.time()
    for N in (10, 15, 20, 30, 40, 60, 80):
        folds = []
        tried = 0
        while tried < draws:
            tried += 1
            tiles = pool.random_subset(N, rng, rng.choice([0.3, 0.7, 0.9]))
            f = fold(pool, tiles)
            if f:
                folds.append(f)
        sw = collections.Counter()
        for f in folds[:12]:
            T = set(f["tiles"])
            cand = []
            for t in sorted(T):
                for u in sorted(pool.nbrs[t] - T):
                    cand.append((t, u))
            rng.shuffle(cand)
            done = 0
            for t, u in cand:
                if done >= 12:
                    break
                f2 = fold(pool, sorted((T - {t}) | {u}))
                if f2 is None:
                    continue
                done += 1
                sw["swaps"] += 1
                sw["deficits change"] += f2["deficits"] != f["deficits"]
                sw["number of cones changes"] += len(f2["deficits"]) != len(f["deficits"])
                sw["pairs change"] += f2["pairs"] != f["pairs"]
        npairs = [len(f["pairs"]) for f in folds]
        ncones = [len(f["deficits"]) for f in folds]
        row = dict(N=N, draws=tried, closed=len(folds), mean_pairs=(sum(npairs) / len(npairs)) if folds else None, max_pairs=max(npairs, default=None),
                   cones_min=min(ncones, default=None), cones_max=max(ncones, default=None), swaps=dict(sw), seconds=round(time.time() - t0))
        rows.append(row)
        print(row, flush=True)
        for f in folds:
            if best is None or len(f["pairs"]) > len(best["pairs"]):
                best = f
    out = dict(rows=rows, best=dict(tiles=sorted(best["tiles"]), pairs=len(best["pairs"]), deficits=list(best["deficits"])) if best else None)
    json.dump(out, open(os.path.join(HERE, "spectre_fold_substrate.json"), "w"), indent=1)
    print("best:", out["best"]["pairs"], "pairs on", len(out["best"]["tiles"]), "tiles, deficits", out["best"]["deficits"])


if __name__ == "__main__":
    main()
