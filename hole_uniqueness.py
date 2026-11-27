#!/usr/bin/env python3
"""
Does every bounded hole in a Spectre tiling have only one filling?  (ROADMAP Stage 9o; the finite check of Stage 9n.)

Stage 9n confines the difference of two Spectre tilings that agree outside a bounded set to the units within
rho* = 11.0 (in units of the level) of a point. This script takes the level-0 form of the remaining statement and
tests it for every local configuration of a substitution patch:

  hole      all tiles whose origin lies within RHO of the origin of a centre tile. RHO = rho* + c + m, where c is the
            covering radius of the tile origins (every point is within c of an origin) and m = 2.7 the largest offset
            of a tile from its base unit; a hole that contains a smaller hole has a unique filling only if the smaller
            one has, so these centred holes cover all holes of radius rho* about arbitrary points;
  classes   two centres with the same pattern of tiles within RHO have congruent holes; one hole per pattern is solved;
  solver    complete depth-first search. The unfilled region has a convex corner v (its extreme point in a generic
            direction); the tile that covers the region next to the boundary side leaving v has a corner at v and an
            edge along that side, so there are at most 14 placements (one per edge of the Spectre, rotations only,
            one hand). Each placement that lies inside the region is tried. The search stops at the second solution.

The geometry is in floating point (shapely): containment means that the part of the tile outside the region has area
below 1e-7, and slivers below 1e-6 are dropped. A node limit guards the run time; a class that hits it is reported as
undecided, never as unique.

Usage: python3 hole_uniqueness.py [--level 5] [--rho 17.0] [--jobs 2] [--limit N] [--tests]          (level 5: about 6 minutes)
       python3 hole_uniqueness.py --level 6 --max-centres 60000 --classes-only hole_uniqueness_L5.json     (saturation of the class list)
"""
import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recognisability_radius as RR  # noqa: E402
SP = RR.SP

GEN = np.array([math.cos(0.3137), math.sin(0.3137)])
NODE_LIMIT = 200000


def tile_shape():
    P = np.array(SP.get_spectre_points(1.0, 1.0))
    return P


def ccw(P):
    x, y = P[:, 0], P[:, 1]
    return P if (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() > 0 else P[::-1]


def clean(R):
    if R.is_empty:
        return R
    R = R.buffer(-1e-7, join_style=2).buffer(1e-7, join_style=2)       # mitred: corners stay corners
    parts = [g for g in getattr(R, "geoms", [R]) if g.area > 1e-6]
    return unary_union(parts) if parts else Polygon()


def solve(hole, S, limit=2, node_limit=NODE_LIMIT):
    """All fillings of the polygon `hole` by rotated copies of the CCW polygon S (list of placements); stops at `limit`."""
    edges = np.roll(S, -1, axis=0) - S
    eang = np.arctan2(edges[:, 1], edges[:, 0])
    sols, nodes = [], [0]

    def rec(R, placed):
        if len(sols) >= limit or nodes[0] > node_limit:
            return
        nodes[0] += 1
        if R.is_empty or R.area < 1e-6:
            sols.append(sorted(placed)); return
        best = None
        for g in getattr(R, "geoms", [R]):
            C = np.asarray(g.exterior.coords)[:-1]
            if g.exterior.is_ccw is False:
                C = C[::-1]
            a = int(np.argmin(C @ GEN))
            if best is None or C[a] @ GEN < best[0]:
                b = (a + 1) % len(C)
                while np.abs(C[b] - C[a]).sum() < 1e-3:      # skip near-duplicate vertices left by the clean-up
                    b = (b + 1) % len(C)
                best = (C[a] @ GEN, C[a], C[b])
        _, v, w = best
        d = math.radians(30 * round(math.degrees(math.atan2(w[1] - v[1], w[0] - v[0])) / 30))     # all sides are multiples of 30 degrees
        for i in range(len(S)):
            th = d - eang[i]
            c, s = math.cos(th), math.sin(th)
            Rm = np.array([[c, -s], [s, c]])
            Q = (S - S[i]) @ Rm.T + v
            T = Polygon(Q)
            if T.difference(R).area > 1e-6:
                continue
            key = (round(float(Q[0, 0]), 4), round(float(Q[0, 1]), 4), int(round(math.degrees(th))) % 360)
            rec(clean(R.difference(T)), placed + [key])
            if len(sols) >= limit or nodes[0] > node_limit:
                return

    sys.setrecursionlimit(10000)
    rec(hole, [])
    return sols, nodes[0]


_G = {}


def work(job):
    cls, centre, idx, count = job
    polys = [_G["polys"][a] for a in idx]
    hole = clean(unary_union([p.buffer(1e-8, join_style=2) for p in polys]))
    t = time.time()
    sols, nodes = solve(hole, _G["S"])
    return dict(cls=cls, centre=int(centre), tiles=len(idx), occurrences=int(count), fillings=len(sols), nodes=nodes,
                undecided=bool(nodes > NODE_LIMIT), tiles_in_filling=[len(s) for s in sols], seconds=round(time.time() - t, 2))


def patterns(leaves, rho, centres):
    """Hash of the pattern of tiles within rho of each centre (as in recognisability_radius.study), and the members."""
    P = np.array([T[:, 2] for T, _, _ in leaves]); L = np.array([T[:, :2] for T, _, _ in leaves])
    tree = cKDTree(P)
    nb = tree.query_ball_point(P[centres], rho + 1e-9)
    rng = np.random.default_rng(12345)
    mult = rng.integers(1, 2 ** 63, size=6, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    keys = np.zeros(len(centres), dtype=np.uint64)
    for a, (u, lst) in enumerate(zip(centres, nb)):
        v = np.array(lst)
        Lt = L[u].T
        p = (P[v] - P[u]) @ Lt.T
        M = np.einsum("ij,njk->nik", Lt, L[v]).reshape(-1, 4)
        q = np.rint(np.concatenate([p, M], axis=1) * 1000).astype(np.int64).astype(np.uint64)
        with np.errstate(over="ignore"):
            h = (q * mult).sum(axis=1)
            h ^= h >> np.uint64(29); h *= np.uint64(0xBF58476D1CE4E5B9); h ^= h >> np.uint64(32)
            keys[a] = h.sum()
    return keys, nb, tree, P


def main():
    arg = lambda k, d, f=float: f(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    level, rho, jobs, lim = arg("--level", 5, int), arg("--rho", None), arg("--jobs", 4, int), arg("--limit", None, int)
    levels, leaves = RR.build(level)
    pts = np.array(SP.get_spectre_points(1.0, 1.0))
    polys = [Polygon((T[:, :2] @ pts.T).T + T[:, 2]) for T, _, _ in leaves]
    S = ccw(np.asarray(polys[0].exterior.coords)[:-1])     # the hand of the patch's own tiles (it alternates with the level)
    groups = defaultdict(list)
    for p, (T, lab, b) in zip(polys, leaves):
        groups[levels[1][levels[0][b][2]][2]].append(p.buffer(1e-6))
    U = unary_union([unary_union(v) for v in groups.values()])
    P = np.array([T[:, 2] for T, _, _ in leaves])
    inside = np.asarray(shapely.distance(shapely.points(P), U.exterior))
    # covering radius of the origins, from a grid of points well inside the patch
    tree = cKDTree(P)
    cen = P[np.argmax(inside)]
    g = np.linspace(-20, 20, 401)
    X = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) + cen
    cov = float(tree.query(X)[0].max())
    if rho is None:
        rho = 11.0 + cov + 2.7
    centres = np.where(inside > rho + 6.0)[0]
    mc = arg("--max-centres", None, int)
    if mc and len(centres) > mc:
        centres = np.sort(np.random.default_rng(7).choice(centres, mc, replace=False))
    keys, nb, _, _ = patterns(leaves, rho, centres)
    cls = defaultdict(list)
    for a, k in enumerate(keys):
        cls[int(k)].append(a)
    print("level %d: %d tiles; covering radius of origins %.2f; RHO = %.2f; %d centres inside; %d hole classes; tiles per hole %d to %d" % (
        level, len(leaves), cov, rho, len(centres), len(cls), min(len(x) for x in nb), max(len(x) for x in nb)), flush=True)
    if "--classes-only" in sys.argv:             # for the saturation check: do the classes of a bigger patch occur here?
        ref = sys.argv[sys.argv.index("--classes-only") + 1]
        known = set(json.load(open(ref))["class_keys"])
        new = [k for k in cls if k not in known]
        cnt = sum(len(cls[k]) for k in new)
        print("classes here %d; of these not in %s: %d (%d of %d centres)" % (len(cls), os.path.basename(ref), len(new), cnt, len(centres)))
        json.dump(dict(level=level, rho=rho, centres=int(len(centres)), classes=len(cls), not_in_reference=len(new), centres_not_in_reference=cnt,
                       class_keys=sorted(cls)), open(os.path.join(HERE, "hole_classes_L%d.json" % level), "w"))
        return
    work_list = [(k, centres[v[0]], list(nb[v[0]]), len(v)) for k, v in sorted(cls.items(), key=lambda kv: -len(kv[1]))]
    if lim:
        work_list = work_list[:lim]
    _G["polys"], _G["S"] = polys, S
    t0 = time.time()
    res = []
    with Pool(jobs) as pool:
        for r in pool.imap_unordered(work, work_list):
            res.append(r)
            if len(res) % 20 == 0 or r["fillings"] != 1:
                print("  %d/%d classes done (%.0f s); last: %d tiles, %d fillings, %d nodes%s" % (
                    len(res), len(work_list), time.time() - t0, r["tiles"], r["fillings"], r["nodes"], "  UNDECIDED" if r["undecided"] else ""), flush=True)
    uniq = sum(r["fillings"] == 1 and not r["undecided"] for r in res)
    out = dict(level=level, tiles=len(leaves), covering_radius=cov, rho=rho, centres=int(len(centres)), classes=len(cls), solved=len(res),
               unique=uniq, two_or_more=sum(r["fillings"] >= 2 for r in res), none_found=sum(r["fillings"] == 0 for r in res),
               undecided=sum(r["undecided"] for r in res), max_nodes=max(r["nodes"] for r in res), mean_nodes=float(np.mean([r["nodes"] for r in res])),
               tiles_per_hole=[min(r["tiles"] for r in res), max(r["tiles"] for r in res)], class_keys=sorted(cls), results=res)
    json.dump(out, open(os.path.join(HERE, "hole_uniqueness_L%d.json" % level), "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("results", "class_keys")})
    if "--tests" in sys.argv:
        assert out["none_found"] == 0, "the solver must at least find the original filling"
        assert all(r["tiles_in_filling"][0] == r["tiles"] for r in res if r["fillings"])
        print("hole_uniqueness: tests passed (every hole solved has its original filling; counts consistent)")


if __name__ == "__main__":
    main()
