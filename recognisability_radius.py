#!/usr/bin/env python3
"""
How far must one look to know which supertile a Spectre belongs to?  (ROADMAP Stage 3, the gap of Result 2.)

In the substitution tree of the Spectre tiling (the `spectre` fork: nine labelled base tiles, Gamma being the two-tile
Mystic, and supertiles built level by level), every unit of level k has a ROLE: the pose of its level-(k+1) supertile
relative to its own frame, and that supertile's label. The role is a property of the hierarchy. This script measures
the smallest radius rho_k such that the pattern of level-k units within rho_k of a unit determines its role, in a
substitution patch:

  level 0   units = single Spectres, seen without labels (pose only); role = own base label, the pose of the level-1
            supertile and its label;
  level k   units = level-k supertiles with their labels (known from the level below); role = pose and label of the
            level-(k+1) supertile.

A radius passes when no two units of the sample with the same pattern have different roles. Only units whose disc lies
inside the patch are used, and a radius is reported only while patterns still repeat (mean multiplicity given), since
with every pattern unique the test is empty. A finite sample can miss ambiguities, so each rho_k is a LOWER estimate.

Use: if the role is determined within r at every level in units of that level (r_k <= r lambda^k, lambda^2 = 4+sqrt15),
two tilings that agree outside radius R have level-n supertiles that agree outside R_n with
R_n / lambda^n -> at most r / (lambda - 1): the disagreement is confined to a bounded number of high-level supertiles.

Usage: python3 recognisability_radius.py [--level 5] [--only 2,3] [--tests]     (level 5: 40 s; level 6 --only 1,2,3: several minutes)
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union
import shapely

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("SPECTRE_DIR", os.path.join(HERE, "..", "spectre")))
import spectre as SP  # noqa: E402

LAM = math.sqrt(4 + math.sqrt(15))


def build(level):
    """Nodes of the substitution tree by level: lists of (affine 2x3, label, parent index in the next level)."""
    tiles = SP.buildSpectreTiles(level, 1.0, 1.0)
    root = tiles["Delta"]
    depth_of = {}
    levels = defaultdict(list)                 # height above the base -> nodes
    leaves = []                                # (T, label, index of the base node)

    def height(node):                          # base tiles: Tile, or the Gamma pair (MetaTile of two Tiles)
        if isinstance(node, SP.Tile):
            return 0
        if all(isinstance(t, SP.Tile) for t in node.tiles):
            return 0
        return 1 + height(node.tiles[-1])

    def walk(node, T, label, parent):
        h = height(node)
        me = len(levels[h])
        levels[h].append((T, label, parent))
        if isinstance(node, SP.Tile):
            leaves.append((T, label, me)); return
        if h == 0:                             # Gamma: two leaves
            for t, tr in zip(node.tiles, node.transformations):
                leaves.append((SP.mul(T, tr), t.label, me))
            return
        names = SUBST[label]
        for t, tr, nm in zip(node.tiles, node.transformations, names):
            walk(t, SP.mul(T, tr), nm, me)

    walk(root, SP.IDENTITY.copy(), "Delta", -1)
    return levels, leaves


SUBST = {k: [x for x in v if x] for k, v in dict(
    Gamma=("Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi", "Gamma"), Delta=("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    Theta=("Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma"), Lambda=("Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    Xi=("Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"), Pi=("Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"),
    Sigma=("Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma"), Phi=("Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"),
    Psi=("Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma")).items()}


def rel(Tu, Tv):
    """Pose of v in the frame of u, rounded: (x, y, the four matrix entries)."""
    L = Tu[:, :2]
    Li = L.T                                    # orthogonal (rotation or reflection)
    p = Li @ (Tv[:, 2] - Tu[:, 2])
    M = Li @ Tv[:, :2]
    return (round(float(p[0]), 3) + 0.0, round(float(p[1]), 3) + 0.0) + tuple(round(float(x), 3) + 0.0 for x in M.ravel())


def study(units, roles, labels, inside_dist, radii, min_mult=3.0):
    """units: list of affine maps; roles: hashable per unit; labels: per unit or None; inside_dist: distance to the patch boundary.
    A pattern is the multiset of rounded relative poses (and labels) of the units within rho; it is compared through a
    64-bit order-independent hash (sum of mixed hashes), so a collision could only hide an ambiguity."""
    n = len(units)
    P = np.array([T[:, 2] for T in units]); L = np.array([T[:, :2] for T in units])
    rid = {r: a for a, r in enumerate(sorted(set(roles), key=repr))}
    role = np.array([rid[r] for r in roles], dtype=np.int64)
    lid = None
    if labels is not None:
        names = sorted(set(labels)); lid = np.array([names.index(x) for x in labels], dtype=np.uint64)
    tree = cKDTree(P)
    pr = tree.query_pairs(float(max(radii)) + 1e-9, output_type="ndarray")
    u = np.concatenate([pr[:, 0], pr[:, 1]]); v = np.concatenate([pr[:, 1], pr[:, 0]])
    dvec = P[v] - P[u]
    dist = np.linalg.norm(dvec, axis=1)
    Lt = np.transpose(L[u], (0, 2, 1))
    p = np.einsum("nij,nj->ni", Lt, dvec)
    M = np.einsum("nij,njk->nik", Lt, L[v]).reshape(-1, 4)
    q = np.rint(np.concatenate([p, M], axis=1) * 1000).astype(np.int64).astype(np.uint64)
    rng = np.random.default_rng(12345)
    mult = rng.integers(1, 2 ** 63, size=7, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    with np.errstate(over="ignore"):
        h = (q * mult[:6]).sum(axis=1)
        if lid is not None:
            h = h + lid[v] * mult[6]
        h ^= h >> np.uint64(29); h *= np.uint64(0xBF58476D1CE4E5B9); h ^= h >> np.uint64(32)
    rows = []
    for rho in radii:
        ok = inside_dist > rho + 1e-9
        if ok.sum() < 50:
            break
        sel = dist <= rho + 1e-9
        key = np.zeros(n, dtype=np.uint64)
        with np.errstate(over="ignore"):
            np.add.at(key, u[sel], h[sel])
            if lid is not None:
                key = key + (lid + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
        kk, rr = key[ok], role[ok]
        order = np.lexsort((rr, kk)); kk, rr = kk[order], rr[order]
        start = np.r_[True, kk[1:] != kk[:-1]]
        grp = np.cumsum(start) - 1
        ng = int(grp[-1]) + 1
        rmin = np.full(ng, np.iinfo(np.int64).max); rmax = np.full(ng, -1)
        np.minimum.at(rmin, grp, rr); np.maximum.at(rmax, grp, rr)
        amb = rmin != rmax
        rows.append(dict(rho=float(rho), units=int(ok.sum()), patterns=ng, mean_multiplicity=float(ok.sum() / ng),
                         ambiguous_patterns=int(amb.sum()), ambiguous_share=float(amb[grp].mean())))
        if ok.sum() / ng < min_mult:
            break
    good = [r for r in rows if r["mean_multiplicity"] >= min_mult]
    first = None
    for i, r in enumerate(good):
        if all(x["ambiguous_patterns"] == 0 for x in good[i:]):
            first = r["rho"]; break
    return rows, first


def main():
    level = int(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else 5
    levels, leaves = build(level)
    pts = SP.SPECTRE_POINTS
    polys = [Polygon((T[:, :2] @ pts.T).T + T[:, 2]) for T, _, _ in leaves]
    area = polys[0].area
    groups = defaultdict(list)                 # union in two stages (per level-2 supertile, then all): much less memory
    for p, (T, lab, b) in zip(polys, leaves):
        g = levels[1][levels[0][b][2]][2] if level >= 2 else 0
        groups[g].append(p.buffer(1e-6))
    U = unary_union([unary_union(v) for v in groups.values()])
    assert abs(U.area - area * len(polys)) < 1e-2 * len(polys), "tiles overlap or leave gaps"
    boundary = U.exterior
    out = dict(level=level, tiles=len(leaves), tile_area=area, lam=LAM, results=[])

    def dist(units):
        return np.asarray(shapely.distance(shapely.points(np.array([T[:, 2] for T in units])), boundary))

    only = [int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")] if "--only" in sys.argv else None
    # level 0: single Spectres without labels
    T0 = [T for T, _, _ in leaves]
    base = levels[0]
    roles0 = []
    for T, lab, b in leaves:
        Tb, labb, par = base[b]
        Tp, labp, _ = levels[1][par]
        roles0.append((lab, rel(T, Tp), labp))
    rows, r0 = study(T0, roles0, None, dist(T0), np.arange(1.0, 20.01, 0.5)) if (only is None or 0 in only) else ([], None)
    out["results"].append(dict(k=0, rows=rows, rho=r0, rho_in_level_units=r0, n_roles=len(set(roles0))))
    print("level 0: %d tiles, %d roles; role determined from radius %s" % (len(T0), len(set(roles0)), r0))
    for r in rows:
        print("   rho %5.1f  units %6d  patterns %6d  mult %7.1f  ambiguous patterns %5d  share %.4f" % (r["rho"], r["units"], r["patterns"], r["mean_multiplicity"], r["ambiguous_patterns"], r["ambiguous_share"]))
    # level k >= 1: supertiles with labels
    for k in range(1, level - 1):
        if only is not None and k not in only:
            continue
        units = [T for T, _, _ in levels[k]]
        labs = [lab for _, lab, _ in levels[k]]
        roles = [(rel(T, levels[k + 1][par][0]), levels[k + 1][par][1]) for T, lab, par in levels[k]]
        s = LAM ** k
        rows, rk = study(units, roles, labs, dist(units), np.arange(1.0, 20.01, 0.5) * s)
        out["results"].append(dict(k=k, rows=rows, rho=rk, rho_in_level_units=(rk / s if rk else None), n_roles=len(set(roles))))
        print("level %d: %d units, %d roles; role determined from radius %s = %s lambda^%d" % (k, len(units), len(set(roles)), rk, None if rk is None else round(rk / s, 2), k))
        for r in rows:
            print("   rho %6.1f (%4.1f)  units %6d  patterns %6d  mult %7.1f  ambiguous patterns %5d  share %.4f" % (r["rho"], r["rho"] / s, r["units"], r["patterns"], r["mean_multiplicity"], r["ambiguous_patterns"], r["ambiguous_share"]))
    # confinement: R_{k+1} = R_k + r lam^k + D lam^{k+1}  =>  R_n / lam^n -> (r + D lam) / (lam - 1)
    found = [x["rho_in_level_units"] for x in out["results"] if x["rho_in_level_units"]]
    if found:
        r = max(found)
        D = max(max(np.linalg.norm(T[:, 2] - levels[k + 1][p][0][:, 2]) for T, _, p in levels[k] if p >= 0) / LAM ** (k + 1) for k in range(level - 1))
        rho_star = (r + D * LAM) / (LAM - 1)
        unit_area = area * len(leaves) / len(levels[0])
        out["confinement"] = dict(r=r, D=float(D), rho_star=float(rho_star), base_unit_area=float(unit_area), supertiles_within=float(math.pi * rho_star ** 2 / unit_area))
        print("confinement: r = %.1f, D = %.2f, rho* = %.2f level units, about %.0f level-n supertiles" % (r, D, rho_star, out["confinement"]["supertiles_within"]))
    json.dump(out, open(os.path.join(HERE, "recognisability_radius_L%d.json" % level), "w"), indent=1)
    if "--tests" in sys.argv:
        assert only is not None or out["results"][0]["rho"] is not None
        print("recognisability_radius: tests passed (tiles cover the patch once; a level-0 radius exists in the sample)")


if __name__ == "__main__":
    main()
