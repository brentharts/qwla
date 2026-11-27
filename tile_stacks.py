#!/usr/bin/env python3
"""
Stacks of same-orientation Spectres, and what they say about the direction of a ribbon.  (ROADMAP Stage 9r.)

Origin: B. Hartshorn's `build_shapes` in spectre/spectre_tiles_blender.py groups tiles that have the same rotation
and sit side by side at (nearly) the same height, and colours the groups. This is a port to numpy and matplotlib,
with the grouping made exact:

  stack     a maximal chain of Spectres of one orientation in which each is the translate of the previous one by the
            vector w that has coordinates ((3 - sqrt3)/2, (3 + sqrt3)/2) in the tile's own frame, |w| = sqrt6. Two such
            tiles touch: this is the side-by-side pair of equal rotation.
  axis      the direction of w in the plane. The six orientations of the (non-Mystic-second) tiles give three axes,
            60 degrees apart.
  mismatch  build_shapes matched heights after rounding and recorded the difference (`tile_match_error`). Here: for
            same-orientation tiles less than 30 apart along a stack axis, the distinct small offsets across the axis.

Part 1 (substitution patch): how many tiles are in stacks, stack lengths, the axes, the mismatch values.
Part 2 (ribbons; needs dumps from generic_phase.py DUMP=...): the share of each ribbon's tiles that are in stacks made
of ribbon tiles, and the angle between the ribbon and the nearest stack axis.

Usage: python3 tile_stacks.py [--level 4] [dump.json ...] [--figure] [--tests]
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recognisability_radius as RR  # noqa: E402
SP = RR.SP
S3 = math.sqrt(3.0)
W = np.array([(3 - S3) / 2, (3 + S3) / 2])
REF = np.array(SP.get_spectre_points(1.0, 1.0))


def stacks(P, L, members=None):
    """chain[u] = v when v = u + w (same orientation). Returns chain, list of stacks (each a list of tiles), axis per stack."""
    tree = cKDTree(P)
    ms = None if members is None else set(int(x) for x in members)
    chain = {}
    for u in (range(len(P)) if ms is None else sorted(ms)):
        for v in tree.query_ball_point(P[u], 2.6):
            if v != u and (ms is None or v in ms) and np.abs(L[v] - L[u]).max() < 1e-6 and np.abs(L[u].T @ (P[v] - P[u]) - W).sum() < 1e-3:
                chain[u] = v
    tails = set(chain.values())
    out = []
    for s in chain:
        if s in tails:
            continue
        st = [s]
        while st[-1] in chain:
            st.append(chain[st[-1]])
        d = P[st[1]] - P[st[0]]
        out.append((st, round(math.degrees(math.atan2(d[1], d[0])) % 180, 1)))
    return out


def part1(level):
    levels, leaves = RR.build(level)
    P = np.array([T[:, 2] for T, _, _ in leaves]); L = np.array([T[:, :2] for T, _, _ in leaves])
    st = stacks(P, L)
    n_in = sum(len(s) for s, _ in st)
    lengths = Counter(len(s) for s, _ in st)
    axes = Counter(a for _, a in st)
    labels = Counter(leaves[u][1] for s, _ in st for u in s)
    all_labels = Counter(lab for _, lab, _ in leaves)
    # mismatch across the axis between same-orientation tiles (build_shapes' tile_match_error)
    tree = cKDTree(P)
    mm = Counter()
    wdir = W / np.linalg.norm(W); wperp = np.array([-wdir[1], wdir[0]])
    for u in range(0, len(P), max(1, len(P) // 4000)):
        for v in tree.query_ball_point(P[u], 30.0):
            if v != u and np.abs(L[v] - L[u]).max() < 1e-6:
                d = L[u].T @ (P[v] - P[u])
                if abs(d @ wperp) < 0.5:
                    mm[round(abs(float(d @ wperp)), 4)] += 1
    return dict(level=level, tiles=len(leaves), in_stacks=n_in, share=n_in / len(leaves), lengths={str(k): v for k, v in sorted(lengths.items())},
                axes={str(k): v for k, v in sorted(axes.items())},
                label_share_in_stacks={k: round(labels.get(k, 0) / all_labels[k], 3) for k in sorted(all_labels)},
                mismatch_values={str(k): v for k, v in sorted(mm.items())}), (P, L, st)


def literal(level, tol=0.05):
    """build_shapes as written: same rotation and the same height after rounding (here |dy| < tol edge lengths), any distance."""
    levels, leaves = RR.build(level)
    P = np.array([T[:, 2] for T, _, _ in leaves]); L = np.array([T[:, :2] for T, _, _ in leaves])
    ang = np.round(np.degrees(np.arctan2(L[:, 1, 0], L[:, 0, 0]))).astype(int) % 360
    n_with, errs, dist = 0, Counter(), Counter()
    idx = np.arange(len(P))
    for u in idx:
        m = idx[(ang == ang[u]) & (np.abs(P[:, 1] - P[u, 1]) < tol) & (idx != u)]
        if len(m):
            n_with += 1
            j = m[np.argmin(np.abs(P[m, 0] - P[u, 0]))]
            dist[round(float(abs(P[j, 0] - P[u, 0])), 2)] += 1
            for v in m:
                errs[round(float(abs(P[v, 1] - P[u, 1])), 4)] += 1
    return dict(level=level, tolerance=tol, tiles=len(P), tiles_with_partner=n_with, height_errors={str(k): v for k, v in sorted(errs.items())},
                nearest_partner_distance={str(k): v for k, v in sorted(dist.items())})


def part2(files):
    import ribbon_hierarchy as RH
    import hole_uniqueness as HU
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    rows, keep = [], None
    for f in files:
        dump = json.load(open(f))[0]
        S = HU.ccw(REF * np.array([-1.0, 1.0]))
        tiles, owner = [], []
        for a, pl in enumerate(dump["tiles"]):
            for Q in RH.split(pl, S):
                tiles.append(RH.pose_of(Q)); owner.append(a)
        P = np.array([T[:, 2] for T in tiles]); L = np.array([T[:, :2] for T in tiles])
        polys = [Polygon(REF @ T[:, :2].T + T[:, 2]) for T in tiles]
        cent = np.array([[p.centroid.x, p.centroid.y] for p in polys])
        inner = np.where(np.linalg.norm(P, axis=1) < dump["R"] - 12)[0]
        base = stacks(P, L)
        in_base = {u for s, _ in base for u in s}
        base_share = len(in_base & set(inner.tolist())) / len(inner)
        axes = sorted({a for _, a in base})
        tree = STRtree(polys)
        pr = tree.query(polys, predicate="dwithin", distance=1e-6)
        nb = defaultdict(list)
        for a, b in zip(*pr):
            if a != b:
                nb[int(a)].append(int(b))
        sp = defaultdict(list)
        for u, a in enumerate(owner):
            sp[a].append(u)
        seen = set()
        for st in dump["steps"]:
            rem = sorted({u for a in st["removed"] for u in sp[a]}); rs = set(rem); comp = {}
            for u in rem:
                if u in comp:
                    continue
                stack = [u]; comp[u] = u
                while stack:
                    x = stack.pop()
                    for y in nb[x]:
                        if y in rs and y not in comp:
                            comp[y] = u; stack.append(y)
            groups = defaultdict(list)
            for u, c in comp.items():
                groups[c].append(u)
            for g in groups.values():
                if len(g) < 20:
                    continue
                g = sorted(g); key = (len(g), tuple(g[:8]))
                if key in seen:
                    continue
                seen.add(key)
                X = cent[g] - cent[g].mean(0)
                w, vv = np.linalg.eigh(X.T @ X)
                ang = math.degrees(math.atan2(vv[1, 1], vv[0, 1])) % 180
                rst = stacks(P, L, g)
                n_in = sum(len(s) for s, _ in rst)
                dev = min(min(abs(ang - a), 180 - abs(ang - a)) for a in axes)
                rows.append(dict(file=os.path.basename(f), tiles=len(g), angle=ang, in_stacks=n_in, share=n_in / len(g),
                                 lengths={str(k): v for k, v in sorted(Counter(len(s) for s, _ in rst).items())},
                                 stacks_per_axis={str(k): v for k, v in sorted(Counter(a for _, a in rst).items())}, deviation_from_nearest_axis=dev))
                if keep is None or (rows[-1]["share"] > 0.6 and len(g) > keep[0]):
                    keep = (len(g), polys, g, rst, cent)
        print("%s: %d Spectres; share of tiles in stacks %.3f; axes %s" % (os.path.basename(f), len(tiles), base_share, axes))
        rows.append(dict(file=os.path.basename(f), baseline_share=base_share, axes=axes))
    return rows, keep


def figure(P, L, st, keep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titleweight": "regular", "axes.titlelocation": "left", "savefig.dpi": 300})
    cols = ["#1f77b4", "#d95f02", "#1b9e77"]
    fig, axs = plt.subplots(1, 2 if keep else 1, figsize=(7.2, 3.6))
    axs = np.atleast_1d(axs)
    axes_sorted = sorted({a for _, a in st})
    col_of = {}
    for s, a in st:
        for u in s:
            col_of[u] = cols[axes_sorted.index(a) % 3]
    polys = [REF @ L[u].T + P[u] for u in range(len(P))]
    c0 = P.mean(0)
    sel = [u for u in range(len(P)) if np.abs(P[u] - c0).max() < 22]
    axs[0].add_collection(PolyCollection([polys[u] for u in sel], facecolors=[col_of.get(u, "#e6e6e6") for u in sel], edgecolors="white", linewidths=0.3))
    axs[0].set_xlim(c0[0] - 22, c0[0] + 22); axs[0].set_ylim(c0[1] - 22, c0[1] + 22)
    axs[0].set_title("a  Stacks of equal-rotation tiles, by axis")
    if keep:
        _, pl, g, rst, cent = keep
        gs = set(g)
        ax_r = sorted({a for _, a in rst})
        rc = {u: cols[ax_r.index(a) % 3] for s, a in rst for u in s}
        c = cent[g].mean(0); half = np.abs(cent[g] - c).max() + 6
        near = [u for u in range(len(pl)) if np.abs(cent[u] - c).max() < half]
        axs[1].add_collection(PolyCollection([np.asarray(pl[u].exterior.coords) for u in near],
                                             facecolors=[rc.get(u, "#9e9e9e") if u in gs else "#f0f0f0" for u in near], edgecolors="white", linewidths=0.2))
        axs[1].set_xlim(c[0] - half, c[0] + half); axs[1].set_ylim(c[1] - half, c[1] + half)
        axs[1].set_title("b  Crossing ribbons: stacked tiles by axis, unstacked grey")
    for a in axs:
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s_ in a.spines.values():
            s_.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    level = int(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else 4
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    p1, (P, L, st) = part1(level)
    print("substitution patch level %d: %d tiles, %d in stacks (%.3f); stack lengths %s; axes %s" % (level, p1["tiles"], p1["in_stacks"], p1["share"], p1["lengths"], p1["axes"]))
    print("  share of each label in stacks:", p1["label_share_in_stacks"])
    print("  offsets across the axis between same-orientation tiles within 30 (value: count):", p1["mismatch_values"])
    lit = literal(level)
    print("  as in build_shapes (same rotation, heights equal within %.2f): %d of %d tiles have a partner; height errors %s; distance to the nearest partner %s" % (
        lit["tolerance"], lit["tiles_with_partner"], lit["tiles"], lit["height_errors"], lit["nearest_partner_distance"]))
    out = dict(patch=p1, build_shapes_literal=lit)
    keep = None
    if files:
        rows, keep = part2(files)
        ribs = [r for r in rows if "tiles" in r]
        for r in ribs:
            print("  ribbon %3d tiles, angle %5.1f: in stacks %.2f, lengths %s, per axis %s, off nearest axis %.1f" % (r["tiles"], r["angle"], r["share"], r["lengths"], r["stacks_per_axis"], r["deviation_from_nearest_axis"]))
        with_s = [r for r in ribs if r["in_stacks"] > 0]; without = [r for r in ribs if r["in_stacks"] == 0]
        dv = [r["deviation_from_nearest_axis"] for r in with_s]; dn = [r["deviation_from_nearest_axis"] for r in without]
        out["ribbons"] = rows
        out["summary"] = dict(ribbons=len(ribs), with_stacks=len(with_s), without_stacks=len(without),
                              deviation_with_stacks=dict(mean=float(np.mean(dv)), max=float(np.max(dv))) if dv else None,
                              deviation_without_stacks=dict(mean=float(np.mean(dn)), max=float(np.max(dn))) if dn else None,
                              chance_all_below_max=float((max(dv) / 30) ** len(dv)) if dv else None)
        print(out["summary"])
    if "--figure" in sys.argv:
        os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
        figure(P, L, st, keep, os.path.join(HERE, "figures", "tile_stacks.png"))
    json.dump(out, open(os.path.join(HERE, "tile_stacks.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        assert len(p1["axes"]) == 3 and set(p1["lengths"]) <= {"2", "3"}
        print("tile_stacks: tests passed (three stack axes; stacks have 2 or 3 tiles)")


if __name__ == "__main__":
    main()
