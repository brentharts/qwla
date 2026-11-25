#!/usr/bin/env python3
"""
Phason flips of a whole substitution patch (generation 3: 559 tiles, 82 pairs; generation 4: 4401 tiles, 804 pairs).

The wada-coudene flip engine (`figures/spectre_flip_moves.py`) moves the phase z of ONE probe tile and looks at the
tiles around it. Here the phase is the one of the patch itself: tile i of class g has its own window point z_i
(the point that the substitution recursion puts in the window of its class frame), the patch is the tiling at the
phase where every tile is present, and a phason step y -> y + delta e moves every z_i by delta R_g^T R_0 e. A tile
leaves the patch when its moved point leaves the window (nearest point of the depth-7 cloud farther than OUT_T);
a tile enters when it overlaps a removed tile (the plane stays covered) and its own window contains its moved point.
The placements that overlap a tile of class a are the engine's placement list with a small radius.

Checks (all counted): every tile of the patch is inside its window at phase 0; for every cluster of changed tiles
(removed and added tiles linked by overlap) the footprints of the removed and the added tiles are equal and the
added tiles do not overlap the persisting ones, which is what makes the cluster a flip of a tiling; clusters with a
tile whose membership is ambiguous (distance between IN_T and OUT_T) or that come within MARGIN of the patch
boundary are dropped and counted.

Result (see ROADMAP_QUANTUM_GRAVITY.md, Stage 3): the set-up checks pass at N = 3, but the substitution patch sits at a
non-generic phase: no tile leaves its window in 22 of 24 directions at delta <= 0.03, and in the others the changes are
not local, so this script produced no closed cluster there. It is kept as the tool for a generic phase.
Output: patch_flips_g{N}_d{DELTA}.json   one record per closed, unambiguous cluster:
  tile counts removed/added (windows and leaves), (dV, dE, dF) of the cell complex, the pairs destroyed and created
  (by flat-vertex key) and the pair count of the patch before.
Usage: SPECTRE_DIR=.. WADA_DIR=.. python3 patch_flips.py N [n_directions] [DELTA]
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WADA = os.environ.get("WADA_DIR", "/home/claude/repos/wada-coudene")
SP = os.environ.get("SPECTRE_DIR", "/home/claude/repos/spectre")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NDIR = int(sys.argv[2]) if len(sys.argv) > 2 else 8
DELTAS = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.03").split(",")]
DELTA = DELTAS[0]
R2 = float(os.environ.get("R2", "0.5"))
MARGIN = float(os.environ.get("MARGIN", "8.0"))
IN_T, OUT_T = 0.0015, 0.0035

t00 = time.time()
src = open(os.path.join(WADA, "figures", "spectre_flip_moves.py")).read()
lines = src.split("\n")
head = "\n".join(lines[:68])                                   # up to the end of `shape` (line 68)
middle = "\n".join(lines[68:105])                              # PH ... cands (lines 69-105)
sys.argv = ["spectre_flip_moves.py", str(N), "0"]
os.environ["SPECTRE_DIR"] = SP
os.environ.setdefault("PBOX", "12")
sys.path.insert(0, WADA)
env = {"__file__": os.path.join(WADA, "figures", "spectre_flip_moves.py"), "__name__": "engine_setup"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, "engine_head", "exec"), env)
    env["RAD"] = env["RAD"]
    exec(compile(middle, "engine_middle", "exec"), env)
print("engine set-up %.0fs; classes %d" % (time.time() - t00, len(env["FR"])), flush=True)

FR, VEC, groups, keys, cands = env["FR"], env["VEC"], env["groups"], env["keys"], env["cands"]
STAR, scale, RAD, shape, in_lattice, trees = env["STAR"], env["scale"], env["RAD"], env["shape"], env["in_lattice"], env["trees"]
SPECTRE, mul, apply, ttrans, trot, IDENT, to_c = (env[k] for k in ("SPECTRE", "mul", "apply", "ttrans", "trot", "IDENT", "to_c"))
LEAVES = env["LEAVES"]
Polygon, Point, LineString, unary_union = env["Polygon"], env["Point"], None, env["unary_union"]
from shapely.geometry import LineString                                       # noqa: E402
from shapely.strtree import STRtree                                           # noqa: E402

sys.path.insert(0, HERE)
from flip_rule import find_pairs_keys                                         # noqa: E402


# ---- the patch ----------------------------------------------------------------------------------
tiles = []                                                       # one entry per window tile of the patch
for g, key in enumerate(keys):
    A_, Y_, Z_ = groups[key]
    for i, (u, z) in enumerate(zip(VEC[key], Z_)):
        tiles.append(dict(g=g, u=tuple(int(x) for x in u), z=np.array([float(z[0]), float(z[1])])))
print("patch window tiles", len(tiles), flush=True)
S_of = [f[0] for f in FR]
R0 = FR[0][2]
tile_set = {(t["g"], t["u"]): k for k, t in enumerate(tiles)}


def poly_of(g, u):
    return shape(FR[g][0], FR[g][1], np.array([int(x) for x in u], dtype=object))


def leaf_keys(g, u):
    out = []
    for L in LEAVES[FR[g][0]]:
        T = mul((FR[g][1], np.array([int(x) for x in u], dtype=object)), L)
        out.append([tuple(int(x) for x in apply(T, v)) for v in SPECTRE])
    return out


polys = [poly_of(t["g"], t["u"]) for t in tiles]
patch_union = unary_union(polys)
bd = patch_union.exterior if patch_union.geom_type == "Polygon" else patch_union.boundary
print("patch area %.1f, radius-equivalent %.1f, %.0fs" % (patch_union.area, (patch_union.area / np.pi) ** 0.5, time.time() - t00), flush=True)
# every tile of the patch is inside its own window at phase 0
d0 = np.array([trees[S_of[t["g"]]].query(t["z"])[0] for t in tiles])
print("phase 0: tiles with nearest cloud point farther than IN_T: %d of %d (max distance %.4f)" % ((d0 > IN_T).sum(), len(tiles), d0.max()), flush=True)

# ---- placements overlapping a tile of class a ------------------------------------------------------
def build_PL(a):
    S1, A1, R1, c1, _, _ = FR[a]
    u_a = np.array(VEC[keys[a]][0], dtype=np.int64)
    P1 = shape(S1, A1, np.zeros(4, dtype=object))
    PL = []
    for b in range(len(FR)):
        Sb, Ab, Rb, cb, _, _ = FR[b]
        v_b = np.array(VEC[keys[b]][0], dtype=np.int64)
        W = cands[in_lattice(cands - (v_b - u_a))]
        dd = ((c1 - cb)[None, :] + W @ STAR.T) @ Rb / scale
        keep = np.linalg.norm(dd, axis=1) <= 2 * RAD + 0.6
        for w, d in zip(W[keep], dd[keep]):
            poly = shape(Sb, Ab, np.array([int(x) for x in w], dtype=object))
            if poly.distance(P1) < R2:
                PL.append((b, Rb.T @ R1, d, tuple(int(x) for x in w), poly))
    return PL


PLs = {}
for a in range(len(FR)):
    PLs[a] = build_PL(a)
print("placements overlapping a tile, per class:", [len(PLs[a]) for a in range(len(FR))], "%.0fs" % (time.time() - t00), flush=True)

# placements of a class-a tile must contain the tiles that the patch has next to it: check on the patch
miss = 0
for k in range(0, len(tiles), max(1, len(tiles) // 60)):
    t = tiles[k]
    if patch_union.exterior.distance(polys[k]) < MARGIN:
        continue
    rel = {(b, tuple(w)) for (b, B, d, w, poly) in PLs[t["g"]]}
    for k2, p2 in enumerate(polys):
        if k2 != k and polys[k].intersection(p2).area > 1e-6:
            if (tiles[k2]["g"], tuple(np.array(tiles[k2]["u"]) - np.array(t["u"]))) not in rel:
                miss += 1
print("check: overlapping patch neighbours missing from the placement lists: %d" % miss, flush=True)


# ---- cell complex counts (copy of the engine's complex_counts) ------------------------------------
def complex_counts(cells, Fb):
    Rd = lambda p: (round(p[0], 6), round(p[1], 6))
    rings = [[Rd(p) for p in c.exterior.coords[:-1]] for c in cells]
    V = set(v for r in rings for v in r)
    E = set()
    for r in rings:
        n = len(r)
        for i in range(n):
            a_, b_ = r[i], r[(i + 1) % n]
            ls = LineString([a_, b_])
            pts = [a_, b_] + [v for v in V if v != a_ and v != b_ and ls.distance(Point(v)) < 1e-6]
            pts.sort(key=lambda v: (v[0] - a_[0]) ** 2 + (v[1] - a_[1]) ** 2)
            for k in range(len(pts) - 1):
                E.add(frozenset((pts[k], pts[k + 1])))
    b = Fb.boundary
    Vb = set(v for v in V if b.distance(Point(v)) < 1e-6)
    Eb = set(e for e in E if all(b.distance(Point(v)) < 1e-6 for v in e) and b.distance(LineString(list(e)).interpolate(0.5, normalized=True)) < 1e-6)
    return len(V), len(E), len(cells), len(Vb), len(Eb)


# ---- the old pair set -----------------------------------------------------------------------------
leafs_old = {k: leaf_keys(t["g"], t["u"]) for k, t in enumerate(tiles)}


def pairs_of(tile_ids, extra=None):
    """Pairs (by flat-vertex key) of a tile set: dict flat key -> frozenset of the two leaf identities."""
    lst, ident = [], []
    for k in tile_ids:
        lv = leafs_old[k] if k in leafs_old else extra[k]
        for li, kk in enumerate(lv):
            lst.append(kk)
            ident.append((k, li))
    prs = find_pairs_keys(lst)
    return {tuple(lst[i][10]): frozenset((ident[i], ident[j])) for i, j in prs}


pairs_old_all = pairs_of(range(len(tiles)))
print("pairs of the patch: %d (expected 82 at N=3, 804 at N=4)" % len(pairs_old_all), flush=True)
flat_pos = lambda key: (lambda c: (c.real, c.imag))(to_c(np.array(key, dtype=object)))
inner_pairs_old = {k for k in pairs_old_all if bd.distance(Point(*flat_pos(k))) > MARGIN}
print("pairs farther than %.1f from the boundary: %d" % (MARGIN, len(inner_pairs_old)), flush=True)


def flip_step(phi):
    e = np.array([np.cos(phi), np.sin(phi)])
    gone, amb_gone = [], []
    shifts = {g: DELTA * (FR[g][2].T @ R0) @ e for g in range(len(FR))}
    znew = {}
    for k, t in enumerate(tiles):
        zn = t["z"] + shifts[t["g"]]
        znew[k] = zn
        d = trees[S_of[t["g"]]].query(zn)[0]
        if d > OUT_T:
            gone.append(k)
        elif d >= IN_T:
            amb_gone.append(k)
    created, amb_new = {}, {}
    for k in gone + amb_gone:
        t = tiles[k]
        for (b, B, d, w, poly) in PLs[t["g"]]:
            u = tuple(int(x) for x in np.array(t["u"]) + np.array(w))
            if (b, u) in tile_set:
                continue                                         # a tile of the patch (present before)
            dist = trees[S_of[b]].query(B @ znew[k] + d)[0]
            if dist < IN_T:
                created[(b, u)] = None
            elif dist <= OUT_T:
                amb_new[(b, u)] = None
    return gone, amb_gone, list(created), list(amb_new)


def clusters_of(gone, amb_gone, created, amb_new):
    nodes = [("g", k, polys[k]) for k in gone] + [("a", k, polys[k]) for k in amb_gone] \
        + [("n", bu, poly_of(*bu)) for bu in created] + [("m", bu, poly_of(*bu)) for bu in amb_new]
    tree = STRtree([n[2] for n in nodes])
    parent = list(range(len(nodes)))

    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, n in enumerate(nodes):
        for j in tree.query(n[2]):
            if j > i and n[2].intersection(nodes[j][2]).area > 1e-6:
                parent[f(i)] = f(j)
    comp = {}
    for i, n in enumerate(nodes):
        comp.setdefault(f(i), []).append(n)
    return list(comp.values())


TREE_P = STRtree(polys)

def run_delta():
    global DELTA
    rows, stats = [], dict(clusters=0, ambiguous=0, near_boundary=0, not_closed=0, overlap_persisting=0, closed=0)
    persist_union_cache = {}
    t1 = time.time()
    for idir in range(NDIR):
        phi = 2 * np.pi * (idir + 0.5) / NDIR
        gone, amb_gone, created, amb_new = flip_step(phi)
        comps = clusters_of(gone, amb_gone, created, amb_new)
        gone_set = set(gone)
        for comp in comps:
            stats["clusters"] += 1
            if any(n[0] in ("a", "m") for n in comp):
                stats["ambiguous"] += 1
                continue
            rem = [n for n in comp if n[0] == "g"]
            add = [n for n in comp if n[0] == "n"]
            if not rem or not add:
                stats["not_closed"] += 1
                continue
            Fo = unary_union([n[2] for n in rem])
            Fn = unary_union([n[2] for n in add])
            if bd.distance(Fo.union(Fn)) < MARGIN:
                stats["near_boundary"] += 1
                continue
            if Fo.symmetric_difference(Fn).area > 1e-6:
                stats["not_closed"] += 1
                continue
            # the added tiles must not overlap the persisting ones
            stay = [polys[int(k)] for k in TREE_P.query(Fn) if int(k) not in gone_set and polys[int(k)].intersects(Fn)]
            ov = sum(p.intersection(n[2]).area for n in add for p in stay)
            if ov > 1e-6:
                stats["overlap_persisting"] += 1
                continue
            stats["closed"] += 1
            Fu = Fo.union(Fn)
            c0 = complex_counts([n[2] for n in rem], Fo)
            c1 = complex_counts([n[2] for n in add], Fn)
            # pairs: old tiles minus the removed ones of this cluster, plus the added ones
            rem_ids = {n[1] for n in rem}
            extra = {("new", n[1]): leaf_keys(n[1][0], n[1][1]) for n in add}
            for kk, lv in extra.items():
                leafs_old[kk] = lv                                     # reuse the dictionary lookup in pairs_of
            old_near = [int(k) for k in TREE_P.query(Fu.buffer(1e-4)) if polys[int(k)].distance(Fu) < 1e-6]
            near = [k for k in old_near if k not in rem_ids] + list(extra.keys())
            # pair sets restricted to the tiles that can meet a flat vertex of the footprint: those touching Fu
            po = pairs_of(old_near)
            pn = pairs_of(near)
            po = {k: v for k, v in po.items() if Fu.distance(Point(*flat_pos(k))) < 1e-6}
            pn = {k: v for k, v in pn.items() if Fu.distance(Point(*flat_pos(k))) < 1e-6}
            for kk in extra:
                leafs_old.pop(kk, None)
            dest = [k for k in po if k not in pn or pn[k] != po[k]]
            crea = [k for k in pn if k not in po or po[k] != pn[k]]
            rows.append(dict(
                idir=idir, nrem=len(rem), nadd=len(add),
                leaf_rem=sum(len(LEAVES[S_of[tiles[n[1]]["g"]]]) for n in rem),
                leaf_add=sum(len(LEAVES[S_of[n[1][0]]]) for n in add),
                cls_rem=sorted(tiles[n[1]]["g"] for n in rem), cls_add=sorted(n[1][0] for n in add),
                dV=c1[0] - c0[0], dE=c1[1] - c0[1], dF=c1[2] - c0[2], bV=[c0[3], c1[3]], bE=[c0[4], c1[4]],
                diam=float(Fu.bounds[2] - Fu.bounds[0]), area=float(Fo.area),
                pairs_destroyed=[list(k) for k in dest], pairs_created=[list(k) for k in crea],
                pairs_destroyed_in_old=[bool(k in pairs_old_all) for k in dest]))
        print("direction %d/%d: removed %d, added %d, ambiguous tiles %d; clusters so far %s; %.0fs" % (
            idir + 1, NDIR, len(gone), len(created), len(amb_gone) + len(amb_new), stats, time.time() - t1), flush=True)

    out = os.path.join(HERE, "patch_flips_g%d_d%g.json" % (N, DELTA))
    json.dump(dict(N=N, DELTA=DELTA, NDIR=NDIR, MARGIN=MARGIN, R2=R2, n_tiles=len(tiles), n_pairs=len(pairs_old_all),
                   n_pairs_inner=len(inner_pairs_old), stats=stats, rows=rows,
                   pairs_old=[list(k) for k in pairs_old_all]), open(out, "w"))
    print("clusters", stats, "-> %s (%.0fs)" % (out, time.time() - t00), flush=True)


if not os.environ.get("NO_RUN"):
    for DELTA in DELTAS:
        run_delta()
