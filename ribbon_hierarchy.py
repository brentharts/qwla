#!/usr/bin/env python3
"""
Do phason ribbons follow the supertiles?  (ROADMAP Stage 9q.)

A patch at a generic phase (generic_phase.py with DUMP=...) has no hierarchy attached. This script puts one on it
with the local rule of Stage 9n: a dictionary  pattern of Spectres within 10.25 -> (level-1 supertile)  and
pattern of level-1 supertiles within 28.3 -> (level-2 supertile), both read off a level-5 substitution patch, is
applied to the tiles of the dump. Then, for every ribbon (a connected set of at least 20 removed tiles):

  boundary   share of ribbon tiles that touch a different level-2 supertile, against the share for all tiles;
  axis       angle between the local direction of the ribbon and the long axis of the level-2 supertile it is in.

The dump's tiles are matched to the substitution code's tile by a rigid fit and snapped to the lattice
(coordinates (a + b sqrt3)/2), so that both sides hash identical numbers. Two-tile Mystics arrive as one 20-gon and
are split with the solver of hole_uniqueness.py.

Usage: python3 ribbon_hierarchy.py dump.json [dump2.json ...] [--tests]
"""
import json
import math
import os
import pickle
import sys
from collections import defaultdict, Counter

import numpy as np
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recognisability_radius as RR  # noqa: E402
import hole_uniqueness as HU  # noqa: E402
SP, LAM = RR.SP, RR.LAM
RHO0, RHO1 = 10.25, 28.3
REF = np.array(SP.get_spectre_points(1.0, 1.0))
S3 = math.sqrt(3.0)
VALS = np.array([0.0, 0.5, -0.5, S3 / 2, -S3 / 2, 1.0, -1.0])
_rng = np.random.default_rng(12345)
MULT = _rng.integers(1, 2 ** 63, size=7, dtype=np.uint64) * np.uint64(2) + np.uint64(1)


def keys_of(P, L, lid, rho, centres):
    """64-bit hash of the multiset of relative poses (and labels) of the units within rho of each centre."""
    tree = cKDTree(P)
    out = np.zeros(len(centres), dtype=np.uint64)
    nbs = tree.query_ball_point(P[centres], rho)
    for a, (u, lst) in enumerate(zip(centres, nbs)):
        v = np.array([x for x in lst if x != u], dtype=int)
        if len(v) == 0:
            continue
        Lt = L[u].T
        p = (P[v] - P[u]) @ Lt.T
        M = np.einsum("ij,njk->nik", Lt, L[v]).reshape(-1, 4)
        q = np.rint(np.concatenate([p, M], axis=1) * 1000).astype(np.int64).astype(np.uint64)
        with np.errstate(over="ignore"):
            h = (q * MULT[:6]).sum(axis=1)
            if lid is not None:
                h = h + lid[v] * MULT[6]
            h ^= h >> np.uint64(29); h *= np.uint64(0xBF58476D1CE4E5B9); h ^= h >> np.uint64(32)
            k = h.sum()
            if lid is not None:
                k = k + (lid[u] + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
        out[a] = k
    return out


NAMES = sorted(SP.TILE_NAMES)


def dictionaries(level=5):
    cache = os.path.join(HERE, ".ribbon_hierarchy_dict_L%d.pkl" % level)
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    levels, leaves = RR.build(level)
    polys = [Polygon((T[:, :2] @ REF.T).T + T[:, 2]) for T, _, _ in leaves]
    groups = defaultdict(list)
    for p, (T, lab, b) in zip(polys, leaves):
        groups[levels[1][levels[0][b][2]][2]].append(p.buffer(1e-6))
    ext = shapely.ops.unary_union([shapely.ops.unary_union(v) for v in groups.values()]).exterior

    def inside(P):
        return np.asarray(shapely.distance(shapely.points(P), ext))

    def rel(Tu, Tv):                              # pose of v in the frame of u (exact floats)
        Lt = Tu[:, :2].T
        return np.concatenate([Lt @ Tv[:, :2], (Lt @ (Tv[:, 2] - Tu[:, 2]))[:, None]], axis=1)

    out = {}
    # level 0: Spectres without labels -> (leaf label, pose of the level-1 supertile in the tile's frame, its label)
    P = np.array([T[:, 2] for T, _, _ in leaves]); L = np.array([T[:, :2] for T, _, _ in leaves])
    cen = np.where(inside(P) > RHO0 + 1e-6)[0]
    ks = keys_of(P, L, None, RHO0, cen)
    d0, clash = {}, 0
    for k, u in zip(ks, cen):
        T, lab, b = leaves[u]
        par = levels[0][b][2]
        val = (lab, rel(T, levels[1][par][0]), levels[1][par][1])
        if int(k) in d0:
            o = d0[int(k)]
            clash += not (o[0] == val[0] and o[2] == val[2] and np.abs(o[1] - val[1]).max() < 1e-6)
        else:
            d0[int(k)] = val
    # level 1: supertiles with labels -> (pose of the level-2 supertile, its label)
    U = levels[1]
    P1 = np.array([T[:, 2] for T, _, _ in U]); L1 = np.array([T[:, :2] for T, _, _ in U])
    lid = np.array([NAMES.index(lab) for _, lab, _ in U], dtype=np.uint64)
    cen = np.where(inside(P1) > RHO1 + 1e-6)[0]
    ks = keys_of(P1, L1, lid, RHO1, cen)
    d1, clash1 = {}, 0
    for k, u in zip(ks, cen):
        T, lab, par = U[u]
        val = (rel(T, levels[2][par][0]), levels[2][par][1])
        if int(k) in d1:
            o = d1[int(k)]
            clash1 += not (o[1] == val[1] and np.abs(o[0] - val[0]).max() < 1e-6)
        else:
            d1[int(k)] = val
    out = dict(d0=d0, d1=d1, clash0=clash, clash1=clash1)
    pickle.dump(out, open(cache, "wb"))
    return out


def snap_coord(x):
    b = np.arange(-400, 401)
    r = 2 * x - b * S3
    f = np.abs(r - np.rint(r))
    j = int(np.argmin(f))
    assert f[j] < 2e-3, ("coordinate not on the lattice", x, f[j])
    return (np.rint(r[j]) + b[j] * S3) / 2


def pose_of(Q):
    """Affine map (2x3, reflection) taking the reference Spectre to the 14-gon Q, snapped to the lattice."""
    Q = np.asarray(Q, float)
    x, y = Q[:, 0], Q[:, 1]
    if (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() > 0:         # the patch's hand is the mirror of REF: clockwise order
        Q = Q[::-1]
    best = None
    for s in range(14):
        Qs = np.roll(Q, -s, axis=0)
        A = Qs - Qs.mean(0); B = REF - REF.mean(0)
        Lm = np.linalg.lstsq(B, A, rcond=None)[0].T
        err = np.abs(B @ Lm.T - A).max()
        if best is None or err < best[0]:
            best = (err, Lm, Qs)
    err, Lm, Qs = best
    assert err < 5e-3, ("not a Spectre", err)
    Ls = VALS[np.argmin(np.abs(Lm[..., None] - VALS), axis=-1)]
    assert abs(np.linalg.det(Ls) + 1) < 1e-9
    t = (Qs - REF @ Ls.T).mean(0)
    return np.concatenate([Ls, np.array([[snap_coord(t[0])], [snap_coord(t[1])]])], axis=1)


def split(poly_list, S):
    """Spectres of one dumped key: a 14-gon, or a 20-gon (Mystic) split in two."""
    out = []
    for Q in poly_list:
        Q = np.asarray(Q, float)
        if len(Q) == 14:
            out.append(Q); continue
        M = Polygon(Q).buffer(2e-3, join_style=2)                # dumped coordinates are rounded to 1e-4
        found = []
        for deg in range(0, 360, 30):
            th = math.radians(deg); c, sn = math.cos(th), math.sin(th)
            Sr = S @ np.array([[c, -sn], [sn, c]]).T
            for v in Q:
                for i in range(14):
                    T = Sr - Sr[i] + v
                    cen = T.mean(0)
                    if any(np.abs(cen - f.mean(0)).sum() < 1e-2 for f in found):
                        continue
                    if Polygon(T).difference(M).area < 1e-3:
                        found.append(T)
        pair = [(x, y) for x in range(len(found)) for y in range(x) if Polygon(found[x]).intersection(Polygon(found[y])).area < 1e-2]
        assert len(pair) == 1, "Mystic did not split into two Spectres in exactly one way"      # several placements fit; one pair is disjoint
        out += [found[pair[0][0]], found[pair[0][1]]]
    return out


def compose(T, R):
    return np.concatenate([T[:, :2] @ R[:, :2], (T[:, :2] @ R[:, 2] + T[:, 2])[:, None]], axis=1)


def analyse(dump, D):
    S = HU.ccw((REF * np.array([-1.0, 1.0])))                       # the hand of the dump (mirror of REF)
    tiles, owner = [], []
    for a, pl in enumerate(dump["tiles"]):
        for Q in split(pl, S):
            tiles.append(pose_of(Q)); owner.append(a)
    n = len(tiles)
    P = np.array([T[:, 2] for T in tiles]); L = np.array([T[:, :2] for T in tiles])
    polys = [Polygon(REF @ T[:, :2].T + T[:, 2]) for T in tiles]
    cent = np.array([[p.centroid.x, p.centroid.y] for p in polys])
    k0 = keys_of(P, L, None, RHO0, np.arange(n))
    Rin = dump["R"] - 4.0
    full0 = np.linalg.norm(P, axis=1) < Rin - RHO0 - 3.0
    sup1 = {}                                                       # level-1 supertile key -> [pose, label, members]
    t1 = np.full(n, -1); found0 = 0
    lab0 = [None] * n; lab1 = [None] * n
    keys1 = []
    for u in range(n):
        v = D["d0"].get(int(k0[u]))
        if v is None:
            continue
        found0 += bool(full0[u])
        lab0[u], lab1[u] = v[0], v[2]
        T1 = compose(tiles[u], v[1])
        kk = (round(float(T1[0, 2]), 3), round(float(T1[1, 2]), 3), round(float(T1[0, 0]), 2), round(float(T1[0, 1]), 2), v[2])
        if kk not in sup1:
            sup1[kk] = [T1, v[2], []]; keys1.append(kk)
        sup1[kk][2].append(u); t1[u] = keys1.index(kk) if False else len(keys1) - 1 if kk == keys1[-1] else keys1.index(kk)
    U1 = [sup1[k] for k in keys1]
    # level 2
    P1 = np.array([u[0][:, 2] for u in U1]); L1 = np.array([u[0][:, :2] for u in U1])
    lid = np.array([NAMES.index(u[1]) for u in U1], dtype=np.uint64)
    k1 = keys_of(P1, L1, lid, RHO1, np.arange(len(U1)))
    sup2, keys2 = {}, []
    s2_of_1 = np.full(len(U1), -1); 
    for a in range(len(U1)):
        v = D["d1"].get(int(k1[a]))
        if v is None:
            continue
        T2 = compose(U1[a][0], v[0])
        kk = (round(float(T2[0, 2]), 3), round(float(T2[1, 2]), 3), round(float(T2[0, 0]), 2), round(float(T2[0, 1]), 2), v[1])
        if kk not in sup2:
            sup2[kk] = len(keys2); keys2.append(kk)
        s2_of_1[a] = sup2[kk]
    t2 = np.where(t1 >= 0, s2_of_1[np.clip(t1, 0, None)], -1)
    # neighbours (shared boundary) and boundary flags
    tree = STRtree(polys)
    pairs = tree.query(polys, predicate="dwithin", distance=1e-6)
    nb = defaultdict(list)
    for a, b in zip(*pairs):
        if a != b:
            nb[int(a)].append(int(b))

    def boundary(lab):
        known = lab >= 0
        flag = np.zeros(n, bool); valid = np.zeros(n, bool)
        for u in range(n):
            if not known[u]:
                continue
            ns = nb[u]
            if all(known[v] for v in ns):
                valid[u] = True
                flag[u] = any(lab[v] != lab[u] for v in ns)
        return flag, valid

    b1, v1 = boundary(t1); b2, v2 = boundary(t2)
    # long axis of each complete-looking level-2 supertile (at least 55 known tiles)
    axis2 = {}
    for s in range(len(keys2)):
        m = np.where(t2 == s)[0]
        if len(m) >= 55:
            X = cent[m] - cent[m].mean(0)
            w, vv = np.linalg.eigh(X.T @ X)
            axis2[s] = math.degrees(math.atan2(vv[1, 1], vv[0, 1])) % 180
    # ribbons
    spectres_of = defaultdict(list)
    for u, a in enumerate(owner):
        spectres_of[a].append(u)
    ribs = []
    for st in dump["steps"]:
        rem = sorted({u for a in st["removed"] for u in spectres_of[a]})
        rs = set(rem); comp = {}
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
            g = np.array(sorted(g))
            X = cent[g] - cent[g].mean(0)
            w, vv = np.linalg.eigh(X.T @ X)
            ang = math.degrees(math.atan2(vv[1, 1], vv[0, 1])) % 180
            d = []
            for u in g:                                              # local direction against the supertile's long axis
                if t2[u] in axis2:
                    loc = g[np.linalg.norm(cent[g] - cent[u], axis=1) < 15]
                    if len(loc) >= 6:
                        Y = cent[loc] - cent[loc].mean(0); w2, v2_ = np.linalg.eigh(Y.T @ Y)
                        la = math.degrees(math.atan2(v2_[1, 1], v2_[0, 1])) % 180
                        dd = abs(la - axis2[t2[u]]) % 180
                        d.append(min(dd, 180 - dd))
            ribs.append(dict(tile_labels=dict(Counter(lab0[u] for u in g if lab0[u])), supertile_labels=dict(Counter(lab1[u] for u in g if lab1[u])), idir=st["idir"], tiles=len(g), angle=ang, aspect=float(math.sqrt(w[1] / max(w[0], 1e-9))),
                             n_b1=int(v1[g].sum()), on_b1=int(b1[g][v1[g]].sum()), n_b2=int(v2[g].sum()), on_b2=int(b2[g][v2[g]].sum()),
                             axis_diffs=d, members=g.tolist()))
    sizes = Counter(len(u[2]) for u in U1)
    return dict(tile_labels=dict(Counter(x for x in lab0 if x)), supertile_labels=dict(Counter(x for x in lab1 if x)), phase=dump["phase"], R=dump["R"], spectres=n, pattern_found_share=float(found0 / max(1, full0.sum())), tiles_with_full_disc=int(full0.sum()),
                level1=len(U1), level1_sizes={str(k): v for k, v in sorted(sizes.items())}, level2=len(keys2), tiles_with_level2=int((t2 >= 0).sum()),
                base_b1=[int(b1[v1].sum()), int(v1.sum())], base_b2=[int(b2[v2].sum()), int(v2.sum())], complete_level2=len(axis2), ribbons=ribs)


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    D = dictionaries()
    print("dictionary: %d level-0 patterns (%d clashes), %d level-1 patterns (%d clashes)" % (len(D["d0"]), D["clash0"], len(D["d1"]), D["clash1"]))
    res = []
    for f in files:
        for dump in json.load(open(f)):
            r = analyse(dump, D); r["file"] = os.path.basename(f); res.append(r)
            print("%s phase %d: %d Spectres; pattern found for %.3f of %d tiles with a full disc; %d level-1 supertiles (sizes %s); %d level-2, %d tiles assigned; complete level-2: %d" % (
                r["file"], r["phase"], r["spectres"], r["pattern_found_share"], r["tiles_with_full_disc"], r["level1"], r["level1_sizes"], r["level2"], r["tiles_with_level2"], r["complete_level2"]))
            print("   all tiles on a level-1 boundary: %d of %d; on a level-2 boundary: %d of %d" % (*r["base_b1"], *r["base_b2"]))
            for rb in r["ribbons"]:
                print("   ribbon dir %2d: %3d tiles, angle %5.1f; level-2 boundary %d of %d; axis differences n=%d mean %s" % (
                    rb["idir"], rb["tiles"], rb["angle"], rb["on_b2"], rb["n_b2"], len(rb["axis_diffs"]), "%.1f" % np.mean(rb["axis_diffs"]) if rb["axis_diffs"] else "-"))
    # pooled over distinct ribbons (same member set counted once)
    seen, pool = set(), dict(n_b2=0, on_b2=0, n_b1=0, on_b1=0, diffs=[])
    rl0, rl1, al0, al1 = Counter(), Counter(), Counter(), Counter()
    for r in res:
        al0.update(r["tile_labels"]); al1.update(r["supertile_labels"])
    for r in res:
        for rb in r["ribbons"]:
            key = (r["file"], r["phase"], tuple(rb["members"][:10]), rb["tiles"])
            if key in seen:
                continue
            seen.add(key)
            for k in ("n_b2", "on_b2", "n_b1", "on_b1"):
                pool[k] += rb[k]
            pool["diffs"] += rb["axis_diffs"]
            rl0.update(rb["tile_labels"]); rl1.update(rb["supertile_labels"])
    base2 = [sum(r["base_b2"][0] for r in res), sum(r["base_b2"][1] for r in res)]
    base1 = [sum(r["base_b1"][0] for r in res), sum(r["base_b1"][1] for r in res)]
    summ = dict(distinct_ribbons=len(seen), ribbon_level2_boundary=[pool["on_b2"], pool["n_b2"]], all_level2_boundary=base2,
                ribbon_level1_boundary=[pool["on_b1"], pool["n_b1"]], all_level1_boundary=base1,
                axis_diff_mean=float(np.mean(pool["diffs"])) if pool["diffs"] else None, axis_diff_n=len(pool["diffs"]),
                axis_diff_hist_15deg=np.histogram(pool["diffs"], bins=[0, 15, 30, 45, 60, 75, 90])[0].tolist() if pool["diffs"] else None)
    def shares(rc, ac):
        return {k: [rc.get(k, 0), ac[k], round(rc.get(k, 0) / max(1, sum(rc.values())), 4), round(ac[k] / sum(ac.values()), 4)] for k in sorted(ac)}
    summ["tile_label_ribbon_all"] = shares(rl0, al0)              # label: [ribbon count, all count, ribbon share, all share]
    summ["supertile_label_ribbon_all"] = shares(rl1, al1)
    print(summ)
    for r in res:
        for rb in r["ribbons"]:
            rb.pop("members")
    json.dump(dict(summary=summ, runs=res), open(os.path.join(HERE, "ribbon_hierarchy.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        assert D["clash0"] == 0 and D["clash1"] == 0
        assert all(r["pattern_found_share"] > 0.99 for r in res)
        print("ribbon_hierarchy: tests passed (dictionary single-valued; the rule applies to more than 99% of the generic-phase tiles)")


if __name__ == "__main__":
    main()
