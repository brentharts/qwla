#!/usr/bin/env python3
"""
Closed flip clusters at a generic phase, by cut and project (no substitution patch involved).

For a random internal phase y0 the tiles present in a disc are all (b, u) with u in the coset of class b, physical position
inside the disc and window membership z = R_b^T (STAR u + y0 - c_b) / sc  in Z_0(S_b) (nearest point of the depth-7 cloud
closer than IN_T; farther than OUT_T is absent; in between is ambiguous). The set is checked to tile the disc exactly once
(sum of the areas of the tiles clipped to the disc equals the disc area). Then phason steps y0 -> y0 + delta e (24 directions,
delta 0.03) give flips: removed tiles, added tiles from the placement lists of the removed ones, clusters of overlapping removed
and added tiles, tested for closure (equal footprints, no overlap with persisting tiles) and kept when away from the
disc boundary and free of ambiguous tiles. Output: per cluster tile counts, (dV, dE, dF), boundary counts, pairs destroyed/created.

Usage: SPECTRE_DIR=.. WADA_DIR=.. python3 generic_phase.py [n_phases=6] [n_directions=24] [delta=0.03] [R=36]
       DUMP=file.json writes every Spectre of each patch and the tiles removed by each step (for ribbon_hierarchy.py).
"""
import collections
import json
import os
import runpy
import sys
import time

import numpy as np

NPH = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NPH_ = NPH
NDIR = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NDIR_ = NDIR
DSMALL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
DSMALL_ = DSMALL
RREG = float(sys.argv[4]) if len(sys.argv) > 4 else 36.0
RREG_ = RREG
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["NO_RUN"] = "1"
os.environ["PBOX"] = "12"
sys.argv = ["patch_flips.py", "3", "1", "0.03"]
_G = runpy.run_path(os.path.join(HERE, "patch_flips.py"), run_name="patch_flips_setup")
globals().update(_G)
NPH, NDIR, DSMALL, RREG = NPH_, NDIR_, DSMALL_, RREG_
sys.path.insert(0, HERE)
import exact_window
trees = {S: exact_window._Tree(S) for S in "OG"}   # exact recursion in place of the depth-7 cloud
t00 = time.time()
PH = np.array([[to_c(np.eye(4, dtype=object)[k]).real for k in range(4)], [to_c(np.eye(4, dtype=object)[k]).imag for k in range(4)]])
V_ = lambda b: np.array(VEC[keys[b]][0], dtype=np.int64)

# ---- lattice candidates for the disc, numpy-chunked ----------------------------------------------------
RP = RREG + 16.0                                                 # footprints of Mystic placements are displaced from PH u
ymax = 2 * RAD * scale + 2 * max(np.linalg.norm(f[3]) for f in FR) + 1 + 4.0
M4 = np.vstack([PH, STAR])
Mi = np.linalg.inv(M4)
bnd = int(np.ceil(np.linalg.norm(Mi, axis=1).max() * np.hypot(RP, ymax))) + 1
print("candidate box edge", bnd, "ymax %.2f" % ymax, flush=True)
rng = np.arange(-bnd, bnd + 1, dtype=np.int64)
X2, X3 = np.meshgrid(rng, rng, indexing="ij")
X2, X3 = X2.ravel(), X3.ravel()
cl = []
for x0 in rng:
    for x1 in rng:
        U = np.stack([np.full_like(X2, x0), np.full_like(X2, x1), X2, X3], 1)
        p = U @ PH.T
        ok = (p[:, 0] ** 2 + p[:, 1] ** 2) < RP ** 2
        if not ok.any():
            continue
        U = U[ok]
        y = U @ STAR.T
        ok2 = (y[:, 0] ** 2 + y[:, 1] ** 2) < ymax ** 2
        if ok2.any():
            cl.append(U[ok2])
CAND = np.concatenate(cl)
print("lattice candidates", len(CAND), "%.0fs" % (time.time() - t00), flush=True)
CLASS_CAND = []
for b in range(len(FR)):
    Wb = CAND[in_lattice(CAND - V_(b))]
    CLASS_CAND.append(Wb)
print("per class candidates", [len(w) for w in CLASS_CAND], flush=True)
REG = Point(0, 0).buffer(RREG)
REGC = REG.exterior


def phase_tiles(y0):
    """Tiles present at phase y0 (dict (b,u) -> z), ambiguous ones (dict), over all classes."""
    present, amb = {}, {}
    for b in range(len(FR)):
        Sb, Ab, Rb, cb, _, _ = FR[b]
        Wb = CLASS_CAND[b]
        z = ((Wb @ STAR.T) + y0[None, :] - cb[None, :]) @ Rb / scale
        keep = np.linalg.norm(z, axis=1) <= 2 * RAD + 0.6
        Wk, zk = Wb[keep], z[keep]
        d = trees[Sb].query(zk)[0]
        for w, zz, dd in zip(Wk, zk, d):
            key = (b, tuple(int(x) for x in w))
            if dd < IN_T:
                present[key] = zz
            elif dd <= OUT_T:
                amb[key] = zz
    return present, amb


PCACHE, LCACHE = {}, {}


def P(key):
    if key not in PCACHE:
        PCACHE[key] = poly_of(*key)
    return PCACHE[key]


def L(key):
    if key not in LCACHE:
        LCACHE[key] = leaf_keys(*key)
    return LCACHE[key]


def step(state, phi, delta):
    e = np.array([np.cos(phi), np.sin(phi)])
    shifts = [delta * (FR[k][2].T @ R0) @ e for k in range(len(FR))]
    gone, amb_gone, znew = [], [], {}
    for key, z in state.items():
        zn = z + shifts[key[0]]
        znew[key] = zn
        d = trees[S_of[key[0]]].query(zn)[0]
        if d > OUT_T:
            gone.append(key)
        elif d >= IN_T:
            amb_gone.append(key)
    created, amb_new = {}, {}
    for key in gone + amb_gone:
        for (b, B, d, w, poly) in PLs[key[0]]:
            k2 = (b, tuple(int(x) for x in np.array(key[1]) + np.array(w)))
            if k2 in state or k2 in created or k2 in amb_new:
                continue
            zb = B @ znew[key] + d
            dist = trees[S_of[b]].query(zb)[0]
            if dist < IN_T:
                created[k2] = zb
            elif dist <= OUT_T:
                amb_new[k2] = zb
    return gone, amb_gone, created, amb_new, znew


def clusters(gone, amb_gone, created, amb_new):
    nodes = [("g", k) for k in gone] + [("a", k) for k in amb_gone] + [("n", k) for k in created] + [("m", k) for k in amb_new]
    polys_ = [P(n[1]) for n in nodes]
    tree = STRtree(polys_)
    parent = list(range(len(nodes)))

    def f(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(nodes)):
        for j in tree.query(polys_[i]):
            j = int(j)
            if j > i and polys_[i].intersection(polys_[j]).area > 1e-6:
                parent[f(i)] = f(j)
    comp = {}
    for i, n in enumerate(nodes):
        comp.setdefault(f(i), []).append(n)
    return list(comp.values())


def pairs_in(keys_):
    lst, ident = [], []
    for k in keys_:
        for li, kk in enumerate(L(k)):
            lst.append(kk)
            ident.append((k, li))
    prs = find_pairs_keys(lst)
    return {tuple(lst[i][10]): frozenset((ident[i], ident[j])) for i, j in prs}


def leaf_at(tile_keys):
    at = collections.defaultdict(list)
    for tk in tile_keys:
        for li, lk in enumerate(L(tk)):
            for v, key in enumerate(lk):
                at[tuple(key)].append(v)
    return at


def tile_pair_count(at, tk):
    """(leaves of tile tk whose flat vertex is a pair site, leaves)."""
    n = 0
    for lk in L(tk):
        lst = at[tuple(lk[10])]
        if len(lst) == 2 and all(v == 10 for v in lst):
            n += 1
    return n, len(L(tk))


def poly_coords(k):
    pl = P(k)
    g = list(pl.geoms)[0] if hasattr(pl, "geoms") else pl
    return [[round(x, 3), round(y, 3)] for x, y in g.exterior.coords]


SEEN = set()
DUMPED = []
rows, log = [], []
rs = np.random.default_rng(int(os.environ.get("SEED", "1")))
done = 0
tries = 0
while done < NPH and tries < 40:
    tries += 1
    y0 = rs.uniform(-1, 1, 2) * 1.5
    present, amb = phase_tiles(y0)
    # tiling check on the disc: tiles near the disc, clipped area
    keys_ = list(present)
    pol = [P(k) for k in keys_]
    TREE = STRtree(pol)
    D = REG.buffer(-4.0)
    cov = sum(pol[int(i)].intersection(D).area for i in TREE.query(D))
    amb_in = [k for k in amb if P(k).intersects(D)]
    ok = abs(cov - D.area) < 1e-3
    log.append(dict(try_=tries, y0=y0.tolist(), tiles=len(present), ambiguous_in_disc=len(amb_in), disc_area=float(D.area), covered=float(cov), tiling=bool(ok)))
    print(log[-1], "%.0fs" % (time.time() - t00), flush=True)
    if not ok:
        continue
    done += 1
    if os.environ.get("PAIRS"):
        _inb = REG.buffer(-MARGIN)
        _mem0 = [k for k in keys_ if _inb.contains(P(k).centroid)]
        _at0 = collections.defaultdict(list)
        for tk in keys_:
            for li, lk in enumerate(L(tk)):
                for v, key in enumerate(lk):
                    _at0[tuple(key)].append(v)
        _c0 = collections.Counter()
        for tk in _mem0:
            for lk in L(tk):
                lst = _at0[tuple(lk[10])]
                _c0["alone" if len(lst) == 1 else ("pair" if len(lst) == 2 and all(v == 10 for v in lst) else ("two_other" if len(lst) == 2 else "three_plus"))] += 1
        log[-1]["baseline_cat"] = dict(_c0)
        print("baseline leaf categories (inner disc):", dict(_c0), flush=True)
    if os.environ.get("DUMP"):                      # every Spectre of the patch, for ribbon_hierarchy.py
        _kid = {k: a for a, k in enumerate(keys_)}
        _polys = [[[[round(x, 4), round(y, 4)] for x, y in g.exterior.coords[:-1]] for g in (list(p.geoms) if hasattr(p, "geoms") else [p])] for p in pol]
        _win = [[S_of[k[0]], round(float(present[k][0]), 5), round(float(present[k][1]), 5)] for k in keys_]      # window point of each key, in Z_0(S)
        DUMPED.append(dict(phase=done, y0=y0.tolist(), R=RREG, delta=DSMALL, tiles=_polys, window=_win, steps=[]))
    for idir in range(NDIR):
        phi = 2 * np.pi * (idir + 0.5) / NDIR
        gone, amb2, created, ambn, zn = step(present, phi, DSMALL)
        if os.environ.get("DUMP"):
            DUMPED[-1]["steps"].append(dict(idir=idir, phi=float(phi), removed=[_kid[k] for k in gone if k in _kid]))
        gset = set(gone)
        for comp in clusters(gone, amb2, created, ambn):
            if any(n[0] in ("a", "m") for n in comp):
                rows.append(dict(status="ambiguous")); continue
            rem = [n[1] for n in comp if n[0] == "g"]
            add = [n[1] for n in comp if n[0] == "n"]
            _all = unary_union([P(n[1]) for n in comp])
            if REGC.distance(_all) < MARGIN or not REG.contains(_all):
                _Fo = unary_union([P(k) for k in rem]) if rem else None; _Fn = unary_union([P(k) for k in add]) if add else None
                _sd = (_Fo.symmetric_difference(_Fn) if (_Fo is not None and _Fn is not None) else _all)
                _in = REG.buffer(-MARGIN)
                _ends = []
                if _Fo is not None and _Fn is not None:
                    _parts = list(_sd.geoms) if hasattr(_sd, "geoms") else [_sd]
                    for _pp in _parts:
                        if _pp.area < 1e-6:
                            continue
                        _mem = [(("rem" if n[0] == "g" else "add"), int(n[1][0]), S_of[n[1][0]]) for n in comp
                                if n[0] in ("g", "n") and P(n[1]).intersection(_pp).area > 1e-6]
                        _ends.append(dict(area=float(_pp.area), pos=[float(_pp.centroid.x), float(_pp.centroid.y)], tiles=_mem))
                _sig = hash(tuple(sorted(rem)))
                _pv = None
                if len(comp) >= 20 and os.environ.get("PAIRS"):
                    _in = REG.buffer(-MARGIN)
                    _near = [keys_[int(i)] for i in TREE.query(_all.buffer(1e-4)) if P(keys_[int(i)]).distance(_all) < 1e-6]
                    _rs = set(rem)
                    _po = {k: v for k, v in pairs_in(_near).items() if _in.contains(Point(*flat_pos(k)))}
                    _pn = {k: v for k, v in pairs_in([k for k in _near if k not in _rs] + add).items() if _in.contains(Point(*flat_pos(k)))}
                    _dest = [k for k in _po if k not in _pn or _pn[k] != _po[k]]
                    _crea = [k for k in _pn if k not in _po or _po[k] != _pn[k]]
                    _cen = np.array([[P(k).centroid.x, P(k).centroid.y] for k in rem])
                    _cin = _cen[[_in.contains(Point(*c)) for c in _cen]] if len(_cen) else _cen
                    if len(_cin) > 2:
                        _u, _sv, _vt = np.linalg.svd(_cin - _cin.mean(0)); _ax = _vt[0]
                        _t = (_cin - _cin.mean(0)) @ _ax; _len = float(_t.max() - _t.min())
                        _tp = lambda q: float((np.array(q) - _cin.mean(0)) @ _ax)
                    else:
                        _ax, _len, _tp = None, 0.0, None
                    def _cats(tile_keys, members):
                        at = collections.defaultdict(list)
                        for tk in tile_keys:
                            for li, lk in enumerate(L(tk)):
                                for v, key in enumerate(lk):
                                    at[tuple(key)].append((tk, li, v))
                        c = collections.Counter()
                        for tk in members:
                            for li, lk in enumerate(L(tk)):
                                lst = at[tuple(lk[10])]
                                if len(lst) == 1:
                                    c["alone"] += 1
                                elif len(lst) == 2:
                                    c["pair" if all(v == 10 for _, _, v in lst) else "two_other"] += 1
                                else:
                                    c["three_plus"] += 1
                        return dict(c)
                    _cat_rem = _cats(_near, rem)
                    _cat_add = _cats([k for k in _near if k not in _rs] + add, add)
                    _wide = [keys_[int(i)] for i in TREE.query(_all.buffer(10.0)) if P(keys_[int(i)]).distance(_all) < 10.0]
                    _dist = {k: P(k).distance(_all) for k in _wide}
                    _mid = lambda k: _in.contains(P(k).centroid)
                    _b0 = [k for k in _wide if k in _rs]                       # removed tiles (old tiling)
                    _b1 = [k for k in _wide if k not in _rs and _dist[k] <= 4.0 and _mid(k)]
                    _b2 = [k for k in _wide if k not in _rs and 4.0 < _dist[k] <= 8.0 and _mid(k)]
                    _env = dict(b1=_cats(_wide, _b1), b2=_cats(_wide, _b2))
                    _at_old = leaf_at(_wide)
                    _at_new = leaf_at([k for k in _wide if k not in _rs] + add)
                    _ti = lambda k: (float((np.array([P(k).centroid.x, P(k).centroid.y]) - _cin.mean(0)) @ _ax) if _ax is not None else None)
                    _trem = [[int(k[0]), S_of[k[0]], *map(float, present[k]), *map(float, zn[k]), float(P(k).centroid.x), float(P(k).centroid.y), *tile_pair_count(_at_old, k), _ti(k)] for k in rem]
                    _tadd = [[int(k[0]), S_of[k[0]], *map(float, created[k]), float(P(k).centroid.x), float(P(k).centroid.y), *tile_pair_count(_at_new, k), _ti(k)] for k in add]
                    _geo = None
                    if os.environ.get("GEO") and (done, _sig) not in SEEN:
                        SEEN.add((done, _sig))
                        _bnd = [k for k in _wide if k not in _rs and _dist[k] <= 8.0 and _mid(k)]
                        _geo = dict(rem=[poly_coords(k) for k in rem], rem_np=[tile_pair_count(_at_old, k) for k in rem],
                                    add=[poly_coords(k) for k in add], add_np=[tile_pair_count(_at_new, k) for k in add],
                                    band=[poly_coords(k) for k in _bnd], band_np=[tile_pair_count(_at_old, k) for k in _bnd],
                                    axis=[float(a) for a in _ax] if _ax is not None else None, mean=[float(a) for a in _cin.mean(0)] if len(_cin) else None)
                    _pv = dict(trem=_trem, tadd=_tadd, geo=_geo, phi=float(phi), env=_env, cat_rem=_cat_rem, cat_add=_cat_add, len_in=_len, ntiles_in=int(len(_cin)), n_pairs_before=len(_po), n_pairs_after=len(_pn),
                               dest=[_tp(flat_pos(k)) if _tp else None for k in _dest], crea=[_tp(flat_pos(k)) if _tp else None for k in _crea],
                               dest_xy=[list(map(float, flat_pos(k))) for k in _dest], crea_xy=[list(map(float, flat_pos(k))) for k in _crea])
                rows.append(dict(status="near boundary or outside", pv=_pv, cls_rem=sorted(int(k[0]) for k in rem), cls_add=sorted(int(k[0]) for k in add), phase=done, idir=idir, sig=_sig, ends=_ends, symdiff=float(_sd.area), symdiff_inner=float(_sd.intersection(_in).area), n=len(comp), nrem=len(rem), nadd=len(add), diam=float(_all.bounds[2]-_all.bounds[0]), rad=float(np.hypot(*_all.centroid.coords[0])))); continue
            if not rem or not add:
                rows.append(dict(status="one-sided", nrem=len(rem), nadd=len(add), pos=[float(P((rem or add)[0]).centroid.x), float(P((rem or add)[0]).centroid.y)])); continue
            Fo = unary_union([P(k) for k in rem]); Fn = unary_union([P(k) for k in add]); Fu = Fo.union(Fn)
            diam = float(max(Fu.bounds[2] - Fu.bounds[0], Fu.bounds[3] - Fu.bounds[1]))
            if Fo.symmetric_difference(Fn).area > 1e-6:
                rows.append(dict(status="not closed", area_diff=float(Fo.symmetric_difference(Fn).area), nrem=len(rem), nadd=len(add), diam=diam)); continue
            near = [keys_[int(i)] for i in TREE.query(Fu.buffer(1e-4)) if P(keys_[int(i)]).distance(Fu) < 1e-6]
            stay = [P(k) for k in near if k not in gset]
            if sum(p.intersection(P(k)).area for k in add for p in stay) > 1e-6:
                rows.append(dict(status="overlaps persisting")); continue
            c0 = complex_counts([P(k) for k in rem], Fo)
            c1 = complex_counts([P(k) for k in add], Fn)
            rem_set = set(rem)
            po = {k: v for k, v in pairs_in(near).items() if Fu.distance(Point(*flat_pos(k))) < 1e-6}
            pn = {k: v for k, v in pairs_in([k for k in near if k not in rem_set] + add).items() if Fu.distance(Point(*flat_pos(k))) < 1e-6}
            dest = [k for k in po if k not in pn or pn[k] != po[k]]
            crea = [k for k in pn if k not in po or po[k] != pn[k]]
            rows.append(dict(
                status="closed", phase=done, idir=idir, nrem=len(rem), nadd=len(add),
                leaf_rem=sum(len(LEAVES[S_of[k[0]]]) for k in rem), leaf_add=sum(len(LEAVES[S_of[k[0]]]) for k in add),
                cls_rem=sorted(k[0] for k in rem), cls_add=sorted(k[0] for k in add),
                dV=c1[0] - c0[0], dE=c1[1] - c0[1], dF=c1[2] - c0[2], bV=[c0[3], c1[3]], bE=[c0[4], c1[4]],
                V=[c0[0], c1[0]], E=[c0[1], c1[1]], F=[c0[2], c1[2]], diam=diam, area=float(Fo.area),
                pairs_destroyed=[list(k) for k in dest], pairs_created=[list(k) for k in crea]))
        print("phase %d direction %d/%d: %s (%.0fs)" % (done, idir + 1, NDIR, dict(collections.Counter(r["status"] for r in rows)), time.time() - t00), flush=True)

out = os.path.join(HERE, "generic_phase_d%g_R%g%s.json" % (DSMALL, RREG, os.environ.get("TAG", "")))
json.dump(dict(DELTA=DSMALL, R=RREG, MARGIN=MARGIN, log=log, rows=rows), open(out, "w"))
print(collections.Counter(r["status"] for r in rows))
print("->", out)
if os.environ.get("DUMP"):
    json.dump(DUMPED, open(os.path.join(HERE, os.environ["DUMP"]), "w"))
    print("->", os.environ["DUMP"])
