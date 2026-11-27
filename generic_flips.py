#!/usr/bin/env python3
"""
Closed flip clusters at generic phases (open problem 1 of quantgrav: do flip clusters tile, and which moves are they?).

`patch_flips.py` showed that the substitution patch is a non-generic phase: its tiles sit where small phason steps
change nothing or everything. Here the patch is first moved by one large step (delta 0.2 to 0.3, which percolates
over the patch and is not itself a local flip) to a phase s. The result T(s) is a tiling: the persisting patch tiles
(window points moved by the step) plus the added ones (window points B z + d from the removed neighbour); it is
checked to cover the patch exactly once. From T(s) small steps (delta 0.03, 24 directions) are generic flips: each
cluster of changed tiles (removed and added tiles linked by overlap) is tested for closure (equal footprints, added tiles
do not overlap persisting ones), kept only if it is away from the patch boundary and has no tile in the ambiguous band.

For every closed cluster: tiles removed/added (windows and leaves), (dV, dE, dF) of the cell complex (the Pachner-type
data), the boundary vertices/edges before and after, and the bump-and-dent pairs destroyed and created.
Result: T(s) is a valid tiling but flips from it are block-like, ambiguous at the depth-7 window cloud, and no closed cluster
was found (ROADMAP Stage 3). Kept as the tool for a phase that is not derived from a substitution patch.
Usage: SPECTRE_DIR=.. WADA_DIR=.. python3 generic_flips.py [N=3] [n_phases=6] [n_directions=24] [delta=0.03]
"""
import os
import runpy
import sys
import json
import time

import numpy as np

sys.argv_saved = list(sys.argv) + [None] * 5
N = sys.argv[1] if len(sys.argv) > 1 else "3"
NPH = int(sys.argv[2]) if len(sys.argv) > 2 else 6
sys.argv_saved[2:5] = [str(NPH), "24", "0.03"]
NDIR = int(sys.argv[3]) if len(sys.argv) > 3 else 24
sys.argv_saved[3] = str(NDIR)
DELTA_SMALL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
sys.argv_saved[4] = str(DELTA_SMALL)
DELTA_BIG = float(os.environ.get("DELTA_BIG", "0.25"))
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["NO_RUN"] = "1"
sys.argv = ["patch_flips.py", N, "1", "0.03"]
g = runpy.run_path(os.path.join(HERE, "patch_flips.py"), run_name="patch_flips_setup")
globals().update(g)                                               # tiles, FR, trees, PLs, poly_of, leaf_keys, ...
NPH, NDIR, DELTA_SMALL = int(sys.argv_saved[2]), int(sys.argv_saved[3]), float(sys.argv_saved[4])
t00 = time.time()

PCACHE = {}


def P(key):
    if key not in PCACHE:
        PCACHE[key] = poly_of(*key)
    return PCACHE[key]


LCACHE = {}


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
            if k2 in state or k2 in created:
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


# ---- phase s: one big step from the patch ---------------------------------------------------------------
base = {(t["g"], t["u"]): t["z"].copy() for t in tiles}
rows, log = [], []
ph_done = 0
seen = set()
for iph in range(120):
    if ph_done >= NPH:
        break
    phi_s = 2 * np.pi * (iph + 0.37) / 120
    DELTA_BIG = [0.2, 0.25, 0.3, 0.4][iph % 4]
    gone, amb_gone, created, amb_new, znew = step(base, phi_s, DELTA_BIG)
    sig = (frozenset(gone), frozenset(created))
    if sig in seen:
        continue
    seen.add(sig)
    if amb_gone or amb_new:
        log.append(("skip phase", iph, "ambiguous", len(amb_gone), len(amb_new)))
        continue
    state = {k: znew[k] for k in base if k not in set(gone)}
    state.update(created)
    # T(s) must cover the patch exactly once
    pol = [P(k) for k in state]
    TREE = STRtree(pol)
    keys_state = list(state)
    cov = 0.0
    for k in keys_state:
        cov += P(k).intersection(patch_union).area
    ok = abs(cov - patch_union.area) < 1e-3
    log.append(("phase", iph, "removed", len(gone), "added", len(created), "tiles", len(state), "covers patch once", bool(ok), "area sum %.3f patch %.3f" % (cov, patch_union.area)))
    print(log[-1], flush=True)
    if not ok:
        continue
    ph_done += 1
    pairs_T = pairs_in(keys_state)
    for idir in range(NDIR):
        phi = 2 * np.pi * (idir + 0.5) / NDIR
        gone2, amb2, created2, ambn2, zn2 = step(state, phi, DELTA_SMALL)
        gset = set(gone2)
        for comp in clusters(gone2, amb2, created2, ambn2):
            if any(n[0] in ("a", "m") for n in comp):
                rows.append(dict(status="ambiguous")); continue
            rem = [n[1] for n in comp if n[0] == "g"]
            add = [n[1] for n in comp if n[0] == "n"]
            if not rem or not add:
                rows.append(dict(status="one-sided")); continue
            Fo = unary_union([P(k) for k in rem])
            Fn = unary_union([P(k) for k in add])
            Fu = Fo.union(Fn)
            if bd.distance(Fu) < MARGIN:
                rows.append(dict(status="near boundary")); continue
            if Fo.symmetric_difference(Fn).area > 1e-6:
                rows.append(dict(status="not closed", area_diff=float(Fo.symmetric_difference(Fn).area), nrem=len(rem), nadd=len(add), diam=float(max(Fu.bounds[2] - Fu.bounds[0], Fu.bounds[3] - Fu.bounds[1])))); continue
            near = [keys_state[int(i)] for i in TREE.query(Fu.buffer(1e-4)) if P(keys_state[int(i)]).distance(Fu) < 1e-6]
            stay = [P(k) for k in near if k not in gset]
            if sum(p.intersection(P(k)).area for k in add for p in stay) > 1e-6:
                rows.append(dict(status="overlaps persisting")); continue
            c0 = complex_counts([P(k) for k in rem], Fo)
            c1 = complex_counts([P(k) for k in add], Fn)
            old_t = near
            new_t = [k for k in near if k not in set(rem)] + add
            po = {k: v for k, v in pairs_in(old_t).items() if Fu.distance(Point(*flat_pos(k))) < 1e-6}
            pn = {k: v for k, v in pairs_in(new_t).items() if Fu.distance(Point(*flat_pos(k))) < 1e-6}
            dest = [k for k in po if k not in pn or pn[k] != po[k]]
            crea = [k for k in pn if k not in po or po[k] != pn[k]]
            rows.append(dict(
                status="closed", phase=iph, idir=idir, nrem=len(rem), nadd=len(add),
                leaf_rem=sum(len(LEAVES[S_of[k[0]]]) for k in rem), leaf_add=sum(len(LEAVES[S_of[k[0]]]) for k in add),
                cls_rem=sorted(k[0] for k in rem), cls_add=sorted(k[0] for k in add),
                dV=c1[0] - c0[0], dE=c1[1] - c0[1], dF=c1[2] - c0[2], bV=[c0[3], c1[3]], bE=[c0[4], c1[4]],
                V=[c0[0], c1[0]], E=[c0[1], c1[1]], F=[c0[2], c1[2]],
                diam=float(max(Fu.bounds[2] - Fu.bounds[0], Fu.bounds[3] - Fu.bounds[1])), area=float(Fo.area),
                pairs_destroyed=[list(k) for k in dest], pairs_created=[list(k) for k in crea]))
        nclosed = sum(r["status"] == "closed" for r in rows)
        print("phase %d direction %d/%d: closed clusters so far %d (%.0fs)" % (iph, idir + 1, NDIR, nclosed, time.time() - t00), flush=True)

import collections
print(collections.Counter(r["status"] for r in rows))
out = os.path.join(HERE, "generic_flips_g%s_d%g.json" % (N, DELTA_SMALL))
json.dump(dict(N=N, DELTA_SMALL=DELTA_SMALL, DELTA_BIG=DELTA_BIG, MARGIN=MARGIN, log=[list(map(str, l)) for l in log], rows=rows), open(out, "w"))
print("->", out)
