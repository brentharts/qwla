#!/usr/bin/env python3
"""
Pair events of the phason-flip engine (wada-coudene `figures/spectre_flip_moves.py`).

Input:  the engine's files for one run, /tmp/PL_c{a}_r{R2}_p{PBOX}.pkl (the placements around the probe tile) and
        /tmp/flip_states.npz (present sets before and after each phase step), and the engine's own set-up code,
        which is executed from its source (up to the construction of the lattice candidates) so that nothing is copied.
Output: flip_pairs_c{a}_d{DELTA}.json   one record per phase step whose region is a valid tiling before and after
        (the engine's `region_valid`): the bump-and-dent pairs of the region before and after, by tile identity.

A tile is identified by (placement index, leaf): a Mystic window is two leaves. A pair is the unordered pair of its
two tile ids. The region is the disc INNER of the engine (probe tile footprint buffered by R2 - DIAM, 8.5 at
R2 = 14, DIAM = 5.5): a pair is counted when its flat vertex lies in INNER, because every tile that could meet
that vertex is then in the placement list.

Usage:  SPECTRE_DIR=.. WADA_DIR=.. PBOX=20 R2=14 DIAM=5.5 DELTA=0.03 python3 flip_pairs.py [class]
"""
import io
import contextlib
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WADA = os.environ.get("WADA_DIR", "/home/claude/repos/wada-coudene")
SP = os.environ.get("SPECTRE_DIR", "/home/claude/repos/spectre")
A_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 0
R2 = float(os.environ.get("R2", "1.0"))
DIAM = float(os.environ.get("DIAM", "0"))
DELTA = float(os.environ.get("DELTA", "0.03"))
PBOX = float(os.environ.get("PBOX", "20"))

# ---- the engine's set-up, from its source -------------------------------------------------------
src = open(os.path.join(WADA, "figures", "spectre_flip_moves.py")).read()
head = src.split("\nPH = np.array(")[0]
sys.argv = ["spectre_flip_moves.py", "3", str(A_CLASS)]
env = {"__file__": os.path.join(WADA, "figures", "spectre_flip_moves.py"), "__name__": "engine_setup"}
os.environ["SPECTRE_DIR"] = SP
sys.path.insert(0, WADA)
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(head, "spectre_flip_moves_head", "exec"), env)
FR, STAR = env["FR"], env["STAR"]
from spectre_exact import SPECTRE, mul, apply, ttrans, trot, IDENT, to_c, det_sign   # noqa: E402

sys.path.insert(0, HERE)
from flip_rule import find_pairs_keys, centroid_xy, ordered_centroids          # noqa: E402
from quantumspectremodel import exact_position                                  # noqa: E402
from pair_hamiltonian import flow_b                                             # noqa: E402

from shapely.geometry import Polygon, Point                                     # noqa: E402
from shapely.ops import unary_union                                             # noqa: E402

LEAVES = {"O": [IDENT], "G": [IDENT, mul(ttrans(SPECTRE[8]), trot(1))]}
TAU = float(os.environ.get("TAU", "0.05"))
B_TAU = flow_b(TAU)

PL = pickle.load(open("/tmp/PL_c%d_r%g_p%g.pkl" % (A_CLASS, R2, PBOX), "rb"))
Z = np.load("/tmp/flip_states.npz")
P0, P1, usable, changed = Z["P0"], Z["P1"], Z["usable"], Z["changed"]
polys = [p[5] for p in PL]
A1 = FR[A_CLASS][1]
S1 = FR[A_CLASS][0]
P1poly = env["shape"](S1, A1, np.zeros(4, dtype=object)) if "shape" in env else None
if P1poly is None:
    LEAF1 = LEAVES[S1]
    P1poly = unary_union([Polygon([(to_c(apply(mul((A1, np.zeros(4, dtype=object)), L), v)).real,
                                    to_c(apply(mul((A1, np.zeros(4, dtype=object)), L), v)).imag) for v in SPECTRE]) for L in LEAF1])
INNER = P1poly.buffer(R2 - DIAM)


def leaves_of(k):
    b, Sb = PL[k][0], PL[k][1]
    w = np.array([int(x) for x in PL[k][4]], dtype=object)
    out = []
    for li, L in enumerate(LEAVES[Sb]):
        T = mul((FR[b][1], w), L)
        keys = [tuple(int(x) for x in apply(T, v)) for v in SPECTRE]
        out.append(dict(id=(int(k), li), keys=keys, refl=det_sign(T) < 0, shape=Sb))
    return out


_cache = {}


def leaves(k):
    if k not in _cache:
        _cache[k] = leaves_of(k)
    return _cache[k]


def pos(key, b=1.0):
    return np.array(exact_position(key, b))


def centroid_b(keys, b):
    return centroid_xy(np.array([exact_position(k, b) for k in keys]))


def region_valid(pres):
    pol = [polys[k] for k in np.where(pres)[0] if polys[k].intersects(INNER) and polys[k].intersection(INNER).area > 1e-9]
    ov = sum(pol[x].intersection(pol[y]).area for x in range(len(pol)) for y in range(x + 1, len(pol)) if pol[x].intersects(pol[y]))
    return ov < 1e-6 and INNER.area - unary_union(pol).intersection(INNER).area < 1e-6


def sample_pairs(pres):
    """Pairs of the region for a present set: list of dict(id, c1, ctau) with c = (cA, cB) centroids at b = 1 and b(tau)."""
    ks = [k for k in np.where(pres)[0] if polys[k].intersects(INNER)]
    tiles = [lf for k in ks for lf in leaves(k)]
    pr = find_pairs_keys([t["keys"] for t in tiles])
    out = []
    for i, j in pr:
        ti, tj = tiles[i], tiles[j]
        if not INNER.contains(Point(*pos(ti["keys"][10]))):
            continue
        c1 = ordered_centroids(centroid_b(ti["keys"], 1.0), centroid_b(tj["keys"], 1.0))
        ct = ordered_centroids(centroid_b(ti["keys"], B_TAU), centroid_b(tj["keys"], B_TAU))
        ids = sorted([ti["id"], tj["id"]])
        out.append(dict(refl=[bool(ti["refl"]), bool(tj["refl"])], shape=[ti["shape"], tj["shape"]], id=[list(ids[0]), list(ids[1])], c1=[c1[0].tolist(), c1[1].tolist()], ctau=[ct[0].tolist(), ct[1].tolist()]))
    # tile statistics in the region: leaves whose centroid lies in INNER
    nleaf = nrefl = nO = nG = 0
    for t in tiles:
        c = centroid_b(t["keys"], 1.0)
        if INNER.contains(Point(*c)):
            nleaf += 1
            nrefl += int(t["refl"])
            nG += int(t["shape"] == "G")
            nO += int(t["shape"] == "O")
    return out, dict(n_leaf=nleaf, n_refl=nrefl, n_single=nO, n_mystic_leaf=nG)


def main():
    rng = np.random.default_rng(5)
    idx_changed = np.where(changed)[0]
    idx_ctrl = rng.choice(np.where(usable & ~changed)[0], min(40, int((usable & ~changed).sum())), replace=False)
    recs = []
    for label, idx in (("changed", idx_changed), ("unchanged", idx_ctrl)):
        for i in idx:
            if not (region_valid(P0[i]) and region_valid(P1[i])):
                continue
            p0, s0 = sample_pairs(P0[i])
            p1, s1 = sample_pairs(P1[i])
            gone = [lf for k in np.where(P0[i] & ~P1[i])[0] for lf in leaves(k)]
            new = [lf for k in np.where(~P0[i] & P1[i])[0] for lf in leaves(k)]
            comp = lambda L: [[t["shape"], bool(t["refl"])] for t in L]
            recs.append(dict(sample=int(i), label=label, pairs0=p0, pairs1=p1, stats0=s0, stats1=s1, gone=comp(gone), new=comp(new),
                             n_tiles_changed=int((P0[i] != P1[i]).sum())))
    out = os.path.join(HERE, "flip_pairs_c%d_d%g.json" % (A_CLASS, DELTA))
    json.dump(dict(R2=R2, DIAM=DIAM, DELTA=DELTA, PBOX=PBOX, TAU=TAU, b_tau=B_TAU, inner_area=INNER.area, records=recs), open(out, "w"))
    print("records", len(recs), "->", out)


if __name__ == "__main__":
    main()
