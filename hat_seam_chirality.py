#!/usr/bin/env python3
"""
Does the geometry of the seams see the 4-ring term chi4?  (ROADMAP Stage 9c, follow-up; see HAT_SEAM_CHIRALITY.md)

hat_chirality.py found that the hand of the hat first shows in the ring bits as chi4 (order 4, helical paths). Here the
tilings are turned back into geometry: for a ring pattern b, N(b) is the seam network of the tiling, the outline of
every hat plus the 12 pentagon loops, as points on the unit sphere. Two measures of how far a network is from being
mirror symmetric:

  M(b)  mean nearest-point distance (degrees) from N(b) to its own mirror image, minimised over rotations;
  D(b)  the same between N(b) and the mirror image of N(s.b), s the icosahedron reflection of hat_chirality.py.
        D = 0 would mean the tiling of s.b is exactly the mirror image of the tiling of b. D is even under
        b <-> s.b, like |chi4|. For an achiral pattern N(s.b) is a rotation of N(b) and D = M.

Outputs: M of the standard tiling, and for random patterns the Spearman correlation of M and D with |chi4| and with
the number of flipped rings. Needs brentharts/hexasphere (set HEXASPHERE_DIR).
Usage: python3 hat_seam_chirality.py [n_patterns=120] [--tests]       (about 0.8 s per pattern)
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import ring_bits as RB  # noqa: E402
import hat_chirality as HC  # noqa: E402
sys.argv = _argv


def densify(poly, step=0.01):
    """Closed polyline of unit vectors, great-arc interpolated every ~step radians."""
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        m = max(1, int(np.ceil(np.arccos(np.clip(a @ b, -1, 1)) / step)))
        t = (np.arange(m) / m)[:, None]
        q = (1 - t) * a + t * b
        out.append(q / np.linalg.norm(q, axis=1, keepdims=True))
    return np.vstack(out)


def network(ball, fl, T):
    pts = [densify(ball.polygon(ball._outline(sorted(fl.placements[i]))[0])) for i in T]
    pts += [densify(ball.polygon(ball.pentagon_loop(p))) for p in ball.pent]
    P = np.vstack(pts)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def mirror_distance(A, B, starts=8, sub=1200, seed=0):
    """min over rotations of the mean nearest-point distance (degrees) from R.mirror(B) to A."""
    rng = np.random.default_rng(seed)
    kd = cKDTree(A)
    Bs = B[rng.choice(len(B), min(sub, len(B)), replace=False)] * np.array([1, 1, -1])
    cost = lambda v: kd.query(Bs @ Rotation.from_rotvec(v).as_matrix().T)[0].mean()
    best = min(minimize(cost, Rotation.random(random_state=rng.integers(1 << 31)).as_rotvec(), method="Nelder-Mead",
                        options=dict(xatol=1e-3, fatol=1e-6, maxiter=300)).fun for _ in range(starts))
    return float(np.degrees(best))


def main(npat=120, tests=False):
    ball, fl, T0, rings, P, edges = RB.setup()
    ball.relax()
    paths = HC.helical_paths(P, edges)
    perms = RB.rotation_perms(ball, fl, T0, rings, P)
    g = np.asarray(ball.G[1])
    refl = [int(np.argmax(P @ (-(g @ p)))) for p in P]
    act = lambda b, perm: tuple(b[perm.index(i)] for i in range(12))        # b2[perm[i]] = b[i]
    std = network(ball, fl, T0)
    res = {"standard_tiling_M_deg": mirror_distance(std, std)}
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(npat):
        b = tuple(int(x) for x in rng.integers(0, 2, 12))
        sb = act(b, refl)
        chiral = sb not in {act(b, p) for p in perms}
        Nb = network(ball, fl, RB.tiling_of(b, T0, rings))
        Ns = Nb if sb == b else network(ball, fl, RB.tiling_of(sb, T0, rings))
        rows.append(dict(bits=b, chi4=HC.chi4(b, paths), chi4_refl=HC.chi4(sb, paths), flipped=sum(b), chiral=chiral,
                         M=mirror_distance(Nb, Nb), D=mirror_distance(Nb, Ns)))
    c = np.abs([r["chi4"] for r in rows]); f = [r["flipped"] for r in rows]
    for key in ("M", "D"):
        v = [r[key] for r in rows]
        res[key] = dict(mean=float(np.mean(v)), vs_abs_chi4=[float(x) for x in spearmanr(c, v)], vs_flipped=[float(x) for x in spearmanr(f, v)])
    res["chi4_nonzero_on_achiral"] = int(sum(1 for r in rows if (r["chi4"] != 0) and not r["chiral"]))
    res["chi4_odd_under_reflection"] = all(r["chi4_refl"] == -r["chi4"] for r in rows)
    print(json.dumps(res, indent=1))
    if tests:
        assert res["chi4_odd_under_reflection"] and res["chi4_nonzero_on_achiral"] == 0
        assert 0.5 < res["standard_tiling_M_deg"] < 1.5
        print("hat_seam_chirality: all tests passed")
    json.dump(res, open(os.path.join(HERE, "hat_seam_chirality.json"), "w"), indent=1)


if __name__ == "__main__":
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    main(int(nums[0]) if nums else 120, "--tests" in sys.argv)
