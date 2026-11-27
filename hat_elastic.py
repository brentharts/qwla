#!/usr/bin/env python3
"""
An elastic energy for each tiling of the Hat Trick ball, and its mirror-odd part (open problem 2 of quantgravbits.tex:
a physical origin for the coupling lambda of the hat's chiral term chi4; ROADMAP Stage 9j).

hexasphere's HatBall.relax() fits the kites to the sphere; its constraints (kites, hexagon edges, triangulation edges) do
not depend on how the kites are grouped into hats, so all 4096 tilings relax to the same geometry. Here the hats are
rigid flat plates (Definition): for every hat, every pair of its distinct corner points must keep its distance in the
flat hat (flat_hat), times one common scale a; the corner points lie on the unit sphere and are shared at the seams.
The energy of a tiling is E = sum of squared relative residuals at the least-squares minimum. A rotation of the sphere
maps the problem of bits b to that of the rotated bits, so E is a function on the 96 rotation classes; a reflection maps
it to a tiling by mirrored hats, which has the same E, but the tiling of the mirrored bits s.b is a different tiling, so
E(s.b) - E(b) is a genuine mirror-odd energy.

Usage: python3 hat_elastic.py [--tests]   mirror pairs;   python3 hat_elastic.py --even   the even part on all 96 classes\n(needs brentharts/hexasphere)
"""
import collections
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_a, sys.argv = sys.argv, [sys.argv[0]]
import ring_bits as RB  # noqa: E402
import hat_chirality as HC  # noqa: E402
sys.argv = _a
from hexasphere.hattrick.kites import embed_hat, flat_hat  # noqa: E402


class Elastic:
    def __init__(self):
        self.ball, self.fl, self.T0, self.rings, self.P, self.edges = RB.setup()
        S = self.ball.S
        self.ordered = {}
        for k in S.kites:
            p = embed_hat(S, k)
            if p:
                self.ordered.setdefault(frozenset(p), p)
        kites, _ = flat_hat()
        self.flat = kites                                    # 8 kites x 4 corners, hat order
        self.ids = {}
        for k in S.kites:
            for cid in S.corners(k):
                self.ids.setdefault(cid, len(self.ids))
        self.X0 = np.array([S.corner_position(c) for c in sorted(self.ids, key=self.ids.get)])
        self.perms = RB.rotation_perms(self.ball, self.fl, self.T0, self.rings, self.P)
        g = np.asarray(self.ball.G[1])
        self.refl = [int(np.argmax(self.P @ (-(g @ p)))) for p in self.P]

    def hat_pairs(self, placement):
        """(i, j, flat distance) for every pair of distinct corner points of one hat."""
        p = self.ordered[frozenset(placement)]
        pos = {}
        for q, k in enumerate(p):
            for s, cid in enumerate(self.ball.S.corners(k)):
                xy = self.flat[q][s]
                if cid in pos:
                    assert np.linalg.norm(pos[cid] - xy) < 1e-9, "flat layout does not match the embedding"
                pos[cid] = xy
        cids = sorted(pos, key=self.ids.get)
        return [(self.ids[a], self.ids[b], float(np.linalg.norm(pos[a] - pos[b]))) for a, b in itertools.combinations(cids, 2)]

    def energy(self, bits, max_nfev=300):
        T = RB.tiling_of(tuple(bits), self.T0, self.rings)
        rows = [r for i in T for r in self.hat_pairs(self.fl.placements[i])]
        P = np.array([(i, j) for i, j, _ in rows])
        D = np.array([d for _, _, d in rows])
        n = len(self.X0)

        def unit(z):
            Z = z.reshape(n, 3)
            return Z / np.linalg.norm(Z, axis=1)[:, None]

        def resid(z):
            X = unit(z[1:])
            return (np.linalg.norm(X[P[:, 0]] - X[P[:, 1]], axis=1) - z[0] * D) / (z[0] * D)
        d0 = np.linalg.norm(self.X0[P[:, 0]] - self.X0[P[:, 1]], axis=1)
        a0 = float((d0 / D).mean())
        z0 = np.concatenate([[a0], self.X0.ravel()])
        A = lil_matrix((len(P), 1 + 3 * n), dtype=int)
        A[:, 0] = 1
        for r, (i, j) in enumerate(P):
            A[r, 1 + 3 * i:4 + 3 * i] = 1
            A[r, 1 + 3 * j:4 + 3 * j] = 1
        sol = least_squares(resid, z0, jac_sparsity=A, method="trf", x_scale="jac", max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        r = sol.fun
        return dict(E=float(r @ r), rms=float(np.sqrt(np.mean(r ** 2))), scale=float(sol.x[0]), nfev=int(sol.nfev), status=int(sol.status), n_res=len(r))

    def act(self, b, perm):
        out = [0] * 12
        for i in range(12):
            out[perm[i]] = b[i]
        return tuple(out)


def classes(el):
    allb = list(itertools.product([0, 1], repeat=12))
    orb = {}
    for b in allb:
        if b not in orb:
            o = frozenset(el.act(b, p) for p in el.perms)
            for x in o:
                orb[x] = o
    reps = {}
    for b in allb:
        reps.setdefault(orb[b], min(orb[b]))
    return allb, orb, reps


def main(tests=False):
    t0 = time.time()
    el = Elastic()
    paths = HC.helical_paths(el.P, el.edges)
    allb, orb, reps = classes(el)
    print("rotation classes", len(reps), "%.0fs" % (time.time() - t0), flush=True)
    b0 = (0,) * 12
    e0 = el.energy(b0)
    print("all-0 tiling:", e0, "%.0fs" % (time.time() - t0), flush=True)
    rng = np.random.default_rng(0)
    b = tuple(int(x) for x in rng.integers(0, 2, 12))
    ea, eb = el.energy(b), el.energy(el.act(b, el.perms[7]))
    print("a pattern and a rotation of it: E = %.10f and %.10f" % (ea["E"], eb["E"]), flush=True)
    assert abs(ea["E"] - eb["E"]) < 1e-6 * max(ea["E"], 1e-12), (ea, eb)
    if tests:
        print("tests passed")
        return
    chiral = [c for c in reps if orb[el.act(reps[c], el.refl)] != c]
    pairs, seen = [], set()
    for c in chiral:
        if c in seen:
            continue
        m = orb[el.act(reps[c], el.refl)]
        seen |= {c, m}
        pairs.append((reps[c], reps[m]))
    print("chiral classes", len(chiral), "mirror pairs of classes", len(pairs), flush=True)
    rows = []
    for b, mb in pairs:
        e1, e2 = el.energy(b), el.energy(mb)
        x = HC.chi4(b, paths)
        rows.append(dict(bits=list(b), mirror_bits=list(mb), flipped=int(sum(b)), chi4=int(x), E=e1["E"], E_mirror=e2["E"],
                         odd=(e1["E"] - e2["E"]) / 2, even=(e1["E"] + e2["E"]) / 2, status=[e1["status"], e2["status"]]))
        print("  flipped %2d chi4 %+d  E %.6f  E(mirror) %.6f  odd part %+.3e  (%.0fs)" % (rows[-1]["flipped"], x, e1["E"], e2["E"], rows[-1]["odd"], time.time() - t0), flush=True)
    odd = np.array([r["odd"] for r in rows])
    chi = np.array([r["chi4"] for r in rows])
    ev = np.array([r["even"] for r in rows])
    res = dict(definition="rigid flat hats on the unit sphere, least squares over all corner pairs of each hat", all0=e0,
               mirror_pairs=rows, max_abs_odd=float(np.abs(odd).max()), mean_even=float(ev.mean()),
               odd_over_even=float(np.abs(odd).max() / ev.mean()))
    if (chi != 0).any():
        lam = float(odd @ chi / (chi @ chi))
        res["lambda_fit"] = lam
        res["residual_after_chi4"] = float(np.abs(odd - lam * chi).max())
    json.dump(res, open(os.path.join(HERE, "hat_elastic.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "mirror_pairs"}, indent=1))




def even_fit(el=None, out=True):
    """E on all 96 rotation classes, extended to the 4096 tilings, and its expansion in monomials of the bits."""
    el = el or Elastic()
    allb, orb, reps = classes(el)
    E = {}
    t0 = time.time()
    for c, b in reps.items():
        E[c] = el.energy(b)["E"]
    f = np.array([E[orb[b]] for b in allb])                     # allb in itertools.product order: bit 0 most significant
    a = f.reshape([2] * 12).copy()
    for ax in range(12):
        a = np.stack([np.take(a, 0, axis=ax), np.take(a, 1, axis=ax) - np.take(a, 0, axis=ax)], axis=ax)
    coef = a.reshape(-1)
    idx = np.indices([2] * 12).reshape(12, -1).T                # monomial = set of bits
    order = idx.sum(1)
    D = el.P @ el.P.T
    dist = lambda i, j: 1 if D[i, j] > 0.3 else (3 if D[i, j] < -0.99 else 2)
    shapes = collections.defaultdict(list)
    for m, c in zip(idx, coef):
        S = tuple(np.where(m)[0])
        if len(S) == 0:
            key = "constant"
        elif len(S) == 1:
            key = "one ring"
        elif len(S) == 2:
            key = "two rings at distance %d" % dist(*S)
        else:
            key = "order %d" % len(S)
        shapes[key].append(c)
    summary = {k: dict(n=len(v), mean=float(np.mean(v)), spread=float(np.max(v) - np.min(v)), max_abs=float(np.max(np.abs(v))))
               for k, v in sorted(shapes.items())}
    # truncated model: constant + one ring + pairs by distance, least squares over the 4096 tilings
    X = [np.ones(len(allb)), idx[:, :].sum(1) * 0]
    bits = np.array(allb)
    cols = {"constant": np.ones(len(allb)), "n": bits.sum(1)}
    for dd in (1, 2, 3):
        cols["pairs_d%d" % dd] = np.array([sum(b[i] * b[j] for i in range(12) for j in range(i + 1, 12) if dist(i, j) == dd) for b in bits])
    M = np.stack(list(cols.values()), 1)
    beta, *_ = np.linalg.lstsq(M, f, rcond=None)
    resid = f - M @ beta
    fit = dict(zip(cols, map(float, beta)))
    res = dict(E_all0=float(f[0]), E_all1=float(f[-1]), E_range=[float(f.min()), float(f.max())], monomials=summary, pair_model=fit,
               pair_model_max_residual=float(np.abs(resid).max()), pair_model_rms_residual=float(np.sqrt(np.mean(resid ** 2))), seconds=round(time.time() - t0))
    if out:
        json.dump(res, open(os.path.join(HERE, "hat_elastic_even.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if "--even" in sys.argv:
        even_fit()
    else:
        main(tests="--tests" in sys.argv)
