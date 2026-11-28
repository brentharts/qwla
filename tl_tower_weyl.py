#!/usr/bin/env python3
"""
Adding and removing a puncture: the Temperley-Lieb tower and the loosened Weyl relation.  (ROADMAP Stage 10a.)

Harper and Samuelson (arXiv:2401.02545) show that on the tower of Temperley-Lieb module categories the functors of
induction (add a strand) and restriction (remove one) satisfy   Res Ind = Ind Res (+) P_0,   P_0 an idempotent that is
not central: on Grothendieck groups   y x - x y = [P_0],   a loosening of the Weyl relation y x - x y = 1. They work at
generic q.

In this repository a strand is a spin-1/2 puncture (a cone point), TL_n acts on n punctures, and the level-k spaces are
the quotients used since Stage 9c. This script computes the same relation on Grothendieck groups from the branching
rules alone, with [n, t] the class of the module of n punctures with total charge t/2:

    Ind [n, t] = [n+1, t-1] + [n+1, t+1],      Res [n, t] = [n-1, t-1] + [n-1, t+1]        (0 <= t <= n; t <= k at level k)

 1  generic q:   Res Ind - Ind Res = projection on t = n (all punctures aligned, the Jones-Wenzl sector), for every n;
 2  level k:     the same for n < k, and zero for n >= k: once there are as many punctures as the level, adding and
                 removing a puncture commute;
 3  the closed surface (total charge 0): adding or removing a PAIR of punctures; the commutator on the singlet classes;
 4  dimensions of [12, 0] at level k: 1, 32, 89, 122, 131, 132 (the numbers of Stage 9c), from the same rules.

The branching rules are standard (Known); everything else is integer arithmetic (Exact).
Usage: python3 tl_tower_weyl.py [--tests]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = 16


def tower(k=None, nmax=NMAX):
    """Index of the classes [n, t] and the matrices of Ind and Res (acting on columns)."""
    idx = {}
    for n in range(nmax + 1):
        for t in range(n % 2, n + 1, 2):
            if k is None or t <= k:
                idx[(n, t)] = len(idx)
    N = len(idx)
    I = np.zeros((N, N), dtype=np.int64); R = np.zeros((N, N), dtype=np.int64)
    for (n, t), a in idx.items():
        for dt in (-1, 1):
            if (n + 1, t + dt) in idx:
                I[idx[(n + 1, t + dt)], a] = 1
            if (n - 1, t + dt) in idx:
                R[idx[(n - 1, t + dt)], a] = 1
    return idx, I, R


def commutator(k=None):
    idx, I, R = tower(k)
    C = R @ I - I @ R
    rows = {}
    for (n, t), a in idx.items():
        if n >= NMAX - 1:
            continue                                   # the truncation of the tower at NMAX spoils the top two rows
        col = {key: int(C[b, a]) for key, b in idx.items() if C[b, a] != 0}
        rows[(n, t)] = col
    return rows


def dims(k, n=12):
    v = {0: 1}
    for _ in range(n):
        w = {}
        for t, c in v.items():
            for dt in (-1, 1):
                if t + dt >= 0 and (k is None or t + dt <= k):
                    w[t + dt] = w.get(t + dt, 0) + c
        v = w
    return v


def main():
    out = {}
    # 1 generic
    rows = commutator(None)
    ok = all(col == ({(n, t): 1} if t == n else {}) for (n, t), col in rows.items())
    out["generic_commutator_is_top_projection"] = bool(ok)
    print("1. generic q: Res Ind - Ind Res = projection on t = n, checked for n = 0..%d: %s" % (NMAX - 2, ok))
    # 2 level k
    out["levels"] = {}
    for k in range(1, 9):
        rows = commutator(k)
        nz = sorted({n for (n, t), col in rows.items() if col})
        good = all(col == ({(n, t): 1} if (t == n and n < k) else {}) for (n, t), col in rows.items())
        out["levels"][str(k)] = dict(nonzero_for_n=nz, as_stated=bool(good))
        print("2. level %d: commutator nonzero exactly for n = %s (projection on t = n); zero for n >= %d: %s" % (k, nz, k, good))
    # 3 pairs on the closed surface
    out["pairs"] = {}
    for k in (None, 2, 3, 6):
        idx, I, R = tower(k)
        I2, R2 = I @ I, R @ R
        res = {}
        for n in range(0, 11, 2):
            a = idx[(n, 0)]
            res[n] = int((R2 @ I2 - I2 @ R2)[a, a])
        out["pairs"]["generic" if k is None else str(k)] = res
        print("3. %s: singlet coefficient of [Res^2, Ind^2] on [n, 0], n = 0, 2, .., 10: %s" % ("generic" if k is None else "level %d" % k, list(res.values())))
    # 4 dimensions
    d12 = [dims(k)[0] for k in range(1, 8)]
    out["dim_12_punctures_by_level"] = d12
    print("4. dimension of the 12-puncture singlet space at level 1..7:", d12)
    json.dump(out, open(os.path.join(HERE, "tl_tower_weyl.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        assert ok and all(v["as_stated"] for v in out["levels"].values()) and d12[:6] == [1, 32, 89, 122, 131, 132]
        print("tl_tower_weyl: tests passed")


if __name__ == "__main__":
    main()
