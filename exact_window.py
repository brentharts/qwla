"""Membership in the Spectre windows Z_0(S), by the exact graph-directed recursion instead of a point cloud.

Z_p(S) = U over (i, S') in MAPS[S p] of  (GA[i,p] + Z_{p+1}(S') / lam) @ O[i,p]   (row vectors; wada-coudene/lakes_of_wada/spectre_window.py).
A point z lies in Z_p(S) iff for some (i, S') the preimage z' = lam (z O^-1 - GA) lies in Z_{p+1}(S'); every window lies in the disc of radius RAD.
Descending the recursion and dropping branches with |z'| > RAD gives: a point whose branches all die by depth KA is outside; a point with a
surviving branch at depth KB is inside up to a band of width about RAD lam^-KB around the window; between the two it is ambiguous.
The code is 1 (in), 0 (out), 2 (ambiguous). With KA=9, KB=13 the ambiguous band is about 1e-4 wide, against 3.5e-3 for the depth-7 point cloud.
"""
import os
import numpy as np

HERE = os.environ.get("WADA_DIR", "/home/claude/repos/wada-coudene")
_d = np.load(os.path.join(HERE, "figures", "spectre_window_maps.npz"))
LAM = float(_d["lam"]); RAD_W = float(_d["rad"])
GA, OO = _d["GA"], _d["O"]
OINV = np.linalg.inv(OO)                      # (8, 2, 2, 2)
MAPS = {(S, p): _d[f"{S}{p}"] for S in "OG" for p in (0, 1)}


def classify(S, Z, KA=9, KB=13):
    """Codes (1 in, 0 out, 2 ambiguous) of the points Z (n, 2) against Z_0(S)."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    n = len(Z)
    alive_A = np.zeros(n, bool); alive_B = np.zeros(n, bool)
    idx = np.arange(n); z = Z.copy(); Ss = np.full(n, 0 if S == "O" else 1); p = 0
    for depth in range(0, KB + 1):
        if depth == KA:
            alive_A[idx] = True
        if depth == KB:
            alive_B[idx] = True
            break
        ni, nz, nS = [], [], []
        for si, Sc in enumerate("OG"):
            m = Ss == si
            if not m.any():
                continue
            for i, s2 in MAPS[(Sc, p)]:
                zz = LAM * (z[m] @ OINV[i, p] - GA[i, p])
                ok = (zz * zz).sum(1) <= RAD_W * RAD_W
                if ok.any():
                    ni.append(idx[m][ok]); nz.append(zz[ok]); nS.append(np.full(ok.sum(), s2))
        if not ni:
            break
        idx = np.concatenate(ni); z = np.concatenate(nz); Ss = np.concatenate(nS); p = 1 - p
        if len(idx) > 400 * max(n, 1):                      # safety: heavy branching only at the boundary
            u = np.unique(idx); keep = np.isin(idx, u[:0])  # not expected
    out = np.full(n, 2, dtype=int)
    out[alive_B] = 1
    out[~alive_A] = 0
    return out


class _Tree:
    """Stand-in for cKDTree: query(z) returns a pseudo distance 0 (in), 1 (out), 0.0025 (ambiguous) for IN_T 0.0015 and OUT_T 0.0035."""
    def __init__(self, S):
        self.S = S

    def query(self, z):
        single = np.ndim(z) == 1
        c = classify(self.S, z)
        d = np.where(c == 1, 0.0, np.where(c == 0, 1.0, 0.0025))
        return (d[0] if single else d), None
