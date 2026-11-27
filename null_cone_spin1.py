#!/usr/bin/env python3
"""Spin-1 version of the corner-state checks in NULL_CONE_CONNECTIONS.md (sections 3A and 3C).

(a) Singlet-sector dimensions for 12 punctures with mixed spins 1/2 and 1 (exact, by weight counting).
(b) Heisenberg ground state of 12 spin-1 punctures on the 30 icosahedron edges, restricted to total spin 0
    (penalty on S_tot^2), and the same entropy fit used for spin-1/2: S ~ a*(edges cut) + b*(region size) + c.
"""
import itertools, json, math, os, sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hopf_sites as HS

N = 12


def singlets(spins2):
    """Number of total-spin-0 states for punctures of spin spins2[i]/2: weight count m=0 minus m=1."""
    w = np.array([1], dtype=object)            # index = total 2m + offset
    off = 0
    for s2 in spins2:
        poly = np.ones(s2 + 1, dtype=object)  # weights 2m = -s2, -s2+2, ..., s2
        w = np.convolve(w, poly); off += s2
    # weights 2m = -off + 2*index ; m=0 -> index off/2 ; m=1 -> index off/2+1
    c0 = w[off // 2] if off % 2 == 0 else 0
    c1 = w[off // 2 + 1] if off % 2 == 0 and off // 2 + 1 < len(w) else 0
    return int(c0 - c1)


def spin1_ops():
    s = 1 / math.sqrt(2)
    Sz = np.diag([1., 0., -1.]); Sp = np.array([[0, s * 2 ** .5, 0], [0, 0, s * 2 ** .5], [0, 0, 0]])
    # Sp = sqrt(2) * ((0,1,0),(0,0,1),(0,0,0)) for spin 1
    Sp = math.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    Sx = (Sp + Sp.T) / 2; Sy = (Sp - Sp.T) / 2j
    return Sx, Sy, Sz


def site_op(op, i):
    left = sp.identity(3 ** i, format="csr"); right = sp.identity(3 ** (N - i - 1), format="csr")
    return sp.kron(sp.kron(left, sp.csr_matrix(op), format="csr"), right, format="csr")


def build(edges):
    ops = [[site_op(o, i) for o in spin1_ops()] for i in range(N)]
    H = sum(ops[i][a] @ ops[j][a] for i, j in edges for a in range(3))
    St = [sum(ops[i][a] for i in range(N)) for a in range(3)]
    S2 = sum(St[a] @ St[a] for a in range(3))
    return H, S2


def entropy_table(psi, edges):
    T = psi.reshape([3] * N)
    rows = []
    for m in range(1, N // 2 + 1):
        for R in itertools.combinations(range(N), m):
            if m == N // 2 and 0 not in R:
                continue
            rest = [a for a in range(N) if a not in R]
            M = np.transpose(T, list(R) + rest).reshape(3 ** m, -1)
            s = np.linalg.svd(M, compute_uv=False) ** 2
            s = s[s > 1e-14]
            Rs = set(R)
            rows.append((m, sum(1 for i, j in edges if (i in Rs) != (j in Rs)), float(-(s * np.log(s)).sum())))
    return rows


def fit(tab):
    y = np.array([t[2] for t in tab])
    A = np.array([[c, m, 1.0] for m, c, _ in tab]); A1 = np.array([[m, 1.0] for m, c, _ in tab])
    co = np.linalg.lstsq(A, y, rcond=None)[0]; co1 = np.linalg.lstsq(A1, y, rcond=None)[0]
    tot = ((y - y.mean()) ** 2).sum()
    return dict(per_cut_edge=float(co[0]), per_puncture=float(co[1]), const=float(co[2]),
                R2_cut_and_size=float(1 - ((A @ co - y) ** 2).sum() / tot), R2_size_only=float(1 - ((A1 @ co1 - y) ** 2).sum() / tot),
                regions=len(tab))


def main():
    out = {}
    mixed = {}
    for a in range(0, N + 1):                      # a = number of spin-1 punctures
        spins = [2] * a + [1] * (N - a)
        d = singlets(spins)
        mixed[a] = dict(dim_one_assignment=d, assignments=math.comb(N, a), total=d * math.comb(N, a))
    out["mixed_spin_sectors"] = mixed
    out["all_spin1_dim"] = singlets([2] * N)
    out["all_spinhalf_dim"] = singlets([1] * N)
    P = HS.icosahedron(); _, cls = HS.classes(P)
    edges = [e for e, c in cls.items() if c == "edge"]
    H, S2 = build(edges)
    w, V = eigsh((H + 4.0 * S2).tocsr(), k=2, which="SA")
    psi = V[:, 0] / np.linalg.norm(V[:, 0])
    s2 = float(np.real(np.vdot(psi, S2 @ psi))); e0 = float(np.real(np.vdot(psi, H @ psi)))
    out["ground"] = dict(E=e0, S_tot_sq=s2, gap_penalised=float(w[1] - w[0]))
    tab = entropy_table(psi, edges)
    out["entropy_fit_spin1"] = fit(tab)
    out["max_entropy_half"] = max(t[2] for t in tab if t[0] == 6); out["ln_dim"] = math.log(out["all_spin1_dim"]); out["6ln3"] = 6 * math.log(3)
    json.dump(out, open("null_cone_spin1.json", "w"), indent=1)
    return out


def tests():
    assert singlets([1] * 12) == 132 and singlets([1] * 4) == 2 and singlets([2, 2]) == 1 and singlets([2] * 3) == 1
    assert singlets([2] * 4) == 3 and singlets([2, 1, 1]) == 1
    print("null_cone_spin1: tests passed")


if __name__ == "__main__":
    tests()
    if "--tests" not in sys.argv:
        o = main(); print(json.dumps(o, indent=1))
