#!/usr/bin/env python3
"""
The hat's chiral term as a quantum operator (open problem 2 of quantgravbits.tex; ROADMAP Stage 9d).

chi4 = sum over helical paths a-b-c-d (three edges from a vertex of the icosahedron to its antipode) of
h n_a n_b n_c n_d, h = +1 / -1 for a left / right turn at b, is the simplest mirror-odd function of the 12 ring bits
(hat_chirality.py). Here it is made an operator:

 1  on the 132 singlets:  Pi chi4 Pi = (1/15) sum_paths h [ (ab)(cd) + (ac)(bd) + (ad)(bc) ],  (pq) = S_p.S_q ;
    all terms of lower order cancel in the signed sum.
 2  the part that survives on edges only,  X = sum_paths h (S_a.S_b)(S_c.S_d)  (two disjoint edges, they commute), has a
    level-k form  X_k = sum_paths h (1/4 - e_ab/d)(1/4 - e_cd/d)  in the pairing basis of qdeform_punctures.py.
 3  symmetry: R (a reflection of the icosahedron) sends X -> -X and A -> 1/A; K (complex conjugation) sends X -> X and
    A -> 1/A. Hence the spectrum of H_k + lam X_k is the same at +lam and -lam for every level, <X_k> = 0 in the ground
    state, and with the scalar chirality C = sum_faces S_i.(S_j x S_k) (odd under R and K) the energy of
    H + lam X + mu C is even in lam and in mu separately: no term lam*mu. Checked numerically.
 4  at level 2, with Gamma_pq = 2 e_pq/d - 1 = i s_pq g_p g_q:  X_2 = (1/4) sum_paths h Gamma_ab Gamma_cd, a
    four-Majorana interaction: the free model H_2 cannot see the hat's hand, an interaction can.

Usage: python3 chi4_operator.py [--tests]
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, [sys.argv[0], "--tests"]
import qdeform_punctures as QD  # noqa: E402
import puncture_hilbert as PH  # noqa: E402
sys.argv = _argv
N = 12


def helical_paths(P, edges):
    """Ordered paths (a, b, c, d) with handedness h; each vertex set appears twice (once from each end), same h."""
    adj = {i: set() for i in range(N)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    out = {}
    for a in range(N):
        d = int(np.argmin(P @ P[a]))
        for b in adj[a]:
            for c in adj[b] & adj[d]:
                h = 1 if np.linalg.det(np.array([P[a], P[b], P[c]])) > 0 else -1
                key = frozenset((a, b, c, d))
                if key in out:
                    assert out[key][1] == h
                else:
                    out[key] = ((a, b, c, d), h)
    return list(out.values())


def reflection_perm(P):
    R = np.eye(3) - 2 * np.outer(P[0] - P[1], P[0] - P[1]) / ((P[0] - P[1]) @ (P[0] - P[1]))   # mirror swapping vertices 0 and 1
    perm = [int(np.argmax(P @ (R @ p))) for p in P]
    assert sorted(perm) == list(range(N))
    return perm


class Spin:
    """Operators on the 132 singlets of 12 spins, as 132 x 132 real matrices in an orthonormal basis."""

    def __init__(self):
        self.B = PH.singlet_space(N)
        D = 1 << N
        s = np.arange(D)
        self.bits = (s[:, None] >> np.arange(N)[None, :]) & 1
        self.SS = {}
        for i, j in itertools.combinations(range(N), 2):
            t = np.where(self.bits[:, i] == self.bits[:, j], s, s ^ (1 << i) ^ (1 << j))
            M = self.B.T @ self.B[t]                    # swap in the singlet basis
            self.SS[(i, j)] = self.SS[(j, i)] = M / 2 - np.eye(self.B.shape[1]) / 4

    def diag(self, f):
        return self.B.T @ (f[:, None] * self.B)

    def perm_op(self, perm):
        t = PH.perm_on_states(N, perm)
        PB = np.zeros_like(self.B)
        PB[t] = self.B
        return self.B.T @ PB


def spin_checks():
    V, E, F = QD.icosa()
    paths = helical_paths(V, E)
    assert len(paths) == 60 and sum(h for _, h in paths) == 0
    sp = Spin()
    n = sp.bits.astype(float)                           # bit = 1 : ring flipped
    chi = np.zeros(1 << N)
    for (a, b, c, d), h in paths:
        chi += h * n[:, a] * n[:, b] * n[:, c] * n[:, d]
    PchiP = sp.diag(chi)
    S = sp.SS
    full = sum(h * (S[(a, b)] @ S[(c, d)] + S[(a, c)] @ S[(b, d)] + S[(a, d)] @ S[(b, c)]) for (a, b, c, d), h in paths) / 15
    X = sum(h * S[(a, b)] @ S[(c, d)] for (a, b, c, d), h in paths)
    H = sum(S[e] for e in E)
    C = np.zeros((132, 132), dtype=complex)
    for i, j, k in F:
        if np.cross(V[j] - V[i], V[k] - V[i]) @ (V[i] + V[j] + V[k]) < 0:
            j, k = k, j
        C += 1j * (S[(i, j)] @ S[(j, k)] - S[(j, k)] @ S[(i, j)])        # S_i.(S_j x S_k)
    Rm = sp.perm_op(reflection_perm(V))
    res = {}
    res["PchiP_formula_error"] = float(np.abs(PchiP - full).max())
    res["PchiP_range"] = [float(x) for x in np.linalg.eigvalsh(PchiP)[[0, -1]]]
    res["X_range"] = [float(x) for x in np.linalg.eigvalsh((X + X.T) / 2)[[0, -1]]]
    res["X_symmetric_error"] = float(np.abs(X - X.T).max())
    res["R_X_R_plus_X"] = float(np.abs(Rm @ X @ Rm.T + X).max())
    res["R_chi_R_plus_chi"] = float(np.abs(Rm @ PchiP @ Rm.T + PchiP).max())
    res["R_H_R_minus_H"] = float(np.abs(Rm @ H @ Rm.T - H).max())
    res["R_C_R_plus_C"] = float(np.abs(Rm @ C @ Rm.T + C).max())
    res["C_hermitian_imaginary"] = float(max(np.abs(C - C.conj().T).max(), np.abs(C.real).max()))
    res["C_range"] = [float(x) for x in np.linalg.eigvalsh(C)[[0, -1]]]
    w, U = np.linalg.eigh(H)
    g = U[:, 0]
    res["ground_expect_X"] = float(g @ X @ g)
    res["ground_expect_chi4"] = float(g @ PchiP @ g)
    res["ground_expect_C"] = float(abs(g @ C @ g))
    grid = {}
    for lam in (-0.6, -0.3, 0.0, 0.3, 0.6):
        for mu in (-0.4, 0.0, 0.4):
            grid[(lam, mu)] = float(np.linalg.eigvalsh(H + lam * X + mu * C)[0])
    res["E0_even_in_lambda"] = max(abs(grid[(l, m)] - grid[(-l, m)]) for l, m in grid)
    res["E0_even_in_mu"] = max(abs(grid[(l, m)] - grid[(l, -m)]) for l, m in grid)
    res["E0_grid"] = {"lam=%g mu=%g" % k: v for k, v in grid.items()}
    res["mixed_second_derivative"] = float((grid[(0.3, 0.4)] - grid[(0.3, -0.4)] - grid[(-0.3, 0.4)] + grid[(-0.3, -0.4)]) / (4 * 0.3 * 0.4))
    res["d2E_dlam2"] = float((grid[(0.3, 0.0)] - 2 * grid[(0.0, 0.0)] + grid[(-0.3, 0.0)]) / 0.09)
    res["d2E_dmu2"] = float((grid[(0.0, 0.4)] - 2 * grid[(0.0, 0.0)] + grid[(0.0, -0.4)]) / 0.16)
    # response of the chirality to the hat term: <C> in the ground state of H + lam X + mu C
    def expC(lam, mu):
        w, U = np.linalg.eigh(H + lam * X + mu * C)
        return float((U[:, 0].conj() @ C @ U[:, 0]).real)
    res["expect_C"] = {"lam=%g mu=%g" % (l, m): expC(l, m) for l in (-0.6, 0.0, 0.6) for m in (0.0, 0.4)}
    return res, np.sort(np.linalg.eigvalsh((X + X.T) / 2))


def level_checks(levels=(2, 3, 6, 20)):
    V, E, F = QD.icosa()
    paths = helical_paths(V, E)
    cyc = QD.hamiltonian_cycles(E, 2)[0]
    ls = QD.left_side(V, cyc, E, F)
    out = {}
    for k in list(levels) + [None]:
        row = {}
        spec = {}
        for conj in (False, True):
            A = 1j if k is None else QD.A_of_level(k)
            m = QD.Model(1 / A if conj else A, cyc, E, F, over_side=ls)
            G = m.G
            w, U = np.linalg.eigh(G)
            Q = U[:, np.abs(w) > 1e-8 * np.abs(w).max()]
            red = lambda M: np.linalg.solve(Q.T @ G @ Q, Q.T @ G @ M @ Q)
            I = np.eye(QD.DIM)
            e = {}
            for a, b in E:
                e[(a, b)] = e[(b, a)] = m.edge_op(a, b)
            Pp = {key: 0.25 * I - val / m.d for key, val in e.items()}
            X = red(sum(h * Pp[(a, b)] @ Pp[(c, d)] for (a, b, c, d), h in paths))
            H = red(m.hamiltonian())
            for lam in (-0.5, 0.0, 0.5):
                ev = np.linalg.eigvals(H + lam * X)
                spec[(conj, lam)] = np.sort(ev.real)
                row["max_imag"] = max(row.get("max_imag", 0.0), float(np.abs(ev.imag).max()))
            if not conj:
                ev, U2 = np.linalg.eig(H)
                g = U2[:, np.argmin(ev.real)]
                Gp = Q.T @ G @ Q
                row["ground_expect_X"] = float(abs((g.conj() @ Gp @ X @ g) / (g.conj() @ Gp @ g)))
                row["X_range"] = [float(v) for v in np.sort(np.linalg.eigvals(X).real)[[0, -1]]]
                row["dim"] = int(H.shape[0])
                if k == 2:
                    Gam = {key: 2 * red(val) / m.d - np.eye(H.shape[0]) for key, val in e.items()}
                    X4 = sum(h * Gam[(a, b)] @ Gam[(c, d)] for (a, b, c, d), h in paths) / 4
                    row["X2_minus_quartic_majorana"] = float(np.abs(X - X4).max())
                    # splitting of the 15-fold first excited level by X
                    evH, UH = np.linalg.eig(H)
                    order = np.argsort(evH.real)
                    lvl = [i for i in order if abs(evH[i].real - evH[order[1]].real) < 1e-6]
                    Ul = UH[:, lvl]
                    Xl = np.linalg.solve(Ul.conj().T @ Gp @ Ul, Ul.conj().T @ Gp @ X @ Ul)
                    row["level2_first_excited_split_by_X"] = [round(float(v), 6) for v in np.sort(np.linalg.eigvals(Xl).real)]
        row["spectrum_lam_vs_minus_lam"] = float(np.abs(spec[(False, 0.5)] - spec[(False, -0.5)]).max())
        row["spectrum_A_vs_inverse_A"] = float(np.abs(spec[(False, 0.5)] - spec[(True, 0.5)]).max())
        row["E0_shift_at_lam_0.5"] = float(spec[(False, 0.5)][0] - spec[(False, 0.0)][0])
        out["classical" if k is None else "k=%d" % k] = row
    return out


def run_tests():
    res, xs = spin_checks()
    assert res["PchiP_formula_error"] < 1e-10
    print("1. Pi chi4 Pi = (1/15) sum_paths h [three pairings of S.S S.S] on the 132 singlets (error %.0e); its spectrum is [%.4f, %.4f]"
          % (res["PchiP_formula_error"], *res["PchiP_range"]))
    assert max(res["R_X_R_plus_X"], res["R_chi_R_plus_chi"], res["R_H_R_minus_H"], res["R_C_R_plus_C"], res["C_hermitian_imaginary"]) < 1e-10
    assert np.abs(xs + xs[::-1]).max() < 1e-9
    print("2. a reflection sends X and chi4 to minus themselves, H to itself, C to minus itself; C is imaginary; spectrum of X is symmetric, [%.4f, %.4f]" % tuple(res["X_range"]))
    assert max(abs(res["ground_expect_X"]), abs(res["ground_expect_chi4"]), res["ground_expect_C"]) < 1e-10
    assert res["E0_even_in_lambda"] < 1e-10 and res["E0_even_in_mu"] < 1e-10 and abs(res["mixed_second_derivative"]) < 1e-9
    print("3. ground state: <X> = <chi4> = <C> = 0; E0(lam, mu) is even in lam and in mu, mixed derivative %.0e; d2E/dlam2 = %.4f, d2E/dmu2 = %.4f"
          % (res["mixed_second_derivative"], res["d2E_dlam2"], res["d2E_dmu2"]))
    lv = level_checks()
    for name, row in lv.items():
        assert row["spectrum_lam_vs_minus_lam"] < 1e-7 and row["spectrum_A_vs_inverse_A"] < 1e-7 and row["ground_expect_X"] < 1e-8 and row["max_imag"] < 1e-7, (name, row)
    assert lv["k=2"]["X2_minus_quartic_majorana"] < 1e-10
    assert abs(lv["classical"]["X_range"][1] - res["X_range"][1]) < 1e-8
    print("4. levels k = 2, 3, 6, 20 and classical: spectrum of H_k + lam X_k equal at +lam and -lam, and at A and 1/A; <X_k> = 0 in the ground state")
    print("5. level 2: X_2 = (1/4) sum_paths h Gamma_ab Gamma_cd exactly (a four-Majorana term); it splits the 15-fold level into", lv["k=2"]["level2_first_excited_split_by_X"])
    print("all tests passed")
    return res, lv


if __name__ == "__main__":
    res, lv = run_tests()
    if "--tests" not in sys.argv:
        json.dump(dict(spin=res, levels=lv), open(os.path.join(HERE, "chi4_operator.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k not in ("E0_grid",)}, indent=1))
        for name, row in lv.items():
            print(name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
        print("-> chi4_operator.json")
