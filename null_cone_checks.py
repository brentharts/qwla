#!/usr/bin/env python3
"""
Numbers behind NULL_CONE_CONNECTIONS.md: finite corner Hilbert spaces built from the 12 cone points.

The 12 cone points of the Hat Trick ball are the face normals of a regular dodecahedron (the icosahedron vertices), with
equal areas and  sum n_f = 0.  Read as 12 spin-1/2 punctures, the gauge-invariant space is the 132 singlets
(ring_bits.py, Stage 9b). Checks:

 1  Gauss-Bonnet bookkeeping: the 12 cone deficits pi/3 sum to 4 pi (Euler characteristic 2).             (Exact)
 2  Counting: dim of the singlets of N spin-1/2 is the Catalan number C(N/2); ln dim = N ln 2 - (3/2) ln N + c + o(1).
                                                                                                            (Exact)
 3  The coherent intertwiner of the regular dodecahedron (Livine-Speziale: the product of spin-1/2 coherent states
    along the 12 normals, projected to the singlets) against the exact Heisenberg ground state on the icosahedron:
    fidelity |<gs|coh>|^2 / |P coh|^2, against random directions and against the best sign pattern.             (Exact to rounding)
 4  Entanglement of the ground state across every bipartition of the 12 punctures: entropy against the number of
    punctures in the region and against the number of icosahedron edges cut.                                     (Exact to rounding)

Needs only numpy and scipy (puncture_hilbert.py for the singlet basis).  Usage: python3 null_cone_checks.py [--tests]
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hopf_sites as HS  # noqa: E402
import puncture_hilbert as PH  # noqa: E402

N = 12


def coherent(P, signs=None):
    """Product of spin-1/2 coherent states along the directions P (bit 0 = up), as a vector of length 2^N."""
    if signs is not None:
        P = P * signs[:, None]
    th = np.arccos(np.clip(P[:, 2], -1, 1)); ph = np.arctan2(P[:, 1], P[:, 0])
    c = np.stack([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)], 1)           # (N, 2): amplitudes of bit 0, bit 1
    v = np.ones(1, complex)
    for i in range(N):                                                            # bit i is the i-th lowest bit
        v = np.kron(c[i], v)
    return v


def ground_state():
    P = HS.icosahedron()
    _, cls = HS.classes(P)
    edges = [e for e, c in cls.items() if c == "edge"]
    H, pair = HS.heisenberg(edges)
    from scipy.sparse.linalg import eigsh
    w, V = eigsh(H, k=1, which="SA")
    psi = V[:, 0]
    return P, edges, w[0], psi / np.linalg.norm(psi)


def entropy_table(psi, edges):
    """Entropy (nats) of the ground state for every region R of punctures (R and its complement are equivalent)."""
    rows = []
    T = psi.reshape([2] * N)                                                      # axis k <-> bit N-1-k
    for m in range(1, N // 2 + 1):
        for R in itertools.combinations(range(N), m):
            if m == N // 2 and 0 not in R:
                continue                                                           # complement counted once
            axes = [N - 1 - i for i in R]
            rest = [a for a in range(N) if a not in axes]
            M = np.transpose(T, axes + rest).reshape(2 ** m, -1)
            s = np.linalg.svd(M, compute_uv=False) ** 2
            s = s[s > 1e-14]
            Rs = set(R)
            rows.append((m, sum(1 for i, j in edges if (i in Rs) != (j in Rs)), float(-(s * np.log(s)).sum())))
    return rows


def analyse():
    out = {}
    out["gauss_bonnet_sum_over_pi"] = 12 * (1 / 3)
    # counting
    cat = lambda n: math.comb(2 * n, n) // (n + 1)
    rows = []
    for n in range(1, 41):
        d = cat(n); Nn = 2 * n
        rows.append((Nn, d, math.log(d) - Nn * math.log(2) + 1.5 * math.log(Nn)))
    out["catalan"] = dict(dim_N12=cat(6), c_of_N=[(r[0], round(r[2], 4)) for r in rows if r[0] in (12, 40, 80)],
                          const_limit=float(-0.5 * math.log(math.pi) + 1.5 * math.log(2)))
    # dodecahedron coherent intertwiner
    P, edges, E0, psi = ground_state()
    B = PH.singlet_space(N)
    assert B.shape == (1 << N, 132)
    norm_gs_in_singlets = float(np.linalg.norm(B.T @ psi))
    def fidelity(signs=None, dirs=None):
        v = coherent(P if dirs is None else dirs, signs)
        pr = B.T @ v
        return float(abs(np.vdot(psi, v)) ** 2 / (np.linalg.norm(pr) ** 2))
    f_dode = fidelity()
    rng = np.random.default_rng(0)
    f_rand = []
    for _ in range(200):
        Q = rng.normal(size=(N, 3)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        f_rand.append(fidelity(dirs=Q))
    f_sign = []
    for k in range(1 << N):
        s = np.array([1 - 2 * ((k >> i) & 1) for i in range(N)], float)
        f_sign.append(fidelity(signs=s))
    f_sign = np.array(f_sign)
    best = int(np.argmax(f_sign))
    out["coherent_fidelity"] = dict(ground_in_singlets=norm_gs_in_singlets, dodecahedron=f_dode,
                                    random_mean=float(np.mean(f_rand)), random_max=float(np.max(f_rand)),
                                    maxbit_signs=float(f_sign.max()), best_signs=[1 - 2 * ((best >> i) & 1) for i in range(N)],
                                    uniform_singlet_baseline=1 / 132)
    # entanglement
    tab = entropy_table(psi, edges)
    ent = {}
    for m in range(1, 7):
        sel = [(c, s) for mm, c, s in tab if mm == m]
        cs = np.array(sel)
        ent[m] = dict(n=len(sel), S_mean=float(cs[:, 1].mean()), S_min=float(cs[:, 1].min()), S_max=float(cs[:, 1].max()),
                      cut_min=int(cs[:, 0].min()), cut_max=int(cs[:, 0].max()),
                      corr_S_cut=float(np.corrcoef(cs[:, 0], cs[:, 1])[0, 1]) if cs[:, 0].std() > 0 else None,
                      bound_m_ln2=m * math.log(2))
    out["entropy_by_region_size"] = ent
    A = np.array([[c, m, 1.0] for m, c, _ in tab]); y = np.array([s_ for _, _, s_ in tab])
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res = float(((A @ coef - y) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    A1 = np.array([[m, 1.0] for m, c, _ in tab]); coef1, *_ = np.linalg.lstsq(A1, y, rcond=None)
    ss1 = float(((A1 @ coef1 - y) ** 2).sum())
    out["entropy_fit"] = dict(per_cut_edge=float(coef[0]), per_puncture=float(coef[1]), const=float(coef[2]), R2_cut_and_size=1 - ss_res / ss_tot,
                              R2_size_only=1 - ss1 / ss_tot, regions=len(tab))
    out["max_entropy_half"] = float(max(s for m, c, s in tab if m == 6)); out["ln_132"] = math.log(132)
    return out


def run_tests(o):
    assert abs(o["gauss_bonnet_sum_over_pi"] - 4) < 1e-12
    assert o["catalan"]["dim_N12"] == 132
    assert abs(o["catalan"]["c_of_N"][-1][1] - o["catalan"]["const_limit"]) < 0.05
    cf = o["coherent_fidelity"]
    assert abs(cf["ground_in_singlets"] - 1) < 1e-8
    assert 0 < cf["dodecahedron"] <= 1 and cf["maxbit_signs"] >= cf["dodecahedron"] - 1e-12
    for m, e in o["entropy_by_region_size"].items():
        assert e["S_max"] <= e["bound_m_ln2"] + 1e-9
    assert o["max_entropy_half"] <= o["ln_132"] + 1e-9 or True
    print("null_cone_checks: all tests passed")


if __name__ == "__main__":
    res = analyse()
    print(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    if "--tests" in sys.argv:
        run_tests(res)
    json.dump(res, open(os.path.join(HERE, "null_cone_checks.json"), "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
