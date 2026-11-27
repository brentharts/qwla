#!/usr/bin/env python3
"""Coherent dodecahedron vs the spin-1 Heisenberg singlet ground state (NULL_CONE_CONNECTIONS.md, section 3D at spin 1).
Spin-1 coherent state along n = symmetric square of the spin-1/2 coherent state: (c0^2, sqrt2 c0 c1, c1^2) in basis (m=+1,0,-1).
Fidelity is |<gs|v>|^2 / |P_singlet v|^2, with the projection onto total spin 0 done by applying the exact penalised
ground-state solver's own singlet condition: since gs is a singlet, |<gs|v>|^2 / |P v|^2 needs |P v|; we get |P v|^2 from the
singlet projector built as a polynomial filter (S^2 eigenvalues are j(j+1), j=0..12) -- here by Lanczos-free power: see proj().
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import null_cone_spin1 as S1
import hopf_sites as HS
from scipy.sparse.linalg import eigsh

N = 12


def coh(P, signs=None):
    if signs is not None:
        P = P * signs[:, None]
    th = np.arccos(np.clip(P[:, 2], -1, 1)); ph = np.arctan2(P[:, 1], P[:, 0])
    v = np.ones(1, complex)
    for i in range(N):
        c0, c1 = np.cos(th[i] / 2), np.exp(1j * ph[i]) * np.sin(th[i] / 2)
        v = np.kron(np.array([c0 * c0, math.sqrt(2) * c0 * c1, c1 * c1]), v)   # site i is the i-th lowest "digit"
    return v


def main():
    P = HS.icosahedron(); _, cls = HS.classes(P)
    edges = [e for e, c in cls.items() if c == "edge"]
    H, S2 = S1.build(edges)
    w, V = eigsh((H + 4.0 * S2).tocsr(), k=1, which="SA")
    psi = V[:, 0] / np.linalg.norm(V[:, 0])

    # Project onto singlets by a filter product (S2 - j(j+1)) over j = 1..12 normalised at 0; S2 eigenvalues are j(j+1).
    def proj(v):
        for j in range(1, 13):
            v = (S2 @ v - j * (j + 1) * v) / (-j * (j + 1))
        return v

    def fid(dirs, signs=None):
        v = coh(dirs, signs)
        pv = proj(v)
        return float(abs(np.vdot(psi, v)) ** 2 / np.vdot(pv, pv).real)

    out = {"dodecahedron": fid(P)}
    rng = np.random.default_rng(0)
    r = []
    for _ in range(30):
        Q = rng.normal(size=(N, 3)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        r.append(fid(Q))
    out["random_mean"] = float(np.mean(r)); out["random_max"] = float(np.max(r))
    sg = [fid(P, np.array([1 - 2 * ((k >> i) & 1) for i in range(N)], float)) for k in range(0, 1 << N, 16)]
    out["signs_sampled"] = len(sg); out["signs_max"] = float(max(sg))
    out["uniform_singlet_baseline"] = 1 / 4213
    json.dump(out, open("null_cone_spin1_fidelity.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
