#!/usr/bin/env python3
"""
The 12 ring bits of the Hat Trick ball as 12 spin-1/2 punctures (ROADMAP Stage 3b / 9b).

hat_flips.py shows that the hat tilings of the ball are the 2^12 states of one bit per pentagon (cone point), and
that the vertex count of the hat complex is  V = 80 - sum_{edges <ij> of the icosahedron} n_i n_j.
Stage 9b built the gauge-invariant space of a sphere with 12 spin-1/2 punctures: the 132 SU(2) singlets of (C^2)^12.
Both live in the same 4096-dimensional space once a ring bit is read as a spin-1/2 at its cone point. Checks:

 1  the 60 rotations act on the tilings by permuting the bits (no bit is flipped): the action on (C^2)^12 is the
    permutation action of Stage 9b; number of tilings up to rotation (Burnside).
 2  the singlet space is spanned by superpositions of tilings with exactly 6 rings flipped (924 of the 4096).
 3  V does not commute with total spin; its projection to the singlets is exactly
        P V P = 72.5 - (1/3) sum_<ij> S_i.S_j      (Heisenberg model on the icosahedron)
 4  spectrum of that operator on the 132 singlets, and of the full Heisenberg sum.
 5  Hamiltonians on all 4096 states:  H = kappa * sum n_i n_j - J * sum X_i  (flip = X; the form of
    pair_hamiltonian.py and of a Rydberg blockade model) : gap and ground-state numbers for a scan of J/kappa.

What is a definition here and not a result: reading a ring bit as a spin-1/2, and asking for SU(2) invariance.
Needs brentharts/hexasphere.  Usage: python3 ring_bits.py [--tests]
"""
import collections
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import hat_flips as HF  # noqa: E402
import puncture_hilbert as PH  # noqa: E402
sys.argv = _argv

N = 12


def setup():
    ball = HF.HatBall.from_frequency(3)
    fl = HF.Flips(ball)
    T0 = fl.tiling_ids(sorted(map(sorted, ball.tilings[0])))
    f = HF.irreducible(fl.flips(T0, 5))
    rings = []
    for C, alts in f.items():
        region = frozenset().union(*[fl.placements[i] for i in C])
        p = collections.Counter(v for k in region for v in ball.F[k // 3] if v in fl.pent).most_common(1)[0][0]
        rings.append((p, C, alts[0], region))
    rings.sort(key=lambda r: r[0])
    P = np.array([ball.V[r[0]] for r in rings])
    adj = (P @ P.T > 0.3) & (P @ P.T < 0.9)
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i, j]]
    return ball, fl, T0, rings, P, edges


def tiling_of(bits, T0, rings):
    T = T0
    for b, (_, C, new, _) in zip(bits, rings):
        if b:
            T = (T - C) | new
    return T


def bits_of(T, rings):
    return tuple(int(not (C <= T)) for _, C, _, _ in rings)


def rotation_perms(ball, fl, T0, rings, P):
    """For each of the 60 rotations: the permutation of pentagons, checked to be the action on tilings."""
    out = []
    rng = np.random.default_rng(0)
    for g, kp in zip(ball.G, ball.kperm):
        Q = P @ np.asarray(g).T
        perm = [int(np.argmax(P @ q)) for q in Q]
        assert sorted(perm) == list(range(N))
        for _ in range(6):
            b = tuple(int(x) for x in rng.integers(0, 2, N))
            T = tiling_of(b, T0, rings)
            gT = frozenset(fl.pidx[frozenset(kp[k] for k in fl.placements[i])] for i in T)
            b2 = [0] * N
            for i in range(N):
                b2[perm[i]] = b[i]
            assert bits_of(gT, rings) == tuple(b2)
        out.append(perm)
    return out


def cycles(perm):
    seen, n = set(), 0
    for i in range(len(perm)):
        if i not in seen:
            n += 1
            while i not in seen:
                seen.add(i)
                i = perm[i]
    return n


def ops(edges):
    D = 1 << N
    s = np.arange(D)
    bit = lambda i: (s >> i) & 1
    nn = np.zeros(D)
    for i, j in edges:
        nn += bit(i) * bit(j)
    SS = np.zeros((D, D))                                   # sum over edges of S_i.S_j = swap/2 - 1/4
    for i, j in edges:
        t = np.where(bit(i) == bit(j), s, s ^ (1 << i) ^ (1 << j))
        SS[t, s] += 0.5
        SS[s, s] -= 0.25
    X = np.zeros((D, D))
    for i in range(N):
        X[s ^ (1 << i), s] += 1.0
    return nn, SS, X


def analyse(verbose=True):
    ball, fl, T0, rings, P, edges = setup()
    res = {}
    perms = rotation_perms(ball, fl, T0, rings, P)
    orbits = sum(2 ** cycles(p) for p in perms) // len(perms)
    assert sum(2 ** cycles(p) for p in perms) % len(perms) == 0
    res["rotations"] = len(perms)
    res["tilings_up_to_rotation"] = orbits
    B = PH.singlet_space(N)
    inv_singlets = round(sum(PH.character(B, N, p) for p in perms) / len(perms))
    res["I_invariant_states"] = {"all_4096": orbits, "singlets_132": inv_singlets}
    nn, SS, X = ops(edges)
    D = 1 << N
    s = np.arange(D)
    popcount = np.array([bin(x).count("1") for x in s])
    assert B.shape == (D, 132) and np.all(np.abs(B[popcount != 6]) < 1e-12)
    res["singlet_dim"] = 132
    res["tilings_with_6_rings_flipped"] = int((popcount == 6).sum())
    V = 80.0 - nn
    # commutator of V with total S^2 (sum over all pairs of swaps)
    PVP = B.T @ (V[:, None] * B)
    PSS = B.T @ SS @ B
    target = 72.5 * np.eye(132) - PSS / 3.0
    res["PVP_minus_heisenberg_form"] = float(np.abs(PVP - target).max())
    leak = np.linalg.norm(V[:, None] * B - B @ PVP) / np.linalg.norm(V[:, None] * B)
    res["V_leaves_singlet_space_relative_norm"] = float(leak)
    w = np.linalg.eigvalsh(PSS)
    res["heisenberg_on_singlets"] = {"min": float(w[0]), "second": float(w[np.argmax(w > w[0] + 1e-9)]), "max": float(w[-1]),
                                     "deg_min": int((w < w[0] + 1e-9).sum()), "deg_max": int((w > w[-1] - 1e-9).sum())}
    wv = np.linalg.eigvalsh(PVP)
    res["V_on_singlets"] = {"min": float(wv[0]), "max": float(wv[-1]), "classical_range": [50, 80]}
    wf = np.linalg.eigvalsh(SS)
    res["heisenberg_all_states"] = {"min": float(wf[0]), "max": float(wf[-1])}
    # classical statistics
    cnt = collections.Counter(int(v) for v in V)
    res["classical_V_counts"] = {"V=80 (no two neighbouring rings flipped)": cnt[80], "V=50": cnt[50]}
    # Rydberg-type Hamiltonians
    scan = []
    for sign, name in ((+1, "H = +kappa sum n n - J sum X  (vertices favoured)"), (-1, "H = -kappa sum n n - J sum X  (vertices disfavoured)")):
        for J in (0.0, 0.1, 0.3, 1.0, 3.0):
            H = np.diag(sign * nn) - J * X
            ev, U = np.linalg.eigh(H)
            g0 = U[:, 0]
            scan.append({"model": name, "J_over_kappa": J, "E0": float(ev[0]), "gap": float(ev[np.argmax(ev > ev[0] + 1e-9)] - ev[0]),
                         "deg0": int((ev < ev[0] + 1e-9).sum()), "mean_rings_flipped": float((g0 ** 2) @ popcount),
                         "mean_V": float((g0 ** 2) @ V), "weight_in_singlets": float(np.linalg.norm(B.T @ g0) ** 2)})
    res["hamiltonian_scan"] = scan
    if verbose:
        print(json.dumps({k: v for k, v in res.items() if k != "hamiltonian_scan"}, indent=1))
        for r in scan:
            print("  %-52s J=%.1f E0=%9.4f gap=%.4f deg=%d <flipped>=%.2f <V>=%.2f singlet weight=%.3f" % (
                r["model"], r["J_over_kappa"], r["E0"], r["gap"], r["deg0"], r["mean_rings_flipped"], r["mean_V"], r["weight_in_singlets"]))
    return res


def run_tests():
    r = analyse(verbose=False)
    assert r["rotations"] == 60
    print("1. the 60 rotations permute the ring bits (checked on 360 random tilings); tilings up to rotation:", r["tilings_up_to_rotation"])
    assert r["tilings_with_6_rings_flipped"] == 924
    print("2. the 132 singlets are supported on the 924 tilings with 6 rings flipped; rotation-invariant singlets:", r["I_invariant_states"]["singlets_132"])
    assert r["PVP_minus_heisenberg_form"] < 1e-10 and r["V_leaves_singlet_space_relative_norm"] > 1e-3
    print("3. P V P = 72.5 - (1/3) sum_edges S_i.S_j on the singlets (error %.1e); V itself leaves the singlet space" % r["PVP_minus_heisenberg_form"])
    assert r["classical_V_counts"]["V=50"] == 1
    print("4. classical: V = 80 for", r["classical_V_counts"]["V=80 (no two neighbouring rings flipped)"], "tilings, V = 50 for 1")
    print("all tests passed")
    return r


if __name__ == "__main__":
    if "--tests" in sys.argv:
        run_tests()
    else:
        run_tests()
        r = analyse()
        json.dump(r, open(os.path.join(HERE, "ring_bits.json"), "w"), indent=1)
        print("-> ring_bits.json")
