#!/usr/bin/env python3
"""
Quantum Monte Carlo for the pair-qubit Hamiltonian (Stage 4 at 82 and 804 pairs)

    H = -J sum_i X_i + sum_{p<q} K_pq Z_p Z_q,        K = kappa * W   (W: dipole couplings of the exact patch),

which is stoquastic (the only off-diagonal element is -J), so it has no sign problem. This is a
discrete-time path integral (Trotter, M slices of width dtau = beta / M) with the Ising ladder

    weight = exp( -dtau sum_m E_zz(s_m) + K_tau sum_{m,i} s_{m,i} s_{m+1,i} ),   K_tau = -1/2 ln tanh(dtau J).

Moves: single spin flips, segment flips (a world-line stretch of one site), and whole world-line
flips. A cluster update is not available because the couplings have both signs (frustrated), so the
sampler is local; `check_convergence` reports the chain-to-chain and beta-to-beta spread instead of
assuming it. Measured: <Z_i Z_j>, <X_i> (exact estimator of the discrete weight) and the energy.

What it does not measure: entanglement. The path integral gives a thermal state at inverse
temperature beta. For beta >> 1/(same-parity gap) that is the even/odd doublet of the ground states,
whose Z-correlators equal the ground-state ones, but the doublet mixture is not the pure even ground
state that the concurrence and entropies of `pair_hamiltonian` refer to.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit


# ---- the sampler -------------------------------------------------------------------------
@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _fields(s, K):
    M, N = s.shape
    F = np.empty((M, N))
    for m in range(M):
        for i in range(N):
            acc = 0.0
            for j in range(N):
                acc += K[i, j] * s[m, j]
            F[m, i] = acc
    return F


@njit(cache=True)
def _flip(s, F, K, m, i):
    """Flip s[m, i] and update the field of every other spin on that slice."""
    old = s[m, i]
    s[m, i] = -old
    N = s.shape[1]
    d = -2.0 * old
    for j in range(N):
        F[m, j] += d * K[j, i]


@njit(cache=True)
def _sweep(s, F, K, dtau, Kt, max_seg):
    """One sweep: M*N single flips, N segment flips and N/4 whole world-line flips (random sites)."""
    M, N = s.shape
    acc1 = 0
    for _ in range(M * N):
        m = np.random.randint(M)
        i = np.random.randint(N)
        x = s[m, i]
        dS = -2.0 * dtau * x * F[m, i] + 2.0 * Kt * x * (s[(m - 1) % M, i] + s[(m + 1) % M, i])
        if dS <= 0.0 or np.random.random() < math.exp(-dS):
            _flip(s, F, K, m, i)
            acc1 += 1
    acc2 = 0
    for _ in range(N):
        i = np.random.randint(N)
        L = 1 + np.random.randint(max_seg)
        m0 = np.random.randint(M)
        dS = 0.0
        for k in range(L):
            m = (m0 + k) % M
            dS += -2.0 * dtau * s[m, i] * F[m, i]
        a = s[(m0 - 1) % M, i] * s[m0, i]
        b = s[(m0 + L - 1) % M, i] * s[(m0 + L) % M, i]
        dS += 2.0 * Kt * (a + b)
        if dS <= 0.0 or np.random.random() < math.exp(-dS):
            for k in range(L):
                _flip(s, F, K, (m0 + k) % M, i)
            acc2 += 1
    acc3 = 0
    for _ in range(max(1, N // 4)):
        i = np.random.randint(N)
        dS = 0.0
        for m in range(M):
            dS += -2.0 * dtau * s[m, i] * F[m, i]
        if dS <= 0.0 or np.random.random() < math.exp(-dS):
            for m in range(M):
                _flip(s, F, K, m, i)
            acc3 += 1
    return acc1, acc2, acc3



@njit(cache=True)
def _bond_sum(s):
    M, N = s.shape
    acc = 0.0
    for m in range(M):
        m1 = (m + 1) % M
        for i in range(N):
            acc += s[m, i] * s[m1, i]
    return acc


@njit(cache=True)
def _zz_total(s, F):
    M, N = s.shape
    acc = 0.0
    for m in range(M):
        for i in range(N):
            acc += s[m, i] * F[m, i]
    return 0.5 * acc


@njit(cache=True)
def _swap_replicas(S, F, dts, Kts, order, parity):
    """
    Exchange configurations between neighbouring ladder positions. The weight of a configuration s at
    position r is exp(-dt_r E(s) + Kt_r B(s)), E = sum over slices of the ZZ energy, B = sum of s s'
    over the time bonds, so the swap ratio is
        exp[ -(dt_r - dt_r') (E_b - E_a) + (Kt_r - Kt_r') (B_b - B_a) ]
    with a the configuration at r and b at r'. `order[r]` is the index of the state array at position r.
    The number of slices M is common, so position r has beta_r = M dt_r. Returns accepted swaps.
    """
    R = len(order)
    n = 0
    for r in range(parity, R - 1, 2):
        a, b = order[r], order[r + 1]
        lw = (-(dts[r] - dts[r + 1]) * (_zz_total(S[b], F[b]) - _zz_total(S[a], F[a]))
              + (Kts[r] - Kts[r + 1]) * (_bond_sum(S[b]) - _bond_sum(S[a])))
        if lw >= 0.0 or np.random.random() < math.exp(lw):
            order[r], order[r + 1] = b, a
            n += 1
    return n


@dataclass
class QMCResult:
    n: int
    beta: float
    dtau: float
    ZZ: np.ndarray            # <Z_i Z_j>, diagonal 1
    X: np.ndarray             # <X_i>
    energy: float
    energy_err: float
    n_meas: int
    acc: Tuple[float, float, float]
    energy_series: np.ndarray
    swap_rate: float = float('nan')


def run_qmc(K: np.ndarray, J: float, beta: float, dtau: float = 0.1, sweeps: int = 2000, therm: int = 500,
            seed: int = 0, measure_every: int = 5, max_seg: Optional[int] = None) -> QMCResult:
    """One chain from a random start. `K` is the full coupling matrix (kappa already included)."""
    K = np.ascontiguousarray(K, dtype=float)
    N = K.shape[0]
    M = max(4, int(round(beta / dtau)))
    dtau = beta / M
    Kt = -0.5 * math.log(math.tanh(dtau * J))
    max_seg = max_seg or max(1, M // 8)
    _seed(seed)
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(M, N))
    F = _fields(s, K)
    ZZ = np.zeros((N, N))
    X = np.zeros(N)
    es: List[float] = []
    nm = 0
    acc = np.zeros(3)
    ch, cc = math.tanh(dtau * J), 1.0 / math.tanh(dtau * J)
    for it in range(therm + sweeps):
        a = _sweep(s, F, K, dtau, Kt, max_seg)
        if it >= therm:
            acc += np.array(a, float) / np.array([M * N, N, max(1, N // 4)])
            if (it - therm) % measure_every == 0:
                ZZ += s.T @ s / M
                same = s * np.roll(s, -1, axis=0) > 0
                xi = np.where(same, ch, cc).mean(axis=0)
                X += xi
                ezz = 0.5 * float(np.mean(np.einsum("mi,mi->m", s, F)))
                es.append(-J * float(xi.sum()) + ezz)
                nm += 1
    es_a = np.array(es)
    ZZ /= nm
    X /= nm
    nact = max(1, sweeps)
    return QMCResult(N, beta, dtau, ZZ, X, float(es_a.mean()), _binned_err(es_a), nm, tuple(acc / nact), es_a)


def run_qmc_pt(K: np.ndarray, J: float, beta: float, dtau: float = 0.1, sweeps: int = 2000, therm: int = 500,
               seed: int = 0, measure_every: int = 5, n_replicas: int = 8, beta_min_factor: float = 0.1,
               max_seg: Optional[int] = None) -> QMCResult:
    """
    Parallel tempering in beta. Position 0 is the target (beta, M = beta / dtau slices); the others keep
    M and shrink dtau geometrically down to beta_min_factor * beta, so they are hotter (and, for a
    fixed M, the Trotter step is finer). Neighbouring positions swap configurations; only position 0
    is measured. Hot replicas cross the barriers between classical basins that trap a single chain.
    Returns the swap acceptance per pair of neighbours in `acc[2]`'s place is not kept: see `swap_rates`.
    """
    K = np.ascontiguousarray(K, dtype=float)
    N = K.shape[0]
    M = max(4, int(round(beta / dtau)))
    dtau = beta / M
    dts = dtau * np.geomspace(1.0, beta_min_factor, n_replicas)
    Kts = -0.5 * np.log(np.tanh(dts * J))
    max_seg = max_seg or max(1, M // 8)
    _seed(seed)
    rng = np.random.default_rng(seed)
    S = np.ascontiguousarray(np.array([rng.choice([-1.0, 1.0], size=(M, N)) for _ in range(n_replicas)]))
    Fs = np.array([_fields(S[r], K) for r in range(n_replicas)])
    order = np.arange(n_replicas)
    ZZ = np.zeros((N, N)); X = np.zeros(N); es: List[float] = []
    nm = 0
    acc = np.zeros(3)
    swaps = 0
    ch, cc = math.tanh(dtau * J), 1.0 / math.tanh(dtau * J)
    for it in range(therm + sweeps):
        for r in range(n_replicas):
            a = _sweep(S[order[r]], Fs[order[r]], K, dts[r], Kts[r], max_seg)
            if r == 0 and it >= therm:
                acc += np.array(a, float) / np.array([M * N, N, max(1, N // 4)])
        sw = _swap_replicas(S, Fs, dts, Kts, order, it % 2)
        if it >= therm:
            swaps += sw
        if it >= therm and (it - therm) % measure_every == 0:
            s, F = S[order[0]], Fs[order[0]]
            ZZ += s.T @ s / M
            same = s * np.roll(s, -1, axis=0) > 0
            xi = np.where(same, ch, cc).mean(axis=0)
            X += xi
            es.append(-J * float(xi.sum()) + 0.5 * float(np.mean(np.einsum("mi,mi->m", s, F))))
            nm += 1
    es_a = np.array(es)
    res = QMCResult(N, beta, dtau, ZZ / nm, X / nm, float(es_a.mean()), _binned_err(es_a), nm,
                    tuple(acc / max(1, sweeps)), es_a)
    res.swap_rate = swaps / max(1, sweeps) / max(1, (n_replicas - 1) / 2)       # per attempted pair, roughly
    return res


def _binned_err(x: np.ndarray, nbins: int = 20) -> float:
    n = len(x) // nbins
    if n < 1:
        return float("nan")
    b = x[: n * nbins].reshape(nbins, n).mean(axis=1)
    return float(b.std(ddof=1) / math.sqrt(nbins))


# ---- exact reference ---------------------------------------------------------------------
def exact_thermal(K: np.ndarray, J: float, beta: float) -> Dict[str, object]:
    """
    Exact thermal <Z_i Z_j>, <X_i> and energy at inverse temperature beta by full diagonalisation
    (n <= 12 or so). This is the reference for the sampler at any beta, including moderate ones.
    """
    import pair_hamiltonian as ph
    n = K.shape[0]
    H = ph.build_hamiltonian(K, J, 1.0).toarray()
    w, v = np.linalg.eigh(H)
    p = np.exp(-beta * (w - w[0]))
    p /= p.sum()
    idx = np.arange(2 ** n)
    z = np.array([1.0 - 2.0 * ((idx >> (n - 1 - q)) & 1) for q in range(n)])
    prob = v ** 2                                           # (basis, level)
    ZZ = (z * (prob @ p)) @ z.T
    X = np.array([float(np.sum(p * np.einsum("bk,bk->k", v, v[idx ^ (1 << (n - 1 - q))]))) for q in range(n)])
    return {"ZZ": ZZ, "X": X, "energy": float(p @ w)}


def exact_thermal_doublet(W_or_K: np.ndarray, J: float, beta: float) -> Dict[str, object]:
    """
    Exact <Z_i Z_j>, <X_i> and energy at inverse temperature beta, from the lowest levels of the
    even and odd sectors (Boltzmann weights), valid when beta times the same-parity gap is large.
    `W_or_K` is K (kappa = 1 passed to `pair_hamiltonian`).
    """
    import pair_hamiltonian as ph
    n = W_or_K.shape[0]
    levels = []
    for par in (+1, -1):
        dim = 2 ** (n - 1)
        if dim <= 64:                                  # dense for tiny systems (eigsh needs k < dim)
            Hs = ph.sector_hamiltonian(W_or_K, J, 1.0, par).toarray()
            w, v = np.linalg.eigh(Hs)
            w, v = w[:2], v[:, :2]
        else:
            w, v = ph.sector_levels(W_or_K, J, 1.0, par, k=2)
        for e, c in zip(w, v.T):
            levels.append((float(e), ph.embed_sector(c, n, par)))
    e0 = min(e for e, _ in levels)
    wts = np.array([math.exp(-beta * (e - e0)) for e, _ in levels])
    wts /= wts.sum()
    idx = np.arange(2 ** n)
    z = np.array([1.0 - 2.0 * ((idx >> (n - 1 - p)) & 1) for p in range(n)])      # (n, 2^n)
    ZZ = np.zeros((n, n))
    X = np.zeros(n)
    E = 0.0
    for (e, psi), wt in zip(levels, wts):
        pr = np.abs(psi) ** 2
        ZZ += wt * (z * pr) @ z.T
        for p in range(n):
            flipped = psi[idx ^ (1 << (n - 1 - p))]
            X[p] += wt * float(np.real(np.vdot(psi, flipped)))
        E += wt * e
    return {"ZZ": ZZ, "X": X, "energy": E, "weights": wts, "levels": [e for e, _ in levels]}


def compare_with_exact(K: np.ndarray, J: float, beta: float, dtau: float, sweeps: int, seed: int = 1) -> Dict[str, float]:
    ex = exact_thermal_doublet(K, J, beta)
    r = run_qmc(K, J, beta, dtau, sweeps=sweeps, seed=seed)
    iu = np.triu_indices(K.shape[0], 1)
    return {"n": K.shape[0], "E_exact": ex["energy"], "E_qmc": r.energy, "E_err": r.energy_err,
            "ZZ_max_dev": float(np.abs(r.ZZ - ex["ZZ"])[iu].max()), "ZZ_rms_dev": float(np.sqrt(np.mean((r.ZZ - ex["ZZ"])[iu] ** 2))),
            "X_max_dev": float(np.abs(r.X - ex["X"]).max()), "acc_single": r.acc[0]}


# ---- checks -------------------------------------------------------------------------------
def _flow_K(n: int, flow_tau: float, generation: int = 3) -> Tuple[np.ndarray, float]:
    import pair_hamiltonian as ph
    fp = ph.flow_parameters(flow_tau)
    return ph.flow_couplings(generation, n, flow_tau), fp.J


def test_free_spins_give_tanh():
    # no couplings: <X> = tanh(beta J) exactly (the estimator is unbiased, only noisy: sigma ~ 0.02 here)
    r = run_qmc(np.zeros((6, 6)), 0.5, 4.0, dtau=0.1, sweeps=6000, therm=300, seed=1)
    assert abs(r.X.mean() - math.tanh(2.0)) < 0.02, r.X.mean()
    assert np.abs(r.ZZ - np.eye(6))[np.triu_indices(6, 1)].max() < 0.15


def test_small_frustrated_clusters_match_exact_thermal():
    # central pairs of the generation-3 patch at flow time 0.05 (mixed-sign dipole couplings)
    for n, sweeps in ((3, 20000), (6, 20000)):
        K, J = _flow_K(n, 0.05)
        ex = exact_thermal(K, J, 25.0)
        r = run_qmc(K, J, 25.0, dtau=0.1, sweeps=sweeps, therm=1500, seed=n, measure_every=2)
        iu = np.triu_indices(n, 1)
        assert abs(r.energy - ex["energy"]) < 4 * r.energy_err + 0.02, (n, r.energy, ex["energy"], r.energy_err)
        assert np.abs(r.ZZ - ex["ZZ"])[iu].max() < 0.06, n
        assert np.abs(r.X - ex["X"]).max() < 0.08, n


def test_tempered_sampler_matches_exact_at_eight_pairs():
    K, J = _flow_K(8, 0.05)
    ex = exact_thermal(K, J, 25.0)
    r = run_qmc_pt(K, J, 25.0, dtau=0.1, sweeps=12000, therm=1500, seed=4, n_replicas=6, beta_min_factor=0.2)
    iu = np.triu_indices(8, 1)
    assert abs(r.energy - ex["energy"]) < 4 * r.energy_err + 0.02, (r.energy, ex["energy"], r.energy_err)
    assert np.abs(r.ZZ - ex["ZZ"])[iu].max() < 0.1 and np.abs(r.X - ex["X"]).max() < 0.1
    assert 0.1 < r.swap_rate <= 1.0


def test_pair_symmetry_and_exact_doublet_agrees_with_full_thermal():
    K, J = _flow_K(8, 0.0)
    a, b = exact_thermal(K, J, 60.0), exact_thermal_doublet(K, J, 60.0)
    assert abs(a["energy"] - b["energy"]) < 1e-6 and np.abs(a["ZZ"] - b["ZZ"]).max() < 1e-5 and np.abs(a["X"] - b["X"]).max() < 1e-5
    r = run_qmc(K, J, 10.0, dtau=0.1, sweeps=500, therm=100, seed=5)
    assert np.allclose(r.ZZ, r.ZZ.T) and np.allclose(np.diag(r.ZZ), 1.0)


def run_tests(verbose: bool = True) -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            if verbose:
                print(f"  {name:<62} PASS")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  {name:<62} FAIL: {exc!r}")
    print(f"{len(tests)} test(s), {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_tests() else 0)
