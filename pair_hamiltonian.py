"""
Stage 2 of ROADMAP_QUANTUM_GRAVITY.md: the pair Hamiltonian on 6 to 14 pairs.

    H = -J sum_p X_p + kappa sum_{p<q} W_pq Z_p Z_q

with W the dipole couplings of `quantumspectremodel.pair_couplings`. H is real and commutes with the
global parity P = prod X_p, so the ground state is real and lives in one parity sector.

This module provides
  * `build_hamiltonian`  sparse H (exact diagonalisation up to about 14 pairs),
  * `ground_state`, `spectrum_low`, `parity_gap`,
  * `classical_ground_state`  the Ising ground state at J -> 0 (and its cat state),
  * `reduced_two_qubit`, `concurrence`, `chsh_max`  measures on the reduced state of two pairs,
  * `trotter1`, `trotter2`  matrix-free evolution with `state.State`, compared with exact evolution.

Run `python3 pair_hamiltonian.py --tests-only` for the checks, or without arguments for the
tau scan (tau = J / (kappa * w), w the rms nearest-neighbour coupling).
"""
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quantumspectremodel import (
    assign_qubits_to_patch,
    build_substitution_tier,
    pair_couplings,
)
from state import State


# ---- couplings --------------------------------------------------------------------------

def central_couplings(generation: int, n_pairs: int) -> np.ndarray:
    """W for the `n_pairs` pairs nearest the centre of the generation-`generation` patch."""
    patch = build_substitution_tier(generation)
    assign_qubits_to_patch(patch, max_qubits=n_pairs)
    return pair_couplings(patch)


def coupling_scale(W: np.ndarray) -> float:
    """w = rms over each pair's largest-|W| partner (a nearest-neighbour coupling scale)."""
    n = W.shape[0]
    big = np.array([np.max(np.abs(W[p][np.arange(n) != p])) for p in range(n)])
    return float(np.sqrt(np.mean(big ** 2)))


# ---- the Spectre-to-Hat flow: J, kappa and h from the tile ---------------------------------
#
# Definitions D6.3, D6.6 and D6.7 of `toymodel_v6` (spectre-monotile-py, research/toymodel_v6,
# STAGE1.md). They are choices, not derivations; they replace the hand-chosen J = 1, h = 0.3 that
# the network used to default to, which put it at tau = J / (kappa w) of about 200 (a product
# state) instead of the window where the pairs entangle.
#
#   h(tau)     = tan(omega0 (1 - tau) / 2),  omega0 = 120 degrees    the jump of the 14th vertex
#                                                                    (sqrt3 at the Spectre, 0 at the Hat)
#   c(tau)     = exp(-(2 h)^2 / (8 sigma^2)),  sigma = 1             overlap of the two positions
#   Gamma(tau) = J0 c(tau)                                           tunnelling,  J0 = 1
#   K(tau)     = kappa(tau) W,  kappa(tau) = kappa_norm h(tau)^2     coupling of the bits read as charges
#   kappa_norm = K0 / (h(0)^2 max|W|) over the six pairs of generation 2, so the strongest
#                coupling at the Spectre end is K0 = 1.
#   b(tau)     = 1 + (sqrt3 - 1) tau                                 the second edge of the tile
#
# Exact patches need a = b = 1, which is the Spectre end tau = 0. For tau > 0 the couplings W are
# kept at b = 1 and only rescaled by (h(tau)/h(0))^2. Against the v6 scan (couplings recomputed at
# b(tau)) on the 10 central pairs of generation 3, the coupling matrix is off by 0.3% (tau = 0.02),
# 0.7% (0.05), 1.8% (0.13), 4.0% (0.3) in the Frobenius norm and the largest concurrence by at most
# 0.014; see `test_flow_agrees_with_toymodel_v6`.

FLOW_OMEGA0_DEG = 120.0
FLOW_SIGMA = 1.0
FLOW_J0 = 1.0
FLOW_K0 = 1.0


def flow_b(tau: float) -> float:
    return 1.0 + (math.sqrt(3.0) - 1.0) * tau


def flow_h(tau: float) -> float:
    return math.tan(math.radians(FLOW_OMEGA0_DEG * (1.0 - tau)) / 2.0)


def flow_overlap(tau: float) -> float:
    return math.exp(-((2.0 * flow_h(tau)) ** 2) / (8.0 * FLOW_SIGMA ** 2))


@lru_cache(maxsize=None)
def flow_kappa_norm() -> float:
    """kappa_norm = K0 / (h(0)^2 max|W|), W over the six pairs of the generation-2 patch (b = 1)."""
    patch = build_substitution_tier(2)
    assign_qubits_to_patch(patch, max_qubits=None)
    return FLOW_K0 / (flow_h(0.0) ** 2 * float(np.abs(pair_couplings(patch)).max()))


@dataclass(frozen=True)
class FlowParameters:
    """Model parameters at flow time tau: H = -J sum X + kappa sum W_pq Z_p Z_q, bit displacement h."""

    tau: float
    b: float
    h: float
    J: float
    kappa: float
    overlap: float


def flow_parameters(tau: float = 0.0) -> FlowParameters:
    """J, kappa and h of the Spectre-to-Hat flow at tau in [0, 1]; tau = 0 is the Spectre (exact patches)."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"flow time must lie in [0, 1], got {tau}")
    h = flow_h(tau)
    c = flow_overlap(tau)
    return FlowParameters(tau=tau, b=flow_b(tau), h=h, J=FLOW_J0 * c, kappa=flow_kappa_norm() * h * h, overlap=c)


def tau_effective(W: np.ndarray, J: float, kappa: float) -> float:
    """tau = J / (kappa w): the ratio of tunnelling to coupling scale that `tau_scan` uses."""
    return J / (kappa * coupling_scale(W))


# ---- exact Hamiltonian ------------------------------------------------------------------

def zz_diagonal(W: np.ndarray) -> np.ndarray:
    """sum_{p<q} W_pq z_p z_q for every basis state (bit p of the index, big endian, 0 -> z=+1)."""
    n = W.shape[0]
    idx = np.arange(2 ** n)
    z = np.empty((n, 2 ** n))
    for p in range(n):
        z[p] = 1.0 - 2.0 * ((idx >> (n - 1 - p)) & 1)
    diag = np.zeros(2 ** n)
    for p in range(n):
        for q in range(p + 1, n):
            if W[p, q] != 0.0:
                diag += W[p, q] * z[p] * z[q]
    return diag


def build_hamiltonian(W: np.ndarray, J: float, kappa: float) -> sp.csr_matrix:
    n = W.shape[0]
    dim = 2 ** n
    idx = np.arange(dim)
    rows, cols = [], []
    for p in range(n):
        rows.append(idx)
        cols.append(idx ^ (1 << (n - 1 - p)))
    X = sp.csr_matrix((np.ones(n * dim), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
    return (-J * X + sp.diags(kappa * zz_diagonal(W))).tocsr()


def spectrum_low(H: sp.csr_matrix, k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    dim = H.shape[0]
    k = min(k, dim - 2)
    if dim <= 512:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k], v[:, :k]
    w, v = spla.eigsh(H, k=k, which="SA", tol=1e-12)
    order = np.argsort(w)
    return w[order], v[:, order]


def parity_of(psi: np.ndarray, n: int) -> float:
    """<prod X> of a real or complex state (flip every bit = reverse the index)."""
    return float(np.real(np.vdot(psi, psi[::-1])))


def sector_hamiltonian(W: np.ndarray, J: float, kappa: float, parity: int) -> sp.csr_matrix:
    """
    H restricted to the global-parity sector P = prod X = parity (+1 or -1), in the basis
    (|s> + parity |~s>)/sqrt2 for the 2^(n-1) configurations s with bit 0 equal to 0.
    Exact, half the dimension, and it separates the near-degenerate cat doublet.
    """
    n = W.shape[0]
    dim = 2 ** (n - 1)
    idx = np.arange(dim)                       # configurations with the leading bit 0
    full = 2 ** n - 1
    rows, cols, vals = [], [], []
    for p in range(1, n):                       # flip a bit other than the leading one
        rows.append(idx)
        cols.append(idx ^ (1 << (n - 1 - p)))
        vals.append(np.full(dim, -J))
    flipped = idx | (1 << (n - 1))              # flipping the leading bit gives leading bit 1 ...
    rows.append(idx)
    cols.append(full ^ flipped)                 # ... whose canonical representative is its complement
    vals.append(np.full(dim, -J * parity))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
    return (H + sp.diags(kappa * zz_diagonal(W)[:dim])).tocsr()


def embed_sector(c: np.ndarray, n: int, parity: int) -> np.ndarray:
    """Sector coefficients -> full 2^n state vector."""
    dim = 2 ** (n - 1)
    psi = np.zeros(2 ** n)
    psi[:dim] = c / np.sqrt(2.0)
    psi[dim:] = parity * c[::-1] / np.sqrt(2.0)   # |~s> sits at index full ^ s = (2^n - 1) - s
    return psi


def sector_levels(W: np.ndarray, J: float, kappa: float, parity: int, k: int = 4):
    w, v = spectrum_low(sector_hamiltonian(W, J, kappa, parity), k=k)
    return w, v


def ground_state(W: np.ndarray, J: float, kappa: float) -> Tuple[float, np.ndarray]:
    """
    The ground state, which has parity +1 (Perron-Frobenius: every off-diagonal element of H is
    -J < 0 in the Z basis, so the ground amplitudes are positive and symmetric under s -> ~s).
    Taken from the even sector, so it is the even cat state at J -> 0, never an arbitrary mix of
    the nearly degenerate doublet.
    """
    n = W.shape[0]
    w, v = sector_levels(W, J, kappa, +1, k=2)
    return float(w[0]), embed_sector(v[:, 0], n, +1)


def parity_gap(W: np.ndarray, J: float, kappa: float, k: int = 4) -> Dict[str, float]:
    """
    E0 is the even-sector ground energy. `gap` = E_odd0 - E0, the cat-doublet splitting (closes
    to 0 as J -> 0); `gap_same_parity` = E_even1 - E0 (stays open at J -> 0, equal to the first
    classical excitation there).
    """
    we, _ = sector_levels(W, J, kappa, +1, k=k)
    wo, _ = sector_levels(W, J, kappa, -1, k=k)
    return {
        "E0": float(we[0]),
        "gap": float(wo[0] - we[0]),
        "gap_same_parity": float(we[1] - we[0]),
        "parity": 1.0,
    }


# ---- classical limit and the cat state ---------------------------------------------------

def classical_ground_state(W: np.ndarray) -> Tuple[np.ndarray, float]:
    """The Ising ground configuration (z = +-1, up to the global flip) and its energy per unit kappa."""
    n = W.shape[0]
    diag = zz_diagonal(W)
    i0 = int(np.argmin(diag))
    z = np.array([1 - 2 * ((i0 >> (n - 1 - p)) & 1) for p in range(n)])
    return z, float(diag[i0])


def cat_state(W: np.ndarray) -> np.ndarray:
    n = W.shape[0]
    diag = zz_diagonal(W)
    i0 = int(np.argmin(diag))
    psi = np.zeros(2 ** n)
    psi[i0] = psi[(2 ** n - 1) ^ i0] = 1.0 / np.sqrt(2.0)
    return psi


def classical_degeneracy(W: np.ndarray, tol: float = 1e-9) -> int:
    diag = zz_diagonal(W)
    return int(np.sum(diag < diag.min() + tol))


# ---- two-qubit measures ------------------------------------------------------------------

def reduced_two_qubit(psi: np.ndarray, n: int, p: int, q: int) -> np.ndarray:
    """4x4 reduced density matrix of qubits p < q of a pure n-qubit state."""
    t = psi.reshape([2] * n)
    t = np.moveaxis(t, [p, q], [0, 1]).reshape(4, -1)
    return t @ t.conj().T


_SY = np.array([[0, -1j], [1j, 0]])
_YY = np.kron(_SY, _SY)


def concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence of a two-qubit density matrix."""
    rt = _YY @ rho.conj() @ _YY
    ev = np.sqrt(np.clip(np.sort(np.real(np.linalg.eigvals(rho @ rt)))[::-1], 0.0, None))
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def chsh_max(rho: np.ndarray) -> float:
    """Horodecki: the largest CHSH value over measurement settings, 2 sqrt(m1 + m2); > 2 violates."""
    s = [np.array([[0, 1], [1, 0]]), _SY, np.array([[1, 0], [0, -1]])]
    T = np.array([[np.real(np.trace(rho @ np.kron(a, b))) for b in s] for a in s])
    m = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
    return float(2.0 * np.sqrt(m[0] + m[1]))


def pairwise_concurrence(psi: np.ndarray, n: int) -> np.ndarray:
    C = np.zeros((n, n))
    for p in range(n):
        for q in range(p + 1, n):
            C[p, q] = C[q, p] = concurrence(reduced_two_qubit(psi, n, p, q))
    return C


def entanglement_entropy(psi: np.ndarray, n: int, region: List[int]) -> float:
    """von Neumann entropy (bits) of the qubits in `region` for a pure state."""
    rest = [q for q in range(n) if q not in region]
    t = np.transpose(psi.reshape([2] * n), list(region) + rest).reshape(2 ** len(region), -1)
    s = np.linalg.svd(t, compute_uv=False) ** 2
    s = s[s > 1e-14]
    return float(-np.sum(s * np.log2(s)))


# ---- evolution ---------------------------------------------------------------------------

def exact_evolve(W: np.ndarray, J: float, kappa: float, psi0: np.ndarray, t: float) -> np.ndarray:
    """exp(-iHt) psi0 by Krylov action (no dense matrix)."""
    H = build_hamiltonian(W, J, kappa)
    return spla.expm_multiply(-1j * t * H, psi0)


def z_expectations(psi: np.ndarray, n: int) -> np.ndarray:
    prob = np.abs(psi) ** 2
    idx = np.arange(2 ** n)
    return np.array([float(np.sum(prob * (1.0 - 2.0 * ((idx >> (n - 1 - p)) & 1)))) for p in range(n)])


def _zz_layer(st: State, W: np.ndarray, dt: float, kappa: float) -> None:
    n = W.shape[0]
    for p in range(n):
        for q in range(p + 1, n):
            if W[p, q] != 0.0:
                st.cx(p, q)
                st.rz(q, 2.0 * dt * kappa * W[p, q])
                st.cx(p, q)


def _x_layer(st: State, n: int, dt: float, J: float) -> None:
    for p in range(n):
        st.h(p)
        st.rz(p, -2.0 * dt * J)
        st.h(p)


def trotter1(st: State, W, J, kappa, dt) -> None:
    _zz_layer(st, W, dt, kappa)
    _x_layer(st, W.shape[0], dt, J)


def trotter2(st: State, W, J, kappa, dt) -> None:
    """Symmetric (Strang) step: error O(dt^3) per step."""
    _x_layer(st, W.shape[0], dt / 2, J)
    _zz_layer(st, W, dt, kappa)
    _x_layer(st, W.shape[0], dt / 2, J)


def state_to_vector(st: State) -> np.ndarray:
    v = np.zeros(2 ** st.n_qubits, complex)
    for b, a in st.state.to_list():
        v[int(b.to01(), 2)] = a
    return v


def run_state(W, J, kappa, t, steps, order=2) -> np.ndarray:
    st = State(n_qubits=W.shape[0])
    step = trotter2 if order == 2 else trotter1
    for _ in range(steps):
        step(st, W, J, kappa, t / steps)
    return state_to_vector(st)


# ---- scans -------------------------------------------------------------------------------

def ground_measures(W: np.ndarray, J: float, kappa: float) -> Dict[str, float]:
    """Gaps, best pairwise concurrence and CHSH, half-system entropy and mean <Z> at one (J, kappa)."""
    n = W.shape[0]
    g = parity_gap(W, J, kappa)
    _, psi = ground_state(W, J, kappa)
    C = pairwise_concurrence(psi, n)
    iu = np.triu_indices(n, 1)
    k = int(np.argmax(C[iu]))
    p, q = iu[0][k], iu[1][k]
    return {
        "gap": g["gap"],
        "gap_same_parity": g["gap_same_parity"],
        "max_concurrence": float(C[iu][k]),
        "best_pair": (int(p), int(q)),
        "chsh_best": chsh_max(reduced_two_qubit(psi, n, p, q)),
        "S_half": entanglement_entropy(psi, n, list(range(n // 2))),
        "mean_Z": float(np.mean(z_expectations(psi, n))),
    }


def tau_scan(W: np.ndarray, taus: np.ndarray) -> List[Dict[str, float]]:
    """Scan tau = J / (kappa w) at fixed kappa = 1 and report gaps and entanglement."""
    w = coupling_scale(W)
    return [{"tau": float(tau), **ground_measures(W, tau * w, 1.0)} for tau in taus]


def flow_scan(W: np.ndarray, flow_taus) -> List[Dict[str, float]]:
    """The same measures along the Spectre-to-Hat flow (J and kappa from `flow_parameters`)."""
    rows = []
    for t in flow_taus:
        fp = flow_parameters(float(t))
        rows.append({"flow_tau": float(t), "tau_eff": tau_effective(W, fp.J, fp.kappa), "J": fp.J,
                     "kappa": fp.kappa, "h": fp.h, **ground_measures(W, fp.J, fp.kappa)})
    return rows


# ---- checks ------------------------------------------------------------------------------

def test_hamiltonian_matches_dense_kron():
    W = central_couplings(2, 4)
    n = 4
    X = np.array([[0, 1], [1, 0]], float)
    Z = np.diag([1.0, -1.0])
    I = np.eye(2)

    def op(m, p):
        mats = [I] * n
        mats[p] = m
        out = mats[0]
        for x in mats[1:]:
            out = np.kron(out, x)
        return out

    H = sum(-0.7 * op(X, p) for p in range(n))
    for p in range(n):
        for q in range(p + 1, n):
            H = H + 1.3 * W[p, q] * op(Z, p) @ op(Z, q)
    assert np.allclose(build_hamiltonian(W, 0.7, 1.3).toarray(), H)


def test_sector_hamiltonians_reproduce_the_full_spectrum():
    W = central_couplings(3, 7)
    J, kappa = 0.3, 5.0
    full = np.sort(np.linalg.eigvalsh(build_hamiltonian(W, J, kappa).toarray()))
    both = np.sort(np.concatenate([
        np.linalg.eigvalsh(sector_hamiltonian(W, J, kappa, s).toarray()) for s in (+1, -1)]))
    assert np.allclose(full, both, atol=1e-10)
    E0, psi = ground_state(W, J, kappa)
    H = build_hamiltonian(W, J, kappa)
    assert abs(E0 - full[0]) < 1e-10 and np.linalg.norm(H @ psi - E0 * psi) < 1e-9
    assert abs(parity_of(psi, 7) - 1.0) < 1e-12


def test_zero_tunnelling_gives_a_cat_ground_state():
    W = central_couplings(3, 8)
    J = 1e-6
    _, psi = ground_state(W, J, 1.0)
    # the ground space is the doublet (|s> +- |-s>)/sqrt2; eigsh may return any mix of the two
    weight = np.vdot(cat_state(W), psi) ** 2     # the even cat, exactly (no solver mixing)
    assert classical_degeneracy(W) == 2, classical_degeneracy(W)
    assert weight > 0.999, weight
    g = parity_gap(W, J, 1.0)
    diag = np.sort(zz_diagonal(W))
    excitation = float(diag[diag > diag[0] + 1e-9][0] - diag[0])   # first classical excitation
    assert g["gap"] < 1e-6, g
    assert abs(g["gap_same_parity"] - excitation) < 1e-3 * excitation, (g, excitation)


def _odd_cat(W):
    n = W.shape[0]
    diag = zz_diagonal(W)
    i0 = int(np.argmin(diag))
    psi = np.zeros(2 ** n)
    psi[i0] = 1.0 / np.sqrt(2.0)
    psi[(2 ** n - 1) ^ i0] = -1.0 / np.sqrt(2.0)
    return psi


def test_cat_state_has_ghz_entropy_one_bit():
    W = central_couplings(3, 8)
    s = entanglement_entropy(cat_state(W), 8, [0, 1, 2])
    assert abs(s - 1.0) < 1e-9, s


def test_concurrence_of_bell_and_product():
    bell = np.zeros(4)
    bell[0] = bell[3] = 1 / np.sqrt(2)
    rho = np.outer(bell, bell)
    assert abs(concurrence(rho) - 1.0) < 1e-9
    assert abs(chsh_max(rho) - 2 * np.sqrt(2)) < 1e-9
    prod = np.zeros(4)
    prod[0] = 1.0
    assert concurrence(np.outer(prod, prod)) < 1e-9
    assert chsh_max(np.outer(prod, prod)) <= 2.0 + 1e-9


def test_trotter_state_matches_exact_evolution_8_pairs():
    # kappa = 20 puts kappa*W near 1, so the ZZ and X terms really fail to commute
    W = central_couplings(3, 8)
    J, kappa, t = 1.0, 20.0, 1.0
    psi0 = np.zeros(2 ** 8)
    psi0[0] = 1.0
    ref = exact_evolve(W, J, kappa, psi0, t)
    assert np.abs(z_expectations(ref, 8) - 1.0).max() > 0.5, "check is trivial if <Z> never moves"

    def err(steps, order):
        v = run_state(W, J, kappa, t, steps, order)
        return float(np.linalg.norm(v * np.exp(-1j * np.angle(np.vdot(ref, v))) - ref))

    e1, e2, e2b = err(16, 1), err(16, 2), err(32, 2)
    assert e2 < 0.01 and e2 < e1 / 5, (e1, e2)
    assert 3.0 < e2 / e2b < 5.0, (e2, e2b)      # second order: error / 4 when dt halves


def test_flow_endpoints_and_normalisation():
    f0, f1 = flow_parameters(0.0), flow_parameters(1.0)
    assert abs(f0.h - math.sqrt(3.0)) < 1e-12 and abs(f1.h) < 1e-12
    assert abs(f0.overlap - math.exp(-1.5)) < 1e-12 and abs(f0.J - 0.22313016014842982) < 1e-12
    assert abs(f0.b - 1.0) < 1e-12 and abs(f1.b - math.sqrt(3.0)) < 1e-12
    assert f1.kappa == 0.0 and f1.J > f0.J, "the Hat is the paramagnet: no coupling, full tunnelling"
    patch = build_substitution_tier(2)
    assign_qubits_to_patch(patch, max_qubits=None)
    assert abs(np.abs(f0.kappa * pair_couplings(patch)).max() - FLOW_K0) < 1e-12, "strongest coupling at tau = 0 is K0"
    kappas = [flow_parameters(t).kappa for t in np.linspace(0, 1, 11)]
    assert all(x > y for x, y in zip(kappas, kappas[1:])), "coupling falls along the flow"


def test_flow_agrees_with_toymodel_v6():
    # Reference numbers from spectre-monotile-py, research/toymodel_v6/t6_patch.py (independent
    # patch code, couplings recomputed at b(tau)): kappa() = 5.787156, and for the 10 central pairs
    # of generation 3 tau_eff = 0.244 and largest concurrence 0.407 at flow time 0 (b = 1, so
    # exact here), 0.647 at flow time 0.05 (b = 1.037: our b = 1 couplings are an approximation).
    assert abs(flow_kappa_norm() - 5.787156) < 1e-5, flow_kappa_norm()
    W = central_couplings(3, 10)
    rows = flow_scan(W, [0.0, 0.05])
    assert abs(rows[0]["tau_eff"] - 0.244) < 1e-3, rows[0]["tau_eff"]
    assert abs(rows[0]["max_concurrence"] - 0.407) < 1e-3, rows[0]["max_concurrence"]
    assert abs(rows[1]["max_concurrence"] - 0.647) < 0.015, rows[1]["max_concurrence"]
    assert 0.2 < rows[0]["tau_eff"] < rows[1]["tau_eff"] < 1.0, "the flow starts inside the entangled window"


def test_old_default_scale_was_far_from_the_flow():
    W = central_couplings(3, 10)
    old = tau_effective(W, 1.0, 0.3 ** 2)           # J = 1, kappa = (a h)^2 with h = 0.3
    new = tau_effective(W, flow_parameters(0.0).J, flow_parameters(0.0).kappa)
    assert old > 100 * new, (old, new)


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
    if "--tests-only" in sys.argv:
        sys.exit(1 if run_tests() else 0)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    W = central_couplings(3, n)
    print(f"{n} central pairs of generation 3: w = {coupling_scale(W):.4f}, "
          f"classical degeneracy {classical_degeneracy(W)}")
    for r in tau_scan(W, np.geomspace(0.02, 20, 14)):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
