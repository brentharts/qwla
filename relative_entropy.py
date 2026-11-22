#!/usr/bin/env python3
"""
Relative entropy of regions of the pair-qubit ground state (ideas taken from NariaiRelativeEntropy)

`NariaiRelativeEntropy` (spectre-monotile-py's sibling fork) builds gravity from the Araki-Umegaki
relative entropy S(rho || sigma) = Tr rho (ln rho - ln sigma) of a perturbed state against a reference,
and uses three things that carry over to the qubits of qwla:

 * the first law S_rel = Delta<K_sigma> - Delta S, with K_sigma = -ln sigma the modular Hamiltonian of
   the reference (there: S_rel = beta Delta E_boost); `first_law_terms` returns the three numbers;
 * positivity and monotonicity under restriction, S(rho_A || sigma_A) <= S(rho_B || sigma_B) for A in B,
   which are properties of the exact reduced states and are tested as such;
 * the discipline of computing it from matrices by an independent route, with the two numerical traps
   of `nariai_exact.fock_relative_entropy`: never clip the log of a vanishing eigenvalue of sigma (a
   rho with weight where sigma has none has infinite relative entropy, which is a result, not a
   rounding error), and drop eigenvalues of rho below the noise floor before x ln x amplifies them.

Reference state: the ground state of the flow Hamiltonian at flow time tau; perturbed state: the
ground state at tau + delta. Entropies are in nats (the Nariai paper's convention); the bit-based
entropies elsewhere in qwla are these divided by ln 2.
"""
from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import pair_hamiltonian as ph


# ---- the quantity -------------------------------------------------------------------------

def reduced_density(psi: np.ndarray, n: int, region: Sequence[int]) -> np.ndarray:
    """Reduced density matrix of the qubits in `region` (pure `psi` on n qubits)."""
    region = list(region)
    rest = [q for q in range(n) if q not in region]
    t = np.transpose(psi.reshape([2] * n), region + rest).reshape(2 ** len(region), -1)
    return t @ t.conj().T


def von_neumann(rho: np.ndarray, floor: float = 1e-14) -> float:
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > floor]
    return float(-np.sum(ev * np.log(ev)))


def relative_entropy(rho: np.ndarray, sigma: np.ndarray, floor: float = 1e-13, support_tol: float = 1e-9) -> float:
    """
    Umegaki relative entropy Tr rho (ln rho - ln sigma) in nats, from eigendecompositions.

    sigma's eigenvalues at or below `floor` are the kernel of sigma. If rho has weight there beyond
    `support_tol`, the relative entropy is infinite and `inf` is returned; the log of such an
    eigenvalue is never clipped to a finite number. Eigenvalues of rho below `floor` are dropped.
    """
    ev_r = np.linalg.eigvalsh(rho)
    keep = ev_r > floor
    tr_rho_ln_rho = float(np.sum(ev_r[keep] * np.log(ev_r[keep])))
    s, U = np.linalg.eigh(sigma)
    weights = np.real(np.einsum("ik,ij,jk->k", U.conj(), rho, U))          # <k| rho |k> in sigma's basis
    ker = s <= floor
    if np.sum(weights[ker]) > support_tol:
        return math.inf
    tr_rho_ln_sigma = float(np.sum(weights[~ker] * np.log(s[~ker])))
    val = tr_rho_ln_rho - tr_rho_ln_sigma
    return 0.0 if -1e-12 < val < 0.0 else val           # a clearly negative value would be a bug and is returned as such


def first_law_terms(rho: np.ndarray, sigma: np.ndarray, floor: float = 1e-13) -> Dict[str, float]:
    """
    S_rel computed directly, and from the first law Delta<K_sigma> - Delta S with K_sigma = -ln sigma.
    The two use different expressions of the same eigendecompositions; `residual` is their difference.
    """
    s, U = np.linalg.eigh(sigma)
    ok = s > floor
    K = (U[:, ok] * (-np.log(s[ok]))) @ U[:, ok].conj().T
    dK = float(np.real(np.trace((rho - sigma) @ K)))
    dS = von_neumann(rho) - von_neumann(sigma)
    direct = relative_entropy(rho, sigma, floor)
    return {"S_rel": direct, "delta_K": dK, "delta_S": dS, "first_law": dK - dS, "residual": direct - (dK - dS)}


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(rho - sigma))))


# ---- regions of the flow ground state -----------------------------------------------------

def flow_ground_state(generation: int, n: int, flow_tau: float) -> np.ndarray:
    fp = ph.flow_parameters(flow_tau)
    return ph.ground_state(ph.flow_couplings(generation, n, flow_tau), fp.J, 1.0)[1]


def region_relative_entropies(generation: int, n: int, flow_tau: float, delta: float,
                              regions: Sequence[Sequence[int]]) -> np.ndarray:
    """S(rho_A(tau + delta) || rho_A(tau)) for each region, in nats."""
    ref = flow_ground_state(generation, n, flow_tau)
    new = flow_ground_state(generation, n, flow_tau + delta)
    return np.array([relative_entropy(reduced_density(new, n, r), reduced_density(ref, n, r)) for r in regions])


def fit_boundary_volume(y: np.ndarray, B: np.ndarray, V: np.ndarray) -> Dict[str, float]:
    """Least squares of y on the boundary B alone, the volume V alone and both (with a constant)."""
    def lsq(cols):
        X = np.column_stack(cols + [np.ones_like(y)])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return c, float(np.sqrt(np.mean((y - X @ c) ** 2)))
    cb, rb = lsq([B]); cv, rv = lsq([V]); cbv, rbv = lsq([B, V])
    return {"rms_boundary": rb, "rms_volume": rv, "rms_both": rbv, "alpha": float(cbv[0]), "beta": float(cbv[1]),
            "std_alpha": float(cbv[0] * B.std()), "std_beta": float(cbv[1] * V.std()),
            "corr_B_V": float(np.corrcoef(B, V)[0, 1]), "corr_y_B": float(np.corrcoef(y, B)[0, 1]),
            "corr_y_V": float(np.corrcoef(y, V)[0, 1])}


def area_ledger_change(flow_tau: float, delta: float, n_pairs_in_region: int) -> float:
    """Change of the total tile area of `n_pairs_in_region` pairs when b moves from b(tau) to b(tau + delta): 2 m dA(1, b)."""
    from quantumspectremodel import tile_area_ab
    b0, b1 = ph.flow_b(flow_tau), ph.flow_b(flow_tau + delta)
    return 2.0 * n_pairs_in_region * (tile_area_ab(1.0, b1) - tile_area_ab(1.0, b0))


# ---- checks -------------------------------------------------------------------------------

def _random_state(d: int, rank: int, rng) -> np.ndarray:
    A = rng.normal(size=(d, rank)) + 1j * rng.normal(size=(d, rank))
    r = A @ A.conj().T
    return r / np.trace(r).real


def test_classical_and_qubit_values():
    p, q = 0.8, 0.3
    got = relative_entropy(np.diag([p, 1 - p]), np.diag([q, 1 - q]))
    assert abs(got - (p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)))) < 1e-14
    assert relative_entropy(np.eye(2) / 2, np.eye(2) / 2) == 0.0
    # a pure state against the maximally mixed state: ln 2
    assert abs(relative_entropy(np.diag([1.0, 0.0]), np.eye(2) / 2) - math.log(2)) < 1e-13


def test_first_law_identity_positivity_and_pinsker():
    rng = np.random.default_rng(1)
    for d in (2, 4, 8):
        for _ in range(5):
            rho, sigma = _random_state(d, d, rng), _random_state(d, d, rng)
            t = first_law_terms(rho, sigma)
            assert abs(t["residual"]) < 1e-10 and t["S_rel"] > 0, t
            assert t["S_rel"] >= 2 * trace_distance(rho, sigma) ** 2 - 1e-12       # Pinsker, in nats
    assert abs(relative_entropy(rho, rho)) < 1e-12


def test_support_violation_is_infinite_not_clipped():
    sigma = np.diag([1.0, 0.0])
    assert relative_entropy(np.diag([0.5, 0.5]), sigma) == math.inf
    assert math.isfinite(relative_entropy(np.diag([1.0, 0.0]), np.diag([0.5, 0.5])))
    # rho inside the support of a rank-deficient sigma is finite
    assert abs(relative_entropy(np.diag([0.25, 0.75, 0.0]), np.diag([0.5, 0.5, 0.0])) - (0.25 * math.log(0.5) + 0.75 * math.log(1.5))) < 1e-14


def test_monotone_under_restriction_on_random_pure_states():
    rng = np.random.default_rng(2)
    n = 7                                       # regions up to 3 qubits are full rank, so every value is finite
    a, b = (rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n) for _ in range(2))
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    chain = [[0], [0, 1], [0, 1, 2]]
    vals = [relative_entropy(reduced_density(a, n, r), reduced_density(b, n, r)) for r in chain]
    assert all(x <= y + 1e-12 for x, y in zip(vals, vals[1:])) and vals[0] > 0, vals
    # the check can fail: tracing out in a wrong (non-nested) way is not required to be monotone, but a
    # broken partial trace (here: the diagonal only) changes the numbers
    wrong = relative_entropy(np.diag(np.diag(reduced_density(a, n, chain[1]))), reduced_density(b, n, chain[1]))
    assert abs(wrong - vals[2]) > 1e-6


def test_flow_ground_states_are_second_order_and_monotone():
    n = 8
    regions = [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]
    s1 = region_relative_entropies(3, n, 0.05, 0.004, regions)
    s2 = region_relative_entropies(3, n, 0.05, 0.002, regions)
    assert np.all(np.diff(s1) >= -1e-12) and np.all(s1 > 0)                       # monotone under inclusion
    ratio = s1 / s2
    assert np.all((ratio > 3.5) & (ratio < 4.5)), ratio                          # S_rel ~ delta^2 (Fisher metric)
    first = np.abs(area_ledger_change(0.05, 0.004, 1) / area_ledger_change(0.05, 0.002, 1))
    assert 1.9 < first < 2.1                                                     # the area change is first order



# ---- area observables of the pair qubits against the relative entropy ----------------------
#
# The Nariai relation compares two quantities that are both quadratic in the amplitude of an
# excitation. The model's own area observables are: the bump tile of a pair, A0 + h Z_i (linear in
# the bit, exactly: `quantumspectremodel.set_pair_bit`); its partner, A0 - h Z_i; their sum, which is
# the neutral ledger and is constant for every state; and the squared deviation of the region's bump
# area, Q_A = (h sum_{i in A} Z_i)^2, which is the only one that is quadratic in the state amplitude.
# `rotate_region` makes a family of excitations |psi(alpha)> = exp(-i alpha sum_i c_i Y_i)|psi_0> (a
# local unitary on the region, so S(rho_A) is unchanged and S_rel = Delta<K_A> exactly).

def rotate_region(psi: np.ndarray, n: int, region: Sequence[int], alpha: float, c: Sequence[float]) -> np.ndarray:
    """Apply exp(-i alpha c_q Y_q) (a real rotation) to each qubit q of `region`."""
    out = psi.copy()
    for q, cq in zip(region, c):
        t = out.reshape([2] * n)
        t0, t1 = np.take(t, 0, axis=q), np.take(t, 1, axis=q)
        co, si = math.cos(alpha * cq), math.sin(alpha * cq)
        out = np.stack([co * t0 + si * t1, -si * t0 + co * t1], axis=q).reshape(-1)
    return out


def z_table(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    return np.array([1.0 - 2.0 * ((idx >> (n - 1 - q)) & 1) for q in range(n)])


def area_observables(psi0: np.ndarray, psi: np.ndarray, n: int, region: Sequence[int], h: float) -> Dict[str, float]:
    """Change between two states of the area observables of the pairs in `region` (units of h)."""
    Z = z_table(n)
    reg = list(region)
    p0, p1 = np.abs(psi0) ** 2, np.abs(psi) ** 2
    dz = (Z[reg] @ p1) - (Z[reg] @ p0)
    q0, q1 = (h * Z[reg].sum(axis=0)) ** 2 @ p0, (h * Z[reg].sum(axis=0)) ** 2 @ p1
    return {"dA_bump": float(h * dz.sum()), "dA_partner": float(-h * dz.sum()), "dA_pair_total": 0.0, "dQ": float(q1 - q0)}


def modular_hamiltonian(rho: np.ndarray, floor: float = 1e-13) -> np.ndarray:
    w, U = np.linalg.eigh(rho)
    return (U[:, w > floor] * (-np.log(w[w > floor]))) @ U[:, w > floor].conj().T


def region_hamiltonian(K: np.ndarray, J: float, region: Sequence[int]) -> np.ndarray:
    """-J sum X + sum K_pq Z_p Z_q restricted to the qubits of `region` (the operator H_A)."""
    m = len(region)
    X, Z, I = np.array([[0, 1], [1, 0.0]]), np.diag([1.0, -1.0]), np.eye(2)

    def op(d):
        M = np.array([[1.0]])
        for q in range(m):
            M = np.kron(M, d.get(q, I))
        return M
    H = -J * sum(op({q: X}) for q in range(m))
    for p in range(m):
        for q in range(p + 1, m):
            H = H + K[region[p], region[q]] * op({p: Z, q: Z})
    return H


def modular_fit(KA: np.ndarray, HA: np.ndarray) -> Dict[str, float]:
    """K_A = beta H_A + const: the best beta and the relative residual (0 would be Bisognano-Wichmann-like)."""
    d = KA.shape[0]
    Kc, Hc = KA - np.trace(KA) / d * np.eye(d), HA - np.trace(HA) / d * np.eye(d)
    beta = float(np.real(np.trace(Kc @ Hc) / np.trace(Hc @ Hc)))
    return {"beta_eff": beta, "residual": float(np.linalg.norm(Kc - beta * Hc) / np.linalg.norm(Kc))}


def geometry_area_change(generation: int, n: int, flow_tau: float, z_before: np.ndarray, z_after: np.ndarray,
                         region: Sequence[int]) -> Dict[str, float]:
    """
    The same area changes read from the actual tiles: set each pair's bit to <Z> (the displacement of
    P8 is z h, `set_pair_bit`) in the patch built at b(flow_tau) and add up tile areas.
    """
    from quantumspectremodel import assign_qubits_to_patch, build_substitution_tier, pair_qubit_tiles, set_pair_bit
    fp = ph.flow_parameters(flow_tau)
    patch = build_substitution_tier(generation, b=fp.b)
    assign_qubits_to_patch(patch, max_qubits=n)
    tiles = pair_qubit_tiles(patch)

    def areas(z):
        a = b = 0.0
        for q in region:
            A, B = tiles[q]
            set_pair_bit(A, B, float(z[q]), fp.h)
            a, b = a + A.area(), b + B.area()
        return a, b
    a0, b0 = areas(z_before)
    a1, b1 = areas(z_after)
    return {"dA_bump": a1 - a0, "dA_partner": b1 - b0, "dA_pair_total": (a1 + b1) - (a0 + b0)}


def test_local_unitary_first_law_and_quadratic_scaling():
    n, flow = 8, 0.05
    psi0 = flow_ground_state(3, n, flow)
    h = ph.flow_parameters(flow).h
    region, c = (0, 1, 2), [0.8, 1.3, -0.6]
    rows = []
    for a in (0.04, 0.02):
        ps = rotate_region(psi0, n, region, a, c)
        r1, r0 = reduced_density(ps, n, region), reduced_density(psi0, n, region)
        t = first_law_terms(r1, r0)
        assert abs(t["delta_S"]) < 1e-10 and abs(t["residual"]) < 1e-9          # unitary inside A: S_rel = Delta<K_A>
        rows.append((t["S_rel"], area_observables(psi0, ps, n, region, h)))
    assert 3.8 < rows[0][0] / rows[1][0] < 4.2                                      # S_rel ~ alpha^2
    assert 1.9 < rows[0][1]["dA_bump"] / rows[1][1]["dA_bump"] < 2.1                # the bump-tile area ~ alpha
    assert 3.8 < rows[0][1]["dQ"] / rows[1][1]["dQ"] < 4.2                          # the squared deviation ~ alpha^2


def test_tile_geometry_reproduces_the_area_observables_and_the_neutral_ledger():
    n, flow = 6, 0.05
    psi0 = flow_ground_state(3, n, flow)
    fp = ph.flow_parameters(flow)
    ps = rotate_region(psi0, n, (0, 1, 2), 0.05, [1.0, 0.7, -1.1])
    Z = z_table(n)
    z0, z1 = Z @ (np.abs(psi0) ** 2), Z @ (np.abs(ps) ** 2)
    geo = geometry_area_change(3, n, flow, z0, z1, (0, 1, 2))
    mod = area_observables(psi0, ps, n, (0, 1, 2), fp.h)
    assert abs(geo["dA_bump"] - mod["dA_bump"]) < 1e-9 and abs(geo["dA_partner"] + mod["dA_bump"]) < 1e-9
    assert abs(geo["dA_pair_total"]) < 1e-9 and abs(mod["dA_bump"]) > 1e-3        # neutral in total, not in the parts


def test_modular_hamiltonian_is_not_the_region_hamiltonian():
    n, flow = 8, 0.05
    psi0 = flow_ground_state(3, n, flow)
    fp = ph.flow_parameters(flow)
    K = ph.flow_couplings(3, n, flow)
    for region in ((0, 1), (0, 1, 2, 3)):
        KA = modular_hamiltonian(reduced_density(psi0, n, region))
        fit = modular_fit(KA, region_hamiltonian(K, fp.J, region))
        assert fit["beta_eff"] > 0 and 0.05 < fit["residual"] < 0.9, fit            # positive weight, not proportional



# ---- a thermal reference: where does the Nariai first law hold? ---------------------------------
#
# With a Gibbs reference sigma = e^{-beta H}/Z and a unitary excitation, Delta S = 0 and the first law
# is exact for the whole system: S_rel = beta Delta<H>. The Nariai relation is that statement for a
# region (K_A = beta H_A, Bisognano-Wichmann). `thermal_first_law` measures how far it holds in the
# model as the temperature falls.

def gibbs_state(K: np.ndarray, J: float, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense Gibbs state of H = -J sum X + sum K ZZ; returns (sigma, H, eigenvalues). Small n only."""
    H = ph.build_hamiltonian(K, J, 1.0).toarray()
    w, U = np.linalg.eigh(H)
    p = np.exp(-beta * (w - w[0]))
    p /= p.sum()
    return (U * p) @ U.T, H, w


def reduce_mixed(rho: np.ndarray, n: int, region: Sequence[int]) -> np.ndarray:
    reg = list(region)
    rest = [q for q in range(n) if q not in reg]
    t = rho.reshape([2] * (2 * n)).transpose(reg + rest + [n + q for q in reg] + [n + q for q in rest])
    t = t.reshape(2 ** len(reg), 2 ** len(rest), 2 ** len(reg), 2 ** len(rest))
    return np.einsum("aibi->ab", t)


def rotation_matrix(n: int, region: Sequence[int], alpha: float, c: Sequence[float]) -> np.ndarray:
    M = np.eye(2 ** n)
    for q, cq in zip(region, c):
        R = np.array([[math.cos(alpha * cq), math.sin(alpha * cq)], [-math.sin(alpha * cq), math.cos(alpha * cq)]])
        M = np.kron(np.kron(np.eye(2 ** q), R), np.eye(2 ** (n - q - 1))) @ M
    return M


def thermal_first_law(generation: int, n: int, flow_tau: float, beta: float, region: Sequence[int],
                      alpha: float = 0.03, c: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """
    Global S_rel / (beta Delta E) (exactly 1), regional S_rel / (beta Delta E_A), and the fit
    K_A = beta_eff H_A + const of the reference's reduced state, for a local unitary on `region`.
    """
    fp = ph.flow_parameters(flow_tau)
    K = ph.flow_couplings(generation, n, flow_tau)
    c = list(c) if c is not None else [1.0, 0.7, -0.6, 1.2][: len(region)]
    sigma, H, _ = gibbs_state(K, fp.J, beta)
    V = rotation_matrix(n, region, alpha, c)
    rho = V @ sigma @ V.T
    HA = region_hamiltonian(K, fp.J, region)
    sA, rA = reduce_mixed(sigma, n, region), reduce_mixed(rho, n, region)
    S_glob = relative_entropy(rho, sigma, floor=1e-14)
    dE = float(np.trace((rho - sigma) @ H))
    S_A, dEA = relative_entropy(rA, sA, floor=1e-14), float(np.trace((rA - sA) @ HA))
    fit = modular_fit(modular_hamiltonian(sA), HA)
    return {"beta": beta, "global_ratio": S_glob / (beta * dE) if math.isfinite(S_glob) else math.nan,
            "regional_ratio": S_A / (beta * dEA) if dEA else math.nan, "S_A": S_A, "dE_A": dEA, **fit}


def test_thermal_reference_first_law_global_exact_regional_only_when_hot():
    kw = dict(generation=3, n=6, flow_tau=0.05, region=(0, 1, 2))
    hot = thermal_first_law(beta=0.2, **kw)
    mid = thermal_first_law(beta=1.0, **kw)
    cold = thermal_first_law(beta=5.0, **kw)
    assert all(abs(t["global_ratio"] - 1.0) < 1e-6 for t in (hot, mid, cold))      # Delta S = 0: exact, any beta
    assert abs(hot["regional_ratio"] - 1.0) < 0.05 and hot["residual"] < 0.05       # K_A ~ beta H_A when hot
    assert abs(hot["beta_eff"] - 0.2) < 0.02
    assert abs(mid["regional_ratio"] - 1.0) > 0.05                                  # and it degrades as it cools
    assert cold["regional_ratio"] < 0.9 and cold["residual"] > 0.2 and cold["beta_eff"] < 0.9 * 5.0



# ---- the structure of the modular Hamiltonian in the cold (ground-state) regime -----------------
#
# `thermal_first_law` shows K_A = beta H_A when the state is hot. In the ground state K_A is still
# mostly one- and two-body (X and ZZ, the terms of H_A) but with position dependent weights; the
# Bisognano-Wichmann picture would make the weight of X_i a local inverse temperature that grows with
# the depth of qubit i inside the region. `local_beta_profile` measures that.

_PAULI = [np.eye(2), np.array([[0, 1], [1, 0.0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]   # I, X, Y, Z


def pauli_coefficients(M: np.ndarray, m: int) -> Dict[Tuple[int, ...], float]:
    """Coefficients c_s = Tr(P_s M)/2^m of a Hermitian M on m qubits in the Pauli basis (0=I, 1=X, 2=Y, 3=Z)."""
    import itertools
    out: Dict[Tuple[int, ...], float] = {}
    for s in itertools.product(range(4), repeat=m):
        P = np.array([[1.0]])
        for k in s:
            P = np.kron(P, _PAULI[k])
        out[s] = float(np.real(np.trace(P.conj().T @ M)) / 2 ** m)
    return out


def pauli_weight_fractions(c: Dict[Tuple[int, ...], float]) -> Dict[int, float]:
    """Share of the squared norm of the traceless part carried by strings of each weight (non-identity sites)."""
    tot = sum(v * v for k, v in c.items() if any(k))
    out: Dict[int, float] = {}
    for k, v in c.items():
        if any(k):
            w = sum(1 for x in k if x)
            out[w] = out.get(w, 0.0) + v * v / tot
    return dict(sorted(out.items()))


def local_beta_profile(psi: np.ndarray, n: int, regions: Sequence[Sequence[int]], mids: np.ndarray, J: float) -> np.ndarray:
    """
    Rows (depth, beta_loc, region size): beta_loc = c_X / (-J) for each qubit of each region, where c_X is the
    coefficient of X_i in K_A = -ln rho_A, and depth is the distance from the pair's position to the nearest pair outside.
    """
    rows = []
    for reg in regions:
        reg = list(reg)
        m = len(reg)
        c = pauli_coefficients(modular_hamiltonian(reduced_density(psi, n, reg)), m)
        outside = [q for q in range(n) if q not in reg]
        for i, q in enumerate(reg):
            key = tuple(1 if j == i else 0 for j in range(m))
            rows.append((min(float(np.linalg.norm(mids[q] - mids[o])) for o in outside), c[key] / (-J), m))
    return np.array(rows)


def test_pauli_expansion_reconstructs_and_hot_beta_loc_equals_beta():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    M = A + A.conj().T
    c = pauli_coefficients(M, 3)
    import itertools
    R = sum(v * np.kron(np.kron(_PAULI[s[0]], _PAULI[s[1]]), _PAULI[s[2]]) for s, v in c.items())
    assert np.allclose(R, M, atol=1e-12)
    # a hot Gibbs state of the region itself has K_A = beta H_A, so c_X / (-J) = beta exactly
    fp = ph.flow_parameters(0.05)
    K = ph.flow_couplings(3, 4, 0.05)
    H = region_hamiltonian(K, fp.J, (0, 1, 2, 3))
    w, U = np.linalg.eigh(H)
    for beta in (0.05, 0.3):
        p = np.exp(-beta * w)
        rho = (U * (p / p.sum())) @ U.T
        cc = pauli_coefficients(modular_hamiltonian(rho), 4)
        assert all(abs(cc[tuple(1 if j == i else 0 for j in range(4))] / (-fp.J) - beta) < 1e-9 for i in range(4))


def test_ground_state_modular_hamiltonian_depth_profile():
    from scipy.stats import spearmanr
    from entanglement_geometry import build_cluster, compact_regions
    n = 12
    _, _, mids, _ = build_cluster(3, n)
    psi = flow_ground_state(3, n, 0.05)
    fp = ph.flow_parameters(0.05)
    regs = [r for r in compact_regions(mids, 5) if len(r) in (4, 5)]
    prof = local_beta_profile(psi, n, regs, mids, fp.J)
    assert len(prof) > 60
    rho_s = spearmanr(prof[:, 0], prof[:, 1])[0]
    assert rho_s > 0.4, rho_s                                                        # deeper qubits: larger local beta
    rng = np.random.default_rng(0)
    null = [spearmanr(rng.permutation(prof[:, 0]), prof[:, 1])[0] for _ in range(100)]
    assert rho_s > 4 * np.std(null)
    KA = modular_hamiltonian(reduced_density(psi, n, regs[0]))
    fr = pauli_weight_fractions(pauli_coefficients(KA, len(regs[0])))
    assert fr[1] + fr[2] > 0.7, fr                                                    # mostly one- and two-body



# ---- depth profiles of the X and ZZ weights, and the size of the region -------------------------
#
# `xz_coefficient` reads single coefficients of K_A without expanding the whole Pauli basis, so regions
# up to half of 16 pairs are affordable. Depth of a pair inside a region is measured three ways:
# Euclidean distance to the nearest pair outside, Delaunay hops to the outside, and the ledger depth
# (the tile area of the region's pairs that are no deeper than it, a monotone radial rank in units of
# 2 A(1, b)).

def xz_coefficient(KA: np.ndarray, m: int, xs: Sequence[int] = (), zs: Sequence[int] = ()) -> float:
    """Tr(P K_A) / 2^m for P = prod_{i in xs} X_i prod_{j in zs} Z_j (qubit 0 is the most significant bit)."""
    d = 2 ** m
    b = np.arange(d)
    flip = 0
    for i in xs:
        flip |= 1 << (m - 1 - i)
    sign = np.ones(d)
    for j in zs:
        sign = sign * (1 - 2 * ((b >> (m - 1 - j)) & 1))
    return float(np.real(np.sum(sign * KA[b, b ^ flip])) / d)


def hops_to_outside(links: np.ndarray, region: Sequence[int]) -> Dict[int, int]:
    """Graph distance in the Delaunay links from each pair of `region` to the nearest pair outside it."""
    from collections import deque
    n = links.shape[0]
    inside = set(int(r) for r in region)
    dist = {o: 0 for o in range(n) if o not in inside}
    queue = deque(dist)
    while queue:
        u = queue.popleft()
        for v in np.nonzero(links[u])[0]:
            if int(v) not in dist:
                dist[int(v)] = dist[u] + 1
                queue.append(int(v))
    return {int(r): dist.get(int(r), 10 ** 6) for r in region}


def weight_profiles(generation: int, n: int, flow_tau: float, sizes: Sequence[int], zz_min: float = 0.25):
    """
    Local weights of K_A against depth for every compact region of the given sizes.
    Returns (X, ZZ): X rows (m, euclid, hops, ledger, beta_loc = c_X/(-J)); ZZ rows (m, euclid, hops, ledger,
    beta_zz = c_ZZ/K_pq, K_pq) for Delaunay-linked pairs with |K_pq| > zz_min max|K|; depth of a link is that of
    its two ends (mean of the Euclidean distances and hops, the ledger of the deeper end's complement).
    """
    from entanglement_geometry import build_cluster, compact_regions, delaunay_links
    from quantumspectremodel import tile_area_ab
    fp = ph.flow_parameters(flow_tau)
    K = ph.flow_couplings(generation, n, flow_tau)
    psi = flow_ground_state(generation, n, flow_tau)
    _, _, mids, _ = build_cluster(generation, n)
    links = delaunay_links(mids)
    A_pair = 2.0 * tile_area_ab(1.0, fp.b)
    kmax = float(np.abs(K).max())
    X, ZZ = [], []
    for reg in compact_regions(mids, max(sizes)):
        m = len(reg)
        if m not in sizes:
            continue
        KA = modular_hamiltonian(reduced_density(psi, n, reg))
        hop = hops_to_outside(links, reg)
        outside = [q for q in range(n) if q not in reg]
        dE = {q: min(float(np.linalg.norm(mids[q] - mids[o])) for o in outside) for q in reg}
        for i, q in enumerate(reg):
            led = A_pair * sum(1 for r in reg if dE[r] <= dE[q])
            X.append((m, dE[q], hop[q], led, xz_coefficient(KA, m, xs=(i,)) / (-fp.J)))
        for i in range(m):
            for j in range(i + 1, m):
                p, q = reg[i], reg[j]
                if links[p, q] and abs(K[p, q]) > zz_min * kmax:
                    led = A_pair * sum(1 for r in reg if dE[r] <= max(dE[p], dE[q]))
                    ZZ.append((m, 0.5 * (dE[p] + dE[q]), 0.5 * (hop[p] + hop[q]), led,
                               xz_coefficient(KA, m, zs=(i, j)) / K[p, q], K[p, q]))
    return np.array(X), np.array(ZZ)


def test_xz_coefficient_matches_the_full_pauli_expansion():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    M = A + A.conj().T
    c = pauli_coefficients(M, 4)
    for xs, zs in (((1,), ()), ((), (0, 2)), ((0,), (3,)), ((0, 2), (1,)), ((), ())):
        key = tuple(1 if (i in xs and i not in zs) else 3 if (i in zs and i not in xs) else 0 for i in range(4))
        if set(xs) & set(zs):
            continue                                            # X and Z on one site is Y: not an X/Z string
        assert abs(xz_coefficient(M, 4, xs, zs) - c[key]) < 1e-12, (xs, zs)


def test_hops_to_outside_on_a_path():
    links = np.zeros((6, 6), bool)
    for i in range(5):
        links[i, i + 1] = links[i + 1, i] = True
    assert hops_to_outside(links, [1, 2, 3, 4]) == {1: 1, 2: 2, 3: 2, 4: 1}


def test_depth_profile_holds_below_half_the_system_and_reverses_above_it_at_12_and_16():
    from scipy.stats import spearmanr
    for n, below, above in ((12, (4, 5), 7), (16, (4, 5), 8)):
        X, ZZ = weight_profiles(3, n, 0.05, list(below) + [above])
        for m in below:
            mk = X[:, 0] == m
            assert X[mk, 4].mean() > 1.5, (n, m)                                   # positive local inverse temperature
            assert spearmanr(X[mk, 1], X[mk, 4])[0] > 0.5, (n, m)                  # growing with depth
        mk = X[:, 0] == above
        assert X[mk, 4].mean() < 0 and spearmanr(X[mk, 1], X[mk, 4])[0] < -0.5, (n, above)   # reversed beyond half
    mz = ZZ[:, 0] == 4                                                              # (n = 16 here; m = 6 of 16 is the crossover)
    assert 2.0 < ZZ[mz, 4].mean() < 7.0                                             # ZZ weight is of the same order as X


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


# ---- the measurement ----------------------------------------------------------------------

def measure(generation: int, n: int, flow_tau: float = 0.05, delta: float = 0.004, seed: int = 0) -> Dict[str, object]:
    from entanglement_geometry import boundary_size, build_cluster, compact_regions, delaunay_links, random_regions
    _, _, mids, _ = build_cluster(generation, n)
    links = delaunay_links(mids)
    out: Dict[str, object] = {"n": n, "flow_tau": flow_tau, "delta": delta}
    for label, regions in (("compact", compact_regions(mids, n // 2)), ("random", random_regions(n, range(2, n // 2 + 1), 250, seed))):
        y = region_relative_entropies(generation, n, flow_tau, delta, regions)
        y = np.where(np.isfinite(y), y, np.nan)
        ok = np.isfinite(y)
        B = np.array([boundary_size(r, links) for r in regions], float)[ok]
        V = np.array([len(r) for r in regions], float)[ok]
        out[label] = {"n_regions": int(ok.sum()), **fit_boundary_volume(y[ok], B, V), "mean_S_rel": float(np.mean(y[ok]))}
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("--tests-only",):
        for n in (int(x) for x in sys.argv[1:]):
            print(measure(3, n))
    else:
        sys.exit(1 if run_tests() else 0)
