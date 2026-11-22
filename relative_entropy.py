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
