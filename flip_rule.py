#!/usr/bin/env python3
"""
Stage 3 of ROADMAP_QUANTUM_GRAVITY.md: a flip rule on the pair qubits.

A phason flip (wada-coudene, SPECTRE_FLIPS.md) changes the set of tiles present in a region. A
bump-and-dent pair is a property of that set (two tiles that meet at their flat vertex P8 alone), so a
flip can destroy a pair, create one, or leave it. The qubit sits on the pair, so the qubit needs a rule.
This module has

  * `find_pairs_keys`    the pair rule of `quantumspectremodel.find_pairs` on any list of exact vertex keys
                         (it is the same function of the vertex keys, so it applies to placed tiles of the
                         flip engine as well as to substitution patches),
  * `pair_W`             the dipole couplings of an arbitrary set of pairs (same formula as `pair_couplings`),
  * `H_present`          H(pi) = -J sum_{p in pi} X_p + kappa sum_{p<q in pi} W_pq Z_p Z_q  for a presence set pi,
  * three flip rules on the register of all slots, with the new state of the qubits:
      dormant   the qubit of a destroyed pair is kept (frozen); the rule is the identity on the register and
                a permutation of the presence flags (a unitary on register x presence),
      reset     the qubit of a destroyed pair is replaced by |+>  (Kraus K_m = |+><m|),
      trace     destroyed qubits are traced out and created ones appended in |+>  (the register changes),
  * measures: energy after the flip, overlap with the new ground state, entropy, reversibility.

Created pairs start in |+> (ground state of one pair under -J X); a slot that has never been present
holds |+> as well. Dense simulation, so at most about 10 slots.

Run `python3 flip_rule.py --tests-only` for the checks.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

FLAT_EXACT = 10          # the flat vertex P8 of the model is vertex 10 of the exact outline (quantumspectremodel.SHIFT)

PLUS = np.array([1.0, 1.0]) / math.sqrt(2.0)


# ---- pairs from vertex keys ---------------------------------------------------------------

def find_pairs_keys(tiles: Sequence[Sequence[Tuple[int, ...]]]) -> List[Tuple[int, int]]:
    """Pairs of tile indices (i < j) whose flat vertices (index 10 of the exact outline) coincide and meet no other
    vertex of any tile. The rule of `quantumspectremodel.find_pairs`, written on the keys."""
    at: Dict[Tuple[int, ...], List[Tuple[int, int]]] = defaultdict(list)
    for t, keys in enumerate(tiles):
        for v, key in enumerate(keys):
            at[tuple(key)].append((t, v))
    out = []
    for lst in at.values():
        if len(lst) == 2 and all(v == FLAT_EXACT for _, v in lst):
            i, j = sorted(t for t, _ in lst)
            out.append((i, j))
    return sorted(out)


def centroid_xy(pts: np.ndarray) -> np.ndarray:
    """Centroid of the straight outline (the formula of `quantumspectremodel._centroid`)."""
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (3.0 * cr.sum())


def ordered_centroids(cA: np.ndarray, cB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Canonical roles of a pair: the tile with the smaller (x, y) centroid is A. A change of role maps Z -> -Z on
    that pair, i.e. a conjugation by X_p, which leaves -J X and the spectrum unchanged, so the choice is a gauge;
    it must only be the same before and after a flip, which a rule on the centroids guarantees."""
    return (cA, cB) if (cA[0], cA[1]) <= (cB[0], cB[1]) else (cB, cA)


def pair_W(cents: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """W_pq = 1/r_AA' - 1/r_AB' - 1/r_BA' + 1/r_BB' for pairs given by the centroids of their two tiles."""
    n = len(cents)
    W = np.zeros((n, n))
    inv = lambda X, Y: 1.0 / np.linalg.norm(np.asarray(X) - np.asarray(Y))
    for p in range(n):
        for q in range(p + 1, n):
            Ap, Bp = cents[p]
            Aq, Bq = cents[q]
            W[p, q] = W[q, p] = inv(Ap, Aq) - inv(Ap, Bq) - inv(Bp, Aq) + inv(Bp, Bq)
    return W


# ---- the Hamiltonian of a presence set ---------------------------------------------------

def _bit(n: int, idx: np.ndarray, p: int) -> np.ndarray:
    return (idx >> (n - 1 - p)) & 1


def H_present(W: np.ndarray, present: Sequence[bool], J: float, kappa: float) -> np.ndarray:
    """Dense H on all n slots; an absent slot has no term (it is the identity there)."""
    n = W.shape[0]
    pres = np.asarray(present, bool)
    dim = 2 ** n
    idx = np.arange(dim)
    z = [1.0 - 2.0 * _bit(n, idx, p) for p in range(n)]
    diag = np.zeros(dim)
    for p in range(n):
        if not pres[p]:
            continue
        for q in range(p + 1, n):
            if pres[q] and W[p, q] != 0.0:
                diag += W[p, q] * z[p] * z[q]
    H = np.diag(kappa * diag)
    for p in range(n):
        if pres[p]:
            H[idx, idx ^ (1 << (n - 1 - p))] += -J
    return H


def ground_state_present(W, present, J, kappa) -> Tuple[float, np.ndarray]:
    """Ground energy and state of H(pi) on all slots, with every absent slot in |+> (its term-free, initial state)."""
    n = W.shape[0]
    pres = np.asarray(present, bool)
    idx_p = [p for p in range(n) if pres[p]]
    Wp = W[np.ix_(idx_p, idx_p)]
    m = len(idx_p)
    if m == 0:
        e, c = 0.0, np.ones(1)
    else:
        Hm = H_present(Wp, [True] * m, J, kappa)
        w, v = np.linalg.eigh(Hm)
        e, c = float(w[0]), v[:, 0]
    return e, embed_product(n, idx_p, c)


def embed_product(n: int, slots: Sequence[int], c: np.ndarray) -> np.ndarray:
    """State of n slots: `c` on the listed slots (in that order), |+> on the others."""
    psi = np.ones(1)
    k = 0
    for p in range(n):
        if p in slots:
            continue
        psi = np.kron(psi, PLUS)
    # build via tensor then permute axes: order is (others..., slots...) -> original order
    full = np.kron(psi, c) if len(slots) else psi
    others = [p for p in range(n) if p not in slots]
    order = others + list(slots)
    full = full.reshape([2] * n)
    inv = np.argsort(order)
    return np.transpose(full, inv).reshape(-1)


# ---- density matrices and partial trace --------------------------------------------------

def rho_of(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conj())


def ptrace(rho: np.ndarray, n: int, keep: Sequence[int]) -> np.ndarray:
    keep = list(keep)
    tr = [p for p in range(n) if p not in keep]
    r = rho.reshape([2] * (2 * n))
    # contract traced axes pairwise
    cur = n
    axes_pos = list(range(2 * n))
    for p in sorted(tr, reverse=True):
        r = np.trace(r, axis1=p, axis2=p + cur)
        cur -= 1
    d = 2 ** len(keep)
    return r.reshape(d, d)


def vn_entropy(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > 1e-12]
    return float(-(w * np.log2(w)).sum())


def mutual_information(rho: np.ndarray, n: int, p: int, q: int) -> float:
    a = vn_entropy(ptrace(rho, n, [p])) + vn_entropy(ptrace(rho, n, [q])) - vn_entropy(ptrace(rho, n, [p, q]))
    return a


# ---- the three flip rules ----------------------------------------------------------------

def kraus_reset(n: int, p: int) -> List[np.ndarray]:
    """K_m = |+><m| on slot p, identity elsewhere (m = 0, 1). Sum K^dagger K = 1."""
    ops = []
    for m in (0, 1):
        k = np.zeros((2, 2))
        k[:, m] = PLUS
        full = np.array([[1.0]])
        for s in range(n):
            full = np.kron(full, k if s == p else np.eye(2))
        ops.append(full)
    return ops


def rule_dormant(rho: np.ndarray, n: int, destroyed: Sequence[int], created: Sequence[int]) -> np.ndarray:
    """Identity on the register: the qubit of a destroyed pair is frozen, a created slot wakes in the state it holds."""
    return rho.copy()


def rule_reset(rho: np.ndarray, n: int, destroyed: Sequence[int], created: Sequence[int]) -> np.ndarray:
    """Destroyed qubits are replaced by |+>; created slots already hold |+> (the initial state of an absent slot)."""
    r = rho
    for p in destroyed:
        r = sum(K @ r @ K.conj().T for K in kraus_reset(n, p))
    return r


def rule_trace(rho: np.ndarray, n: int, pres0: Sequence[bool], pres1: Sequence[bool]) -> Tuple[np.ndarray, List[int]]:
    """Trace out the destroyed slots, append the created ones in |+>. The register is the present slots of pi1, in slot
    order; the state of the absent slots of pi0 is dropped (they hold nothing). Returns (rho on that register, its slots)."""
    pres0 = np.asarray(pres0, bool)
    pres1 = np.asarray(pres1, bool)
    reg0 = [p for p in range(n) if pres0[p]]
    persist = [p for p in range(n) if pres0[p] and pres1[p]]
    r0 = ptrace(rho, n, reg0)                                   # state of the present slots of pi0
    pos = {p: i for i, p in enumerate(reg0)}
    rp = ptrace(r0, len(reg0), [pos[p] for p in persist]) if persist else np.ones((1, 1))
    reg1 = [p for p in range(n) if pres1[p]]
    created = [p for p in reg1 if p not in persist]
    r = rp
    # slots in the order persist + created, then permute to slot order
    for _ in created:
        r = np.kron(r, rho_of(PLUS))
    order = persist + created
    m = len(order)
    if m > 1:
        t = r.reshape([2] * (2 * m))
        perm = [order.index(p) for p in reg1]
        t = np.transpose(t, perm + [m + i for i in perm])
        r = t.reshape(2 ** m, 2 ** m)
    return r, reg1


# ---- measures ----------------------------------------------------------------------------

def energy(rho: np.ndarray, H: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ H)))


def fidelity_pure(rho: np.ndarray, psi: np.ndarray) -> float:
    return float(np.real(psi.conj() @ rho @ psi))


def evaluate_flip(W: np.ndarray, pres0: Sequence[bool], pres1: Sequence[bool], J: float, kappa: float) -> Dict[str, float]:
    """The three rules on the ground state of H(pi0). Returns energies, overlaps, entropies and the reversibility."""
    n = W.shape[0]
    pres0 = np.asarray(pres0, bool)
    pres1 = np.asarray(pres1, bool)
    E0, psi0 = ground_state_present(W, pres0, J, kappa)
    E1, psi1 = ground_state_present(W, pres1, J, kappa)
    rho0 = rho_of(psi0)
    H1 = H_present(W, pres1, J, kappa)
    destroyed = [p for p in range(n) if pres0[p] and not pres1[p]]
    created = [p for p in range(n) if (not pres0[p]) and pres1[p]]
    persist = [p for p in range(n) if pres0[p] and pres1[p]]
    out = dict(n=n, n_destroyed=len(destroyed), n_created=len(created), n_persist=len(persist), E0=E0, E1=E1)
    # dormant
    rd = rule_dormant(rho0, n, destroyed, created)
    out["dormant_E"] = energy(rd, H1)
    out["dormant_F"] = fidelity_pure(rd, psi1)
    # reset
    rr = rule_reset(rho0, n, destroyed, created)
    out["reset_E"] = energy(rr, H1)
    out["reset_F"] = fidelity_pure(rr, psi1)
    out["reset_loss_bits"] = vn_entropy(rr) - vn_entropy(rho0)
    # reversibility: flip then reverse flip (pi1 -> pi0), rule reset (the dormant rule is the identity)
    destroyed_rev, created_rev = created, destroyed
    rr2 = rule_reset(rr, n, destroyed_rev, created_rev)
    out["reset_return_F"] = fidelity_pure(rr2, psi0)
    out["dormant_return_F"] = fidelity_pure(rule_dormant(rd, n, destroyed_rev, created_rev), psi0)
    # trace rule: the state on the present slots of pi1
    rt, reg1 = rule_trace(rho0, n, pres0, pres1)
    # reference: ground state of H(pi1) on its own register
    if reg1:
        pos = {p: i for i, p in enumerate(reg1)}
        Wp = W[np.ix_(reg1, reg1)]
        Hm = H_present(Wp, [True] * len(reg1), J, kappa)
        w, v = np.linalg.eigh(Hm)
        out["trace_E"] = energy(rt, Hm)
        out["trace_F"] = fidelity_pure(rt, v[:, 0])
    # the marginal on the persisting slots is unchanged by every rule
    if persist:
        base = ptrace(rho0, n, persist)
        out["marginal_dev_reset"] = float(np.abs(ptrace(rr, n, persist) - base).max())
        out["marginal_dev_dormant"] = float(np.abs(ptrace(rd, n, persist) - base).max())
        # entanglement among persisting slots, now and in the new ground state
        if len(persist) >= 2:
            def mean_I(r):
                vals = [mutual_information(r, n, a, b) for i, a in enumerate(persist) for b in persist[i + 1:]]
                return float(np.mean(vals))
            out["I_persist_before"] = mean_I(rho0)
            out["I_persist_new_ground"] = mean_I(rho_of(psi1))
    return out


# ---- unitarity of the dormant rule ------------------------------------------------------

def dormant_unitary(n: int, pres0: Sequence[bool], pres1: Sequence[bool]) -> np.ndarray:
    """The dormant rule as an operator on register x presence, with the two presence configurations (pi0, pi1):
    U = |pi1><pi0| + |pi0><pi1| (the flip, which is its own inverse when the step is reversed) tensor 1 on the register."""
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    return np.kron(X, np.eye(2 ** n))


# ---- checks -----------------------------------------------------------------------------

def _rand_W(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    W = (A + A.T) / 2
    np.fill_diagonal(W, 0.0)
    return W


def test_pair_rule_matches_qwla_on_exact_patches():
    import spectre_exact as se
    from quantumspectremodel import build_substitution_tier
    counts = []
    for k in (2, 3, 4):
        patch = build_substitution_tier(k)
        # model-order keys of qwla are the exact keys shifted by 2; rebuild in exact order from the leaves
        leaves = se.build(k)[k][0]["Delta"]
        tiles = [[tuple(int(x) for x in v) for v in se.outline(T)] for (T, _) in leaves]
        mine = find_pairs_keys(tiles)
        assert mine == sorted(patch.pairs), (k, len(mine), len(patch.pairs))
        counts.append(len(mine))
    assert counts == [6, 82, 804], counts


def test_pair_W_matches_qwla_up_to_roles():
    import spectre_exact as se
    from quantumspectremodel import build_substitution_tier, assign_qubits_to_patch, pair_couplings, _xy
    patch = build_substitution_tier(2)
    assign_qubits_to_patch(patch, max_qubits=None)
    Wq = pair_couplings(patch)
    leaves = se.build(2)[2][0]["Delta"]
    tiles = [[tuple(int(x) for x in v) for v in se.outline(T)] for (T, _) in leaves]
    cents = []
    qubit_of = patch.pair_to_qubit
    order = [p for p, q in sorted(qubit_of.items(), key=lambda t: t[1])]
    for p in order:
        i, j = patch.pairs[p]
        ci = centroid_xy(np.array([_xy(v) for v in tiles[i]]))
        cj = centroid_xy(np.array([_xy(v) for v in tiles[j]]))
        cents.append(ordered_centroids(ci, cj))
    W = pair_W(cents)
    assert np.allclose(np.abs(W), np.abs(Wq), atol=1e-12)
    # equal spectra: the sign pattern differs by conjugations by X_p only
    from pair_hamiltonian import build_hamiltonian
    e1 = np.linalg.eigvalsh(build_hamiltonian(Wq, 1.0, 0.3).toarray())
    e2 = np.linalg.eigvalsh(build_hamiltonian(W, 1.0, 0.3).toarray())
    assert np.allclose(e1, e2, atol=1e-10)


def test_kraus_maps_are_trace_preserving_and_reset_is_not_unitary():
    n = 3
    K = kraus_reset(n, 1)
    S = sum(k.conj().T @ k for k in K)
    assert np.allclose(S, np.eye(2 ** n))
    # reset of one slot of a Bell pair loses one bit
    bell = np.zeros(4); bell[0] = bell[3] = 1 / math.sqrt(2)
    rho = rho_of(bell)
    r = rule_reset(rho, 2, [0], [])
    assert abs(vn_entropy(r) - 1.0) < 1e-9 and abs(vn_entropy(ptrace(r, 2, [1])) - 1.0) < 1e-9   # pure Bell pair -> one bit of entropy
    assert fidelity_pure(r, bell) < 1.0 - 1e-6


def test_dormant_rule_is_unitary_and_reversible():
    n = 4
    U = dormant_unitary(n, [1, 1, 1, 0], [1, 1, 0, 1])
    assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]))
    assert np.allclose(U @ U, np.eye(U.shape[0]))                # flip then reversed flip is the identity
    W = _rand_W(n)
    res = evaluate_flip(W, [1, 1, 1, 0], [1, 1, 0, 1], 1.0, 0.3)
    assert abs(res["dormant_return_F"] - 1.0) < 1e-12
    assert res["marginal_dev_dormant"] < 1e-12 and res["marginal_dev_reset"] < 1e-12


def test_reset_loses_information_when_a_destroyed_pair_is_entangled():
    n = 4
    W = _rand_W(n, 5)
    res = evaluate_flip(W, [1, 1, 1, 0], [1, 1, 0, 1], 0.2, 1.0)    # strong coupling: entangled
    assert res["reset_loss_bits"] > 1e-3, res
    assert res["reset_return_F"] < 1.0 - 1e-6, res


def test_trace_and_reset_agree_on_the_present_register():
    n = 5
    W = _rand_W(n, 7)
    pres0 = np.array([1, 1, 1, 0, 1], bool)
    pres1 = np.array([1, 0, 1, 1, 1], bool)
    E0, psi0 = ground_state_present(W, pres0, 0.5, 0.8)
    rho0 = rho_of(psi0)
    rt, reg1 = rule_trace(rho0, n, pres0, pres1)
    rr = rule_reset(rho0, n, [1], [3])
    assert np.allclose(rt, ptrace(rr, n, reg1), atol=1e-12)


def test_no_change_flip_is_the_identity_and_energy_is_the_ground_energy():
    n = 4
    W = _rand_W(n, 11)
    res = evaluate_flip(W, [1, 1, 1, 1], [1, 1, 1, 1], 0.7, 0.5)
    assert abs(res["dormant_E"] - res["E0"]) < 1e-10 and abs(res["reset_F"] - 1) < 1e-10


def test_isolated_pair_ground_state_is_plus():
    W = np.zeros((1, 1))
    e, psi = ground_state_present(W, [True], 1.0, 0.5)
    assert abs(e + 1.0) < 1e-12 and abs(abs(psi @ PLUS) - 1.0) < 1e-12


def test_created_pair_in_plus_contributes_minus_J_and_no_coupling_energy():
    # slot 2 is created in |+>: its <X> = 1 and <Z> = 0, so it adds -J and nothing else to the energy of the old state
    n = 3
    W = _rand_W(n, 13)
    J, kappa = 0.6, 0.9
    pres0 = [1, 1, 0]
    pres1 = [1, 1, 1]
    res = evaluate_flip(W, pres0, pres1, J, kappa)
    assert abs(res["dormant_E"] - (res["E0"] - J)) < 1e-10, res


def run_tests(verbose: bool = True) -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        if verbose:
            print("ok", t.__name__)
    return len(tests)


if __name__ == "__main__":
    n = run_tests()
    print(f"{n} checks passed")
