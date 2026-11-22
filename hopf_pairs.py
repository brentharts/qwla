"""
hopf_pairs.py - two things from the HopfFibration fork, tested on the pair qubits.

PART 1. The quaternionic Hopf map S^7 -> S^4 on a pure two-qubit state.
   Write psi = (a, b, c, d) as the quaternion pair q1 = a + b j, q2 = c + d j. Then
       h(q1, q2) = (2 q1 conj(q2), |q1|^2 - |q2|^2)  in R^5, unit length,
   and 2 q1 conj(q2) = 2(a conj c + b conj d) + 2(b c - a d) j.   So (EXACT, tested):
       (x1, x2, x5) is the Bloch vector of qubit 1,      (x3, x4) has length C = 2|ad - bc|, the concurrence,
       |r_1|^2 + C^2 = 1                                  (the point lies on S^4).
   The fibre of the map is right multiplication (q1 u, q2 u), i.e. a local SU(2) on qubit 2: it does not move
   the base point. So S^4 is "qubit 1 plus how entangled it is", and the entanglement is the S^2-fibre radius
   over the Bloch ball. (A plain search for "C = sqrt(1 - x5^2)" fails; the concurrence is the length of the
   j-part of the 4-vector, not a function of x5 alone.)

   For a MIXED reduced state of two pair qubits the map is not defined, but the same bookkeeping gives a number:
       D = (1 - |r_p|^2) - C(rho_pq)^2 >= 0      ("leak": the part of qubit p's entanglement that goes elsewhere).
   D = 0 exactly when the pair (p, q) is a pure two-qubit state, i.e. when the S^7 -> S^4 picture applies.

PART 2. Tier-to-tier entanglement without hand-chosen bridges.
   `TriSpectreNetwork.entangle_across_computers` joins the tiers with CX/CZ gates chosen for the toy. Here the
   tiers are joined by the same dipole law as everything else: the pairs of all tiers are put in one plane and
   W_pq = 1/r_AA' - 1/r_AB' - 1/r_BA' + 1/r_BB' is evaluated between pairs of different tiers. The tiers do NOT
   nest at one scale in the exact patches (checked: a generation-k patch is not a translated, rotated subset of
   generation k+1), so the common frame is a MODEL CHOICE, with two options and a random-offset control.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

import pair_hamiltonian as ph
from quantumspectremodel import _centroid, _dipole_matrix, pair_qubit_tiles, tile_centroid, tile_polygon_at_b

# ======================================================================= PART 1: S^7 -> S^4

def qmul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w, x, y, z) with i j = k."""
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(p: np.ndarray) -> np.ndarray:
    return np.asarray(p) * np.array([1.0, -1.0, -1.0, -1.0])


def complex_pair_to_quaternion(a: complex, b: complex) -> np.ndarray:
    """a + b j with a, b complex (i = x axis, j = y axis, so b j has components Re b on j and Im b on k)."""
    return np.array([a.real, a.imag, b.real, b.imag])


def hopf_map_quaternionic(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """The map exactly as in HopfFibration/hopf_quaternionic.py: (2 q1 conj(q2), |q1|^2 - |q2|^2)."""
    v = 2.0 * qmul(q1, qconj(q2))
    return np.append(v, float(q1 @ q1 - q2 @ q2))


def hopf_s4(psi: np.ndarray) -> np.ndarray:
    """S^4 point of a normalised 2-qubit state (a, b, c, d) in the basis |00>,|01>,|10>,|11>."""
    a, b, c, d = np.asarray(psi, complex)
    return hopf_map_quaternionic(complex_pair_to_quaternion(a, b), complex_pair_to_quaternion(c, d))


def bloch_vector(rho1: np.ndarray) -> np.ndarray:
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1.0, -1.0]).astype(complex)
    return np.array([np.real(np.trace(rho1 @ s)) for s in (sx, sy, sz)])


def s4_decomposition(psi: np.ndarray) -> Dict[str, float]:
    """The S^4 point split as in the docstring: |r_1|, the j-part length (= concurrence), the norm check."""
    x = hopf_s4(psi)
    return {"norm": float(np.linalg.norm(x)), "bloch_len": float(np.hypot(np.hypot(x[0], x[1]), x[4])),
            "j_part": float(np.hypot(x[2], x[3])), "concurrence": 2.0 * abs(psi[0] * psi[3] - psi[1] * psi[2])}


def right_fibre_action(psi: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(q1, q2) -> (q1 u, q2 u) for a unit quaternion u, returned as a two-qubit state vector."""
    a, b, c, d = np.asarray(psi, complex)
    q1, q2 = qmul(complex_pair_to_quaternion(a, b), u), qmul(complex_pair_to_quaternion(c, d), u)
    return np.array([q1[0] + 1j * q1[1], q1[2] + 1j * q1[3], q2[0] + 1j * q2[1], q2[2] + 1j * q2[3]])


def leak(rho_pq: np.ndarray) -> Dict[str, float]:
    """D = (1 - |r_p|^2) - C^2 for a two-qubit density matrix (qubit p first). D >= 0; D = 0 iff pure."""
    r = rho_pq.reshape(2, 2, 2, 2)
    rho_p = np.einsum("ajbj->ab", r)
    C = ph.concurrence(rho_pq)
    r2 = float(np.sum(bloch_vector(rho_p) ** 2))
    return {"bloch_len2": r2, "C": C, "D": 1.0 - r2 - C * C}


def pair_leaks(psi: np.ndarray, n: int, links: np.ndarray | None = None) -> List[Dict[str, float]]:
    """Leak of every pair of pair qubits (p < q); with `links` only the nearest-neighbour ones."""
    out = []
    for p in range(n):
        for q in range(p + 1, n):
            if links is not None and not links[p, q]:
                continue
            d = leak(ph.reduced_two_qubit(psi, n, p, q))
            d.update(p=p, q=q)
            out.append(d)
    return out


# ======================================================================= PART 2: tiers without bridges

LAMBDA = math.sqrt(4.0 + math.sqrt(15.0))      # Spectre inflation factor per generation, lambda^2 = 4 + sqrt 15


def tier_centroids(generation: int, n_pairs: int, b: float) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """(A, B) tile centroids of the n_pairs central pairs of a tier at Tile(1, b)."""
    patch = ph._central_patch(generation, n_pairs)
    cents = {}
    for p, q in patch.pair_to_qubit.items():
        i, j = patch.pairs[p]
        cents[q] = (_centroid(tile_polygon_at_b(patch.tiles[i], b)), _centroid(tile_polygon_at_b(patch.tiles[j], b)))
    return cents


def overlay_couplings(generations: Sequence[int], n_per: int, b: float, mode: str = "centre",
                      rng: np.random.Generator | None = None, spread: float = 0.0) -> Tuple[np.ndarray, List[range]]:
    """
    Dipole matrix W for all pairs of all tiers in one plane. Each tier's pair cluster is translated so that its
    mean pair position is the origin ("centre"), optionally shrunk by lambda^-(g - g0) ("inflated": a generation-g
    supertile is lambda^(g-g0) larger, so its pairs are packed that much more closely when seen from the top
    tier), and optionally shifted by a random vector of size `spread` ("control").
    Returns W and the qubit range of each tier. Tier qubits are numbered tier by tier.
    """
    allc: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    ranges: List[range] = []
    g0 = min(generations)
    for g in generations:
        cents = tier_centroids(g, n_per, b)
        mid = np.mean([0.5 * (A + B) for A, B in cents.values()], axis=0)
        s = LAMBDA ** (-(g - g0)) if mode == "inflated" else 1.0
        shift = rng.normal(size=2) * spread if rng is not None else np.zeros(2)
        start = len(allc)
        for k, (A, B) in sorted(cents.items()):
            allc[start + k] = (s * (A - mid) + shift, s * (B - mid) + shift)
        ranges.append(range(start, start + len(cents)))
    return _dipole_matrix(allc), ranges


def block_diagonal_part(W: np.ndarray, ranges: Sequence[range]) -> np.ndarray:
    out = np.zeros_like(W)
    for r in ranges:
        idx = np.array(list(r))
        out[np.ix_(idx, idx)] = W[np.ix_(idx, idx)]
    return out


def entropy_of(psi: np.ndarray, n: int, region: Sequence[int]) -> float:
    """von Neumann entropy in nats of a region of a pure state."""
    rest = [q for q in range(n) if q not in region]
    t = np.transpose(psi.reshape([2] * n), list(region) + rest).reshape(2 ** len(region), -1)
    s = np.linalg.svd(t, compute_uv=False) ** 2
    s = s[s > 1e-14]
    return float(-np.sum(s * np.log(s)))


def tier_mutual_information(psi: np.ndarray, ranges: Sequence[range]) -> np.ndarray:
    """I(T_i : T_j) = S_i + S_j - S_ij between tiers (pure global state). Symmetric, zero diagonal."""
    n = sum(len(r) for r in ranges)
    S = [entropy_of(psi, n, list(r)) for r in ranges]
    m = len(ranges)
    I = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            Sij = entropy_of(psi, n, list(ranges[i]) + list(ranges[j]))
            I[i, j] = I[j, i] = S[i] + S[j] - Sij
    return I


def ground_state(W: np.ndarray, J: float, kappa: float) -> Tuple[float, np.ndarray]:
    """Ground state; dense for n <= 3 (the parity-sector solver needs n >= 3 pairs per sector of dimension > 2)."""
    if W.shape[0] <= 3:
        w, v = np.linalg.eigh(ph.build_hamiltonian(W, J, kappa).toarray())
        return float(w[0]), v[:, 0].astype(complex)
    return ph.ground_state(W, J, kappa)


def ground_state_tier_mi(W: np.ndarray, ranges: Sequence[range], J: float, kappa: float) -> Tuple[np.ndarray, np.ndarray]:
    _, psi = ground_state(W, J, kappa)
    return tier_mutual_information(psi, ranges), psi


def symmetry_broken_ground_state(W: np.ndarray, J: float, kappa: float) -> Tuple[float, np.ndarray]:
    """(|E0, P=+1> + |E0', P=-1>)/sqrt2 from the two lowest levels, resolved by the parity P = prod X.
    Where the doublet is split by less than ~1e-6 this is the 'one classical pattern' state of the cat pair."""
    w, v = np.linalg.eigh(ph.build_hamiltonian(W, J, kappa).toarray())
    a, b = v[:, 0], v[:, 1]
    M = np.array([[np.vdot(x, y[::-1]).real for y in (a, b)] for x in (a, b)])
    e, u = np.linalg.eigh(M)
    plus, minus = u[0, 1] * a + u[1, 1] * b, u[0, 0] * a + u[1, 0] * b
    return float(w[1] - w[0]), (plus + minus) / math.sqrt(2.0)


def quench_tier_mi(W: np.ndarray, ranges: Sequence[range], J: float, kappa: float, times: Sequence[float]) -> np.ndarray:
    """Exact evolution of |00..0> under H (no gates, no bridges); tier mutual information at each time."""
    n = W.shape[0]
    H = ph.build_hamiltonian(W, J, kappa).toarray()
    w, v = np.linalg.eigh(H)
    psi0 = np.zeros(2 ** n, complex)
    psi0[0] = 1.0
    c = v.conj().T @ psi0
    out = []
    for t in times:
        psi = v @ (np.exp(-1j * w * t) * c)
        out.append(tier_mutual_information(psi, ranges))
    return np.array(out)


def state_to_vector(state) -> np.ndarray:
    """Amplitudes of a `state.State` as a vector with qubit 0 as the most significant bit."""
    n = state.n_qubits
    psi = np.zeros(2 ** n, complex)
    for bits, amp in state.state.to_list():
        psi[int("".join(str(int(x)) for x in bits), 2)] += amp
    return psi


# ======================================================================= tests

def test_s7_s4_map_is_the_reduced_state_of_qubit_one_and_the_concurrence():
    rng = np.random.default_rng(0)
    for _ in range(50):
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi /= np.linalg.norm(psi)
        d = s4_decomposition(psi)
        assert abs(d["norm"] - 1.0) < 1e-12
        assert abs(d["j_part"] - d["concurrence"]) < 1e-12
        assert abs(d["bloch_len"] ** 2 + d["j_part"] ** 2 - 1.0) < 1e-12
        rho = psi.reshape(2, 2)
        rho1 = rho @ rho.conj().T            # qubit 1 = row index
        x = hopf_s4(psi)
        r = bloch_vector(rho1)
        assert np.allclose([x[0], -x[1], x[4]], [r[0], r[1], r[2]], atol=1e-12) or \
            np.allclose([x[0], x[1], x[4]], [r[0], r[1], r[2]], atol=1e-12)


def test_the_fibre_is_right_multiplication_and_is_a_local_su2_on_qubit_two():
    rng = np.random.default_rng(1)
    for _ in range(20):
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        psi /= np.linalg.norm(psi)
        u = rng.normal(size=4)
        u /= np.linalg.norm(u)
        moved = right_fibre_action(psi, u)
        assert np.allclose(hopf_s4(moved), hopf_s4(psi), atol=1e-12)            # same base point
        assert abs(np.linalg.norm(moved) - 1.0) < 1e-12
        # it is a product with a unitary on the second qubit: the reduced state of qubit 1 and C do not change
        r0, r1 = psi.reshape(2, 2), moved.reshape(2, 2)
        assert np.allclose(r0 @ r0.conj().T, r1 @ r1.conj().T, atol=1e-12)


def test_leak_is_zero_for_pure_pairs_and_positive_when_the_pair_is_entangled_with_a_third_qubit():
    rng = np.random.default_rng(2)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    assert abs(leak(np.outer(psi, psi.conj()))["D"]) < 1e-9
    ghz = np.zeros(8, complex)
    ghz[0] = ghz[7] = 1 / math.sqrt(2)
    assert leak(ph.reduced_two_qubit(ghz, 3, 0, 1))["D"] > 0.9          # all of the entanglement is shared 3-way


def test_overlay_has_the_cross_tier_couplings_and_the_diagonal_blocks_are_the_tier_couplings():
    # same patch geometry => same W inside a tier up to the common translation/scale (centre mode: exact)
    W, ranges = overlay_couplings((2, 3, 4), 4, b=1.0, mode="centre")
    for g, r in zip((2, 3, 4), ranges):
        idx = np.array(list(r))
        assert np.allclose(W[np.ix_(idx, idx)], ph.central_couplings(g, 4, 1.0), atol=1e-9)
    assert np.max(np.abs(W - block_diagonal_part(W, ranges))) > 0.0
    assert np.allclose(W, W.T)


def test_without_cross_tier_couplings_the_tiers_are_a_product_state_and_with_them_they_are_not():
    fp = ph.flow_parameters(0.05)
    W, ranges = overlay_couplings((2, 3, 4), 3, fp.b, mode="centre")
    I0, _ = ground_state_tier_mi(block_diagonal_part(W, ranges), ranges, fp.J, fp.kappa)
    I1, _ = ground_state_tier_mi(W, ranges, fp.J, fp.kappa)
    assert np.max(I0) < 1e-8
    assert np.max(I1) > 1e-3


def test_cat_ground_state_has_ln2_tier_mutual_information_that_is_parity_not_tier_entanglement():
    fp = ph.flow_parameters(0.05)
    W, R = overlay_couplings((2, 3, 4), 4, fp.b, mode="inflated")
    I_cat, _ = ground_state_tier_mi(W, R, fp.J, fp.kappa)
    gap, psi = symmetry_broken_ground_state(W, fp.J, fp.kappa)
    I_broken = tier_mutual_information(psi, R)
    assert gap < 1e-6
    assert abs(I_cat[0, 1] - math.log(2)) < 0.01 and abs(I_cat[0, 2] - math.log(2)) < 0.01   # GHZ-like parity correlation
    assert I_broken.max() < 0.01                                                              # no tier entanglement once it is removed


def strongest_partner_leaks(W: np.ndarray, psi: np.ndarray) -> List[float]:
    """D of the pair (p, strongest-coupled partner of p), each pair counted once."""
    n = W.shape[0]
    pairs = set()
    for p in range(n):
        q = max((x for x in range(n) if x != p), key=lambda x: abs(W[p, x]))
        pairs.add((min(p, q), max(p, q)))
    return [leak(ph.reduced_two_qubit(psi, n, p, q))["D"] for p, q in sorted(pairs)]


def test_hopf_pair_picture_is_exact_for_two_pairs_and_improves_with_heat_for_many():
    for tau in (0.0, 0.05, 0.2):
        fp = ph.flow_parameters(tau)
        W2 = ph.central_couplings(3, 2, fp.b) * fp.kappa
        _, psi = ground_state(W2, fp.J, 1.0)
        assert abs(leak(ph.reduced_two_qubit(psi, 2, 0, 1))["D"]) < 1e-6
    Dm = []
    for tau in (0.05, 0.2):
        fp = ph.flow_parameters(tau)
        W = ph.central_couplings(3, 8, fp.b)
        _, psi = ph.ground_state(W, fp.J, fp.kappa)
        Dm.append(float(np.mean(strongest_partner_leaks(W, psi))))
    assert Dm[0] > 0.3 and Dm[1] < 0.15, Dm


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
    raise SystemExit(run_tests())
