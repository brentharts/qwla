"""
Stage 4 of ROADMAP_QUANTUM_GRAVITY.md: geometry from entanglement.

For the ground state of H = -J sum X + kappa sum W_pq Z_p Z_q on the n central pairs of a patch
(tau = J / (kappa w), see `pair_hamiltonian`), this module measures

  1. the mutual information I(p:q) = S_p + S_q - S_pq between pair qubits and how it ranks against
     the Euclidean distance r_pq between the pairs, the tile-contact-graph distance d_pq, and the
     Hamiltonian's own coupling |W_pq|;
  2. the entanglement entropy S(R) of compact regions R against their boundary B(R) (Delaunay links
     of the pair positions that cross the cut) and their volume |R|, by least squares
     S = alpha B + beta |R| + gamma.

Two controls keep the result honest. (a) The couplings W are built from the tile positions, so a
correlation of I with r is partly inherited from the Hamiltonian; the "shuffled" control reuses W
with the positions permuted, which destroys any real link between I and the geometry. (b) With
n <= 20 qubits the regions are small and B and |R| are correlated; the fit reports both and says
which is better, it does not claim a thermodynamic scaling law.

Run `python3 entanglement_geometry.py --tests-only`, or without arguments for the measurement.
"""
import sys
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import spearmanr

from pair_hamiltonian import (
    coupling_scale,
    entanglement_entropy,
    ground_state,
    pairwise_concurrence,
)
from quantumspectremodel import (
    assign_qubits_to_patch,
    build_substitution_tier,
    pair_couplings,
    pair_qubit_tiles,
    tile_centroid,
)


# ---- geometry of the chosen pairs ---------------------------------------------------------

def build_cluster(generation: int, n_pairs: int):
    """Patch, W, midpoints (n, 2) and tile-contact distances (n, n) of the n central pairs."""
    patch = build_substitution_tier(generation)
    assign_qubits_to_patch(patch, max_qubits=n_pairs)
    W = pair_couplings(patch)
    tiles = pair_qubit_tiles(patch)
    mids = np.array([0.5 * (tile_centroid(A) + tile_centroid(B)) for A, B in (tiles[q] for q in range(n_pairs))])
    # tile contact graph: two tiles touch when they share an exact vertex
    at: Dict[tuple, List[int]] = {}
    for t_idx, tile in enumerate(patch.tiles):
        for key in tile.data["exact_vertices"]:
            at.setdefault(key, []).append(t_idx)
    nbrs: List[set] = [set() for _ in patch.tiles]
    for lst in at.values():
        for a in lst:
            nbrs[a].update(b for b in lst if b != a)
    index_of = {id(t): i for i, t in enumerate(patch.tiles)}
    pair_tiles = [[index_of[id(t)] for t in tiles[q]] for q in range(n_pairs)]
    D = np.zeros((n_pairs, n_pairs))
    for p in range(n_pairs):
        dist = {t: 0 for t in pair_tiles[p]}
        queue = deque(pair_tiles[p])
        while queue:
            u = queue.popleft()
            for v in nbrs[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for q in range(n_pairs):
            D[p, q] = min(dist.get(t, 10 ** 6) for t in pair_tiles[q])
    return patch, W, mids, D


def delaunay_links(mids: np.ndarray) -> np.ndarray:
    """Boolean adjacency of the Delaunay triangulation of the pair positions (parameter free)."""
    n = len(mids)
    A = np.zeros((n, n), bool)
    for simplex in Delaunay(mids).simplices:
        for i in simplex:
            for j in simplex:
                if i != j:
                    A[i, j] = True
    return A


# ---- mutual information -------------------------------------------------------------------

def mutual_information(psi: np.ndarray, n: int) -> np.ndarray:
    S1 = np.array([entanglement_entropy(psi, n, [p]) for p in range(n)])
    I = np.zeros((n, n))
    for p in range(n):
        for q in range(p + 1, n):
            I[p, q] = I[q, p] = S1[p] + S1[q] - entanglement_entropy(psi, n, [p, q])
    return I


def rank_correlations(I: np.ndarray, mids: np.ndarray, D: np.ndarray, W: np.ndarray) -> Dict[str, float]:
    """Spearman rho of I(p:q) against r, d and |W| over all pairs of pairs (rho < 0: nearer = more entangled)."""
    iu = np.triu_indices(len(mids), 1)
    r = np.linalg.norm(mids[:, None] - mids[None], axis=-1)[iu]
    return {
        "rho_I_r": float(spearmanr(I[iu], r)[0]),
        "rho_I_graph": float(spearmanr(I[iu], D[iu])[0]),
        "rho_I_absW": float(spearmanr(I[iu], np.abs(W[iu]))[0]),
        "rho_W_r": float(spearmanr(np.abs(W[iu]), r)[0]),
    }


def _rank(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of x and y with the (rank) effect of z removed."""
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    cxy, cxz, cyz = np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1], np.corrcoef(ry, rz)[0, 1]
    return float((cxy - cxz * cyz) / np.sqrt((1 - cxz ** 2) * (1 - cyz ** 2)))


def geometry_vs_null(I: np.ndarray, mids: np.ndarray, W: np.ndarray, draws: int = 300, seed: int = 0) -> Dict[str, float]:
    """
    rho(I, r) and the partial rho(I, r | |W|), each against `draws` random relabellings of the
    positions (W and I kept). z = (value - null mean) / null sd. The partial one asks whether the
    entanglement knows the distance beyond what the Hamiltonian's couplings already say.
    """
    n = len(mids)
    iu = np.triu_indices(n, 1)
    absW, Iv = np.abs(W[iu]), I[iu]

    def stats(m):
        r = np.linalg.norm(m[:, None] - m[None], axis=-1)[iu]
        return spearmanr(Iv, r)[0], partial_spearman(Iv, r, absW)

    rho, part = stats(mids)
    rng = np.random.default_rng(seed)
    null = np.array([stats(mids[rng.permutation(n)]) for _ in range(draws)])
    return {
        "rho_I_r": float(rho), "null_rho_mean": float(null[:, 0].mean()), "null_rho_sd": float(null[:, 0].std()),
        "z_rho": float((rho - null[:, 0].mean()) / null[:, 0].std()),
        "partial_rho_I_r_given_W": float(part), "null_partial_mean": float(null[:, 1].mean()),
        "null_partial_sd": float(null[:, 1].std()),
        "z_partial": float((part - null[:, 1].mean()) / null[:, 1].std()),
    }


# ---- area versus volume -------------------------------------------------------------------

def compact_regions(mids: np.ndarray, max_size: int) -> List[Tuple[int, ...]]:
    """Distinct regions made of the m pairs nearest to a seed pair (m = 1 .. max_size)."""
    n = len(mids)
    seen, out = set(), []
    for seed in range(n):
        order = np.argsort(np.linalg.norm(mids - mids[seed], axis=1), kind="stable")
        for m in range(1, max_size + 1):
            reg = tuple(sorted(int(x) for x in order[:m]))
            if reg not in seen:
                seen.add(reg)
                out.append(reg)
    return out


def random_regions(n: int, sizes, count: int, seed: int = 0) -> List[Tuple[int, ...]]:
    """Random subsets (not compact), which decouple the boundary from the volume."""
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    for _ in range(count):
        m = int(rng.choice(sizes))
        reg = tuple(sorted(int(x) for x in rng.choice(n, m, replace=False)))
        if reg not in seen:
            seen.add(reg)
            out.append(reg)
    return out


def boundary_size(region: Tuple[int, ...], links: np.ndarray) -> int:
    inside = np.zeros(links.shape[0], bool)
    inside[list(region)] = True
    return int(np.sum(links[np.ix_(inside, ~inside)]))


def fit_area_volume(psi: np.ndarray, n: int, mids: np.ndarray, links: np.ndarray, regions=None) -> Dict[str, float]:
    if regions is None:
        regions = compact_regions(mids, n // 2)
    S = np.array([entanglement_entropy(psi, n, list(r)) for r in regions])
    B = np.array([boundary_size(r, links) for r in regions], float)
    V = np.array([len(r) for r in regions], float)

    def lsq(cols):
        X = np.column_stack(cols + [np.ones_like(S)])
        coef, *_ = np.linalg.lstsq(X, S, rcond=None)
        res = S - X @ coef
        return coef, float(np.sqrt(np.mean(res ** 2)))

    cb, rb = lsq([B])
    cv, rv = lsq([V])
    cbv, rbv = lsq([B, V])
    return {
        "n_regions": len(regions),
        "rms_area_only": rb, "alpha_area_only": float(cb[0]),
        "rms_volume_only": rv, "beta_volume_only": float(cv[0]),
        "rms_both": rbv, "alpha": float(cbv[0]), "beta": float(cbv[1]), "gamma": float(cbv[2]),
        "corr_B_V": float(np.corrcoef(B, V)[0, 1]),
        # standardised coefficients: how much S moves per sd of B and of V
        "std_alpha": float(cbv[0] * B.std()), "std_beta": float(cbv[1] * V.std()),
    }


# ---- the measurement ----------------------------------------------------------------------

def ground_state_at(W: np.ndarray, tau: float) -> np.ndarray:
    return ground_state(W, tau * coupling_scale(W), 1.0)[1]


def peak_tau(W: np.ndarray, taus: np.ndarray) -> float:
    n = W.shape[0]
    best, best_c = taus[0], -1.0
    for tau in taus:
        c = float(pairwise_concurrence(ground_state_at(W, tau), n).max())
        if c > best_c:
            best, best_c = float(tau), c
    return best


def measure(generation: int, n: int, taus=None, seed: int = 0) -> List[Dict[str, float]]:
    patch, W, mids, D = build_cluster(generation, n)
    links = delaunay_links(mids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)                      # the shuffled control: same W, positions permuted
    if taus is None:
        taus = [0.1, peak_tau(W, np.geomspace(0.15, 3.0, 12)), 3.0]
    rows = []
    for tau in taus:
        psi = ground_state_at(W, tau)
        I = mutual_information(psi, n)
        real = rank_correlations(I, mids, D, W)
        ctrl = rank_correlations(I, mids[perm], D[np.ix_(perm, perm)], W)
        fit = fit_area_volume(psi, n, mids, links)
        rnd = fit_area_volume(psi, n, mids, links, random_regions(n, range(2, n // 2 + 1), 250))
        nul = geometry_vs_null(I, mids, W)
        iu = np.triu_indices(n, 1)
        rows.append({"n": n, "tau": float(tau), "I_mean": float(I[iu].mean()), "I_max": float(I[iu].max()),
                     **real, "ctrl_rho_I_r": ctrl["rho_I_r"], "ctrl_rho_I_graph": ctrl["rho_I_graph"], **fit,
                     **{f"null_{k}": v for k, v in nul.items()}, **{f"rand_{k}": v for k, v in rnd.items()}})
    return rows


# ---- checks ------------------------------------------------------------------------------

def test_mutual_information_of_bell_pair_and_product():
    bell = np.zeros(8)
    bell[0b000] = bell[0b110] = 1 / np.sqrt(2)          # qubits 0,1 Bell, qubit 2 in |0>
    I = mutual_information(bell, 3)
    assert abs(I[0, 1] - 2.0) < 1e-9 and abs(I[0, 2]) < 1e-9 and abs(I[1, 2]) < 1e-9


def test_ghz_has_uniform_mutual_information_of_one_bit():
    n = 6
    ghz = np.zeros(2 ** n)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    I = mutual_information(ghz, n)
    iu = np.triu_indices(n, 1)
    assert np.allclose(I[iu], 1.0), "the cat phase carries no distance information: I is flat"


def test_fit_recovers_a_synthetic_area_law():
    mids = np.random.default_rng(1).random((12, 2))
    links = delaunay_links(mids)
    regions = compact_regions(mids, 6)
    B = np.array([boundary_size(r, links) for r in regions], float)
    X = np.column_stack([B, np.ones_like(B)])
    S = 0.7 * B + 0.1
    coef, *_ = np.linalg.lstsq(X, S, rcond=None)
    assert np.allclose(coef, [0.7, 0.1])
    assert len(set(regions)) == len(regions) and B.min() >= 0


def test_cluster_geometry_is_consistent():
    patch, W, mids, D = build_cluster(3, 8)
    assert W.shape == (8, 8) and mids.shape == (8, 2)
    assert np.allclose(D, D.T) and np.all(np.diag(D) == 0) and D.max() < 100, "cluster must be connected"
    assert delaunay_links(mids).sum() >= 2 * 8


def test_partial_spearman_removes_a_common_driver():
    rng = np.random.default_rng(3)
    z = rng.random(400)
    x, y = z + 0.05 * rng.random(400), z + 0.05 * rng.random(400)
    assert spearmanr(x, y)[0] > 0.9
    assert abs(partial_spearman(x, y, z)) < 0.2


def test_random_regions_decouple_boundary_from_volume():
    mids = np.random.default_rng(2).random((14, 2))
    links = delaunay_links(mids)
    rr = random_regions(14, range(2, 8), 200)
    comp = compact_regions(mids, 7)
    c = lambda regs: np.corrcoef([boundary_size(r, links) for r in regs], [len(r) for r in regs])[0, 1]
    assert c(rr) < c(comp) or c(rr) < 0.9 and len(rr) > 100


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
    sizes = [int(a) for a in sys.argv[1:]] or [12, 14, 16]
    for n in sizes:
        for r in measure(4, n):
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
