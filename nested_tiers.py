"""
nested_tiers.py - the substitution tiers are nested EXACTLY, so "tier-to-tier" is region-to-region in one tiling.

Found while testing the ray fork's idea of a ring tree (a recursive object whose parent-child links carry the
structure): spectre_exact builds the level-k patch of species S as the union, over the eight slots idx, of the
level-(k-1) patches of the child species SUPER_RULES[S][idx], each moved by the slot transform Ts[idx]. For the
single-tile species (Delta) slot 1 is again Delta, so
    patch(k) = Ts_k[1] . patch(k-1)   as sets of exact leaf tiles, for k = 2, 3, 4  (9 < 71 < 559 < 4401 tiles),
and every pair of patch(k) is still a pair of patch(k+1) (6 < 82 < 804 pairs). The three "tiers" of
`TriSpectreNetwork` are therefore not three independent systems: they are nested regions of one tiling.

(An earlier test in hopf_pairs.py / the roadmap said the patches "do not nest at one scale". That was wrong: it
 looked for a translation and a rotation by a multiple of 30 degrees only, and missed the slot transform.)

What this module does with that:
  * `embedding(k)`: index of each tile of patch(k) in patch(k+1);
  * `nested_cluster(...)`: a cluster of pair qubits made of the pairs of tier 2, then the next-nearest pairs that
    are new in tier 3, then those new in tier 4 - geometry of ONE patch, so the dipole matrix W is the real one;
  * `shell_entanglement(...)`: entropies and mutual informations between the shells, with a shuffled-label control.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

import pair_hamiltonian as ph
import spectre_exact as se
from quantumspectremodel import _centroid, _dipole_matrix, build_substitution_tier, tile_polygon_at_b
import hopf_pairs as hp

_B = None


def _build():
    global _B
    if _B is None:
        _B = se.build(4)
    return _B


def _key(T, kind):
    if isinstance(T, tuple):
        return tuple(tuple(np.asarray(a).ravel().tolist()) for a in T), kind
    return tuple(np.asarray(T).ravel().tolist()), kind


def embedding(k: int) -> np.ndarray:
    """embedding(k)[i] = index in patch(k+1) of tile i of patch(k) (single-tile species, exact)."""
    B = _build()
    Ts = B[k + 1][2]
    big = {_key(T, kd): i for i, (T, kd) in enumerate(B[k + 1][0]["Delta"])}
    return np.array([big[_key(se.mul(Ts[1], T), kd)] for T, kd in B[k][0]["Delta"]])


def embedding_to_top(k: int, top: int) -> np.ndarray:
    m = np.arange(len(_build()[k][0]["Delta"]))
    for j in range(k, top):
        m = embedding(j)[m]
    return m


def nested_pairs(top: int = 4) -> Dict[int, set]:
    """For each tier k <= top: its pairs as frozensets of tile indices of patch(top)."""
    out = {}
    for k in range(2, top + 1):
        p = build_substitution_tier(k)
        m = embedding_to_top(k, top)
        out[k] = {frozenset(int(m[i]) for i in pr) for pr in p.pairs}
    return out


def pair_centroids(patch, pair, b: float):
    i, j = pair
    return _centroid(tile_polygon_at_b(patch.tiles[i], b)), _centroid(tile_polygon_at_b(patch.tiles[j], b))


def nested_cluster(sizes: Sequence[int], b: float, top: int = 4) -> Tuple[np.ndarray, List[range], List[Tuple[int, int]]]:
    """
    Pair-qubit cluster from the nesting. sizes[0] = pairs of tier 2 (<= 6, all of them if 6), sizes[1] = pairs
    new in tier 3 nearest to tier 2's centre, sizes[2] = pairs new in tier 4 nearest to that centre, ...
    Returns W (real dipole matrix of the one patch(top)), the qubit range of each shell, and the tile pairs.
    """
    patch = build_substitution_tier(top)
    nest = nested_pairs(top)
    tiers = sorted(nest)
    pairs_by_set = {frozenset(p): tuple(p) for p in patch.pairs}
    centre = np.mean([np.mean(pair_centroids(patch, pairs_by_set[s], b), axis=0) for s in nest[tiers[0]]], axis=0)
    shells: List[List[frozenset]] = [sorted(nest[tiers[0]], key=lambda s: 0)[: sizes[0]]]
    seen = set(shells[0])
    for lvl, n in zip(tiers[1:], sizes[1:]):
        cand = [s for s in nest[lvl] if s not in seen]
        cand.sort(key=lambda s: np.linalg.norm(np.mean(pair_centroids(patch, pairs_by_set[s], b), axis=0) - centre))
        shells.append(cand[:n])
        seen |= set(cand[:n])
    cents: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    ranges: List[range] = []
    order: List[Tuple[int, int]] = []
    q = 0
    for sh in shells:
        start = q
        for s in sh:
            cents[q] = pair_centroids(patch, pairs_by_set[s], b)
            order.append(pairs_by_set[s])
            q += 1
        ranges.append(range(start, q))
    return _dipole_matrix(cents), ranges, order


def shell_entanglement(W: np.ndarray, ranges: Sequence[range], J: float, kappa: float, broken: bool = True) -> Dict[str, object]:
    """Entropies of each shell and nested union, and shell-shell mutual information, in the ground state
    (the symmetry-broken one by default, to remove the ln2 of the parity cat)."""
    n = W.shape[0]
    w, v = ph.sector_levels(W, J, kappa, +1, k=1)
    psi = ph.embed_sector(v[:, 0], n, +1)
    if broken:
        w2, v2 = ph.sector_levels(W, J, kappa, -1, k=1)
        psi = (psi + ph.embed_sector(v2[:, 0], n, -1)) / np.sqrt(2.0)
        gap = float(w2[0] - w[0])
    else:
        gap = float("nan")
    S = [hp.entropy_of(psi, n, list(r)) for r in ranges]
    I = hp.tier_mutual_information(psi, ranges)
    nested = []
    acc: List[int] = []
    for r in ranges[:-1]:
        acc += list(r)
        nested.append(hp.entropy_of(psi, n, list(acc)))
    return {"gap": gap, "S_shell": S, "I": I, "S_nested": nested}


def shuffled_control(W: np.ndarray, ranges: Sequence[range], J: float, kappa: float, seed: int, broken: bool = True):
    """Same couplings, same shell sizes, but pairs assigned to shells at random."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(W.shape[0])
    sizes = [len(r) for r in ranges]
    new, q = [], 0
    for s in sizes:
        new.append(range(q, q + s))
        q += s
    Wp = W[np.ix_(perm, perm)]
    return shell_entanglement(Wp, new, J, kappa, broken)


# ======================================================================= tests

def test_tiers_nest_exactly_through_slot_one_and_pairs_nest_with_them():
    for k in (1, 2, 3):
        m = embedding(k)
        assert len(m) == [9, 71, 559][k - 1] and len(set(m.tolist())) == len(m)
    nest = nested_pairs(4)
    assert [len(nest[k]) for k in (2, 3, 4)] == [6, 82, 804]
    assert nest[2] <= nest[3] <= nest[4]


def test_nested_cluster_uses_one_patch_and_has_the_requested_shells():
    W, ranges, order = nested_cluster((6, 5), b=1.0)
    assert W.shape == (11, 11) and [len(r) for r in ranges] == [6, 5]
    assert np.allclose(W, W.T) and len(set(map(frozenset, order))) == 11
    # the first shell is the six pairs of tier 2: their couplings are those of the central tier-2 pairs
    # up to the (common) rigid motion, so the sorted off-diagonal values agree
    W2 = ph.central_couplings(2, 6, 1.0)
    a = np.sort(np.abs(W[np.ix_(range(6), range(6))][np.triu_indices(6, 1)]))
    c = np.sort(np.abs(W2[np.triu_indices(6, 1)]))
    assert np.allclose(a, c, atol=1e-9)


def test_shell_entanglement_is_nonnegative_and_the_shuffled_control_runs():
    fp = ph.flow_parameters(0.2)
    W, ranges, _ = nested_cluster((6, 4), fp.b)
    r = shell_entanglement(W, ranges, fp.J, fp.kappa, broken=False)
    assert r["I"][0, 1] >= -1e-9 and r["S_shell"][0] >= 0.0
    c = shuffled_control(W, ranges, fp.J, fp.kappa, 0, broken=False)
    assert c["I"][0, 1] >= -1e-9


def run_tests(verbose: bool = True) -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            if verbose:
                print(f"  {name:<70} PASS")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  {name:<70} FAIL: {exc!r}")
    print(f"{len(tests)} test(s), {failed} failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(run_tests())
